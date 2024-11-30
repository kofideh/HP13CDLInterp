from tensorflow.keras.layers import Conv2D, Input, Add, UpSampling2D, Concatenate, LeakyReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import time
from discriminator import build_discriminator, build_srgan
from data import load_datasets2d
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend like 'Agg' to prevent windows
import matplotlib.pyplot as plt
from model256_perfFocus import build_superresolution_model
from keras.utils.vis_utils import plot_model
from data import dispImgs
import random
import platform

# Compile models
generator = build_superresolution_model()
discriminator = build_discriminator()
srgan = build_srgan(generator, discriminator)
if os.path.exists('generator_weights.h5'):
    generator.load_weights('generator_weights.h5')
if os.path.exists('discriminator_weights.h5'):
    discriminator.load_weights('discriminator_weights.h5')
if os.path.exists('srgan_weights.h5'):
    srgan.load_weights('srgan_weights.h5')

generator_optimizer = Adam(learning_rate=1e-4)
discriminator_optimizer = Adam(learning_rate=1e-4)
discriminator.compile(optimizer=discriminator_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
srgan.compile(optimizer=generator_optimizer, loss=["mean_squared_error", "binary_crossentropy"], loss_weights=[1, 1e-3])


perf_dirs = ["/lustre/xg-bio240004/PerfSR/TamasAllCases"]*2
anat_dirs = ["/lustre/xg-bio240004/PerfSR/100307_registered_T1w"]*2
if platform.system() == 'Windows':
    perf_dirs = ["C:\Python\TamasAllCases_1"]
    anat_dirs = ["C:\Python\\100307_registered_T1w_1"]
elif platform.system() == 'Linux':
    perf_dirs = ["/lustre/xg-bio240004/PerfSR/TamasAllCases"]*4
    anat_dirs = ["/lustre/xg-bio240004/PerfSR/100307_registered_T1w"]*4
else:
    print(f"Running on {platform.system()}")

low_res_perf_images = []
high_res_perf_images = []
high_res_anat_images = []
for p, a in zip(perf_dirs, anat_dirs):
    lrp, hrp, hra, _ = load_datasets2d(p, a, lr=(16, 16, 1), hr=(256, 256, 1), ar=(256,256,1), degrad=None)
    low_res_perf_images.extend(lrp)
    high_res_perf_images.extend(hrp)
    high_res_anat_images.extend(hra)

# Convert the lists to numpy arrays
low_res_perf_images = np.array(low_res_perf_images)
high_res_perf_images = np.array(high_res_perf_images)
high_res_anat_images = np.array(high_res_anat_images)



# Split dataset into training and validation sets (80% training, 20% validation)
low_res_perf_train, low_res_perf_val, high_res_perf_train, high_res_perf_val, high_res_anat_train, high_res_anat_val = train_test_split(
    low_res_perf_images, high_res_perf_images, high_res_anat_images, test_size=0.2, random_state=42
)


# Custom training loop with validation
epochs = 1000
batch_size = 16
steps_per_epoch = len(low_res_perf_train) // batch_size

# Initialize lists to track loss history
gen_loss_history = []
disc_loss_history = []

def plot_loss(gen_loss_history, disc_loss_history, epoch=0):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_loss_history, label="Generator Loss")
    plt.plot(disc_loss_history, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SRGAN Training Loss History")
    plt.legend()
    plt.savefig('SRGAN_Training_Loss_History{}.png'.format(epoch))
    plt.close()

# Training function (simplified example)
def train_srgan(generator, discriminator, srgan, epochs=1000, batch_size=16):
    for epoch in range(epochs):
        # print(f"\nEpoch {epoch + 1}/{epochs}")
    
        for step in range(steps_per_epoch):
            # Select a batch of random training images
            idx = np.random.randint(0, low_res_perf_train.shape[0], batch_size)
            low_res_imgs = low_res_perf_train[idx]
            high_res_imgs = high_res_perf_train[idx]
            anat_imgs = high_res_anat_train[idx]
        
            # Generate high-resolution images
            gen_imgs = generator.predict([low_res_imgs, anat_imgs])
        
            # Train the discriminator (real high-res -> class 1, generated -> class 0)
            d_loss_real = discriminator.train_on_batch(high_res_imgs, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
            # Train the generator (using adversarial loss)
            g_loss = srgan.train_on_batch([low_res_imgs, anat_imgs], [high_res_imgs, np.ones((batch_size, 1))])
            disc_loss_history.append(d_loss[0])  # Append only the loss value, not metrics
            gen_loss_history.append(g_loss[0])  # Append only the loss value, not metrics
        # Validation at the end of each epoch
        val_gen_imgs = generator.predict([low_res_perf_val, high_res_anat_val])
      
    
                # Print progress
        if epoch % 100 == 0:
            generator.save_weights("generator_weights_{}.h5".format(epoch))
            discriminator.save_weights("discriminator_weights_{}.h5".format(epoch))
            srgan.save_weights("srgan_weights_{}.h5".format(epoch))
            plot_loss(gen_loss_history, disc_loss_history, epoch=epoch)
            idx = random.randrange(0, val_gen_imgs.shape[0])
            dispImgs(low_res_perf_val[idx],  high_res_perf_val[idx], high_res_anat_val[idx], val_gen_imgs[idx])
            
    # Save generator weights
    generator.save_weights("generator_weights_final.h5")

    # Save discriminator weights
    discriminator.save_weights("discriminator_weights_final.h5")

    # Save SRGAN (combined model) weights
    srgan.save_weights("srgan_weights_final.h5")

# Run training
train_srgan(generator, discriminator, srgan, epochs=epochs, batch_size=batch_size)


