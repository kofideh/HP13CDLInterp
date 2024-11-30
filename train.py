import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# from model_2d import build_superresolution_model
from model256_perfFocus import build_superresolution_model
from data import load_datasets2d
from tensorflow.keras.losses import MeanSquaredError
import platform


# perf_dirs = ["C:\Python\TamasAllCases_1"]
# anat_dirs = ["C:\Python\\100307_registered_T1w_1"]
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
    lrp, hrp, hra, _ = load_datasets2d(p, a, lr=(16, 16, 1), hr=(256, 256, 1), ar=(256,256,1), degrad=True)
    low_res_perf_images.extend(lrp)
    high_res_perf_images.extend(hrp)
    high_res_anat_images.extend(hra)

# Convert the lists to numpy arrays
low_res_perf_images = np.array(low_res_perf_images)
high_res_perf_images = np.array(high_res_perf_images)
high_res_anat_images = np.array(high_res_anat_images)

print(f"LR:HR:Anat Shapes: {low_res_perf_images.shape}, {high_res_perf_images.shape}, {high_res_anat_images.shape}")

# Split the dataset into training and validation sets
# low_res_perf_images, high_res_perf_images, high_res_anat_images = load_images()
x_train_perf, x_val_perf, y_train_perf, y_val_perf = train_test_split(
    low_res_perf_images, high_res_perf_images, test_size=0.2, random_state=42)
x_train_anat, x_val_anat = train_test_split(high_res_anat_images, test_size=0.2, random_state=42)

# Build and display the model
model = build_superresolution_model(output_size=(256, 256))
# model.summary()

# # Compile the model with loss and optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
# Compile the model with the custom loss function and optimizer
#model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mae'])


# Training parameters
batch_size = 8
epochs = 250

from tensorflow.keras.callbacks import ModelCheckpoint
# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint_epoch_{epoch:02d}.h5',  # Save model with epoch number in filename
    save_weights_only=False,  # Save the entire model (architecture + weights)
    save_freq='epoch',  # Save at the end of every epoch
    monitor='val_loss',  # Monitor validation loss
    save_best_only=True,  # Save only the best model based on the monitored metric
    verbose=1  # Print messages when saving the model
)

# Train the model
history = model.fit(
    [x_train_perf, x_train_anat], y_train_perf,  # Inputs are perfusion and anatomical images; output is perfusion
    validation_data=([x_val_perf, x_val_anat], y_val_perf),
    batch_size=batch_size,
    epochs=epochs, 
    steps_per_epoch=None
)

#model_edsr.fit(train_ds, epochs=300, steps_per_epoch=1000)

# model.fit(...)

# Example inputs for visualization
# low_res_input_example = low_res_perf_images[10,:,:]  # Your low-res input example
# high_res_anat_input_example = high_res_anat_images[10,:,:]  # Your high-res anatomical input example

# low_res_input_example = low_res_input_example.reshape((1, 16, 16, 1))
# high_res_anat_input_example = high_res_anat_input_example.reshape((1, 256, 256, 1))
# # Visualize intermediate outputs
# visualize_intermediate_outputs(model, [low_res_input_example, high_res_anat_input_example])


# Save the trained model
model.save('CNN_model_256_perfFocus_lowSNR.h5')

# Optionally, you can plot the training history if you want to visualize it
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('loss.png')
