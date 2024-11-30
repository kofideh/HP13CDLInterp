from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


# Define the discriminator model
def build_discriminator(input_shape=(256, 256, 1)):
    input_img = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(input_img)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(512, (3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(input_img, x)
    return model



def build_srgan(generator, discriminator):
    # Combine generator and discriminator in the SRGAN
    discriminator.trainable = False
    low_res_input, high_res_anat_input = generator.inputs
    high_res_output = generator([low_res_input, high_res_anat_input])
    valid = discriminator(high_res_output)
    gan_model = Model([low_res_input, high_res_anat_input], [high_res_output, valid])
    return gan_model


# # Compile the discriminator
# discriminator = build_discriminator()
# discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss=BinaryCrossentropy(), metrics=['accuracy'])

# # Build the generator (already defined as build_superresolution_model)
# generator = build_superresolution_model()

# Define the combined GAN model
# low_res_input = Input(shape=(16, 16, 1), name="low_res_input")
# high_res_anat_input = Input(shape=(256, 256, 1), name="high_res_anat_input")

# generated_high_res = generator([low_res_input, high_res_anat_input])
# discriminator.trainable = False
# validity = discriminator(generated_high_res)

# gan_model = Model([low_res_input, high_res_anat_input], validity)
# gan_model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss=BinaryCrossentropy())

# Summary of models
# generator.summary()
# discriminator.summary()
# gan_model.summary()