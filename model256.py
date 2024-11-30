import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Concatenate, Add, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def residual_block_2d(x, filters=64, kernel_size=(3, 3)):
    skip = x
    x = Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = Add()([x, skip])  # Skip connection
    return x

def build_superresolution_model(output_size=(256, 256)):
    # Input layers for low-res perfusion and high-res anatomical images
    low_res_input = Input(shape=(16, 16, 1), name="low_res_input")
    high_res_anat_input = Input(shape=(256, 256, 1), name="high_res_anat_input")
    
    # Process low-res perfusion input with convolutional layers and residual blocks
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(low_res_input)
    for _ in range(4):
        x = residual_block_2d(x)

    # Downsample high-res anatomical input to match low-res input
    y = Conv2D(64, (3, 3), strides=(16, 16), padding="same", activation="relu")(high_res_anat_input)  # down to (16, 16)
    for _ in range(4):
        y = residual_block_2d(y)

    # Concatenate the processed features
    combined = Concatenate()([x, y])
    
    # Upsampling layers to reach the target output size
    combined = UpSampling2D(size=(2, 2))(combined)  # up to 32x32
    combined = Conv2D(128, (3, 3), padding="same", activation="relu")(combined)
    combined = UpSampling2D(size=(2, 2))(combined)  # up to 64x64
    combined = Conv2D(128, (3, 3), padding="same", activation="relu")(combined)
    combined = UpSampling2D(size=(2, 2))(combined)  # up to 128x128
    combined = Conv2D(128, (3, 3), padding="same", activation="relu")(combined)
    combined = UpSampling2D(size=(2, 2))(combined)  # up to 256x256
    combined = Conv2D(128, (3, 3), padding="same", activation="relu")(combined)
    
    # Final output layer
    # output = Conv2D(1, (3, 3), padding="same", activation="sigmoid")(combined)
    # output = Conv2D(1, (3, 3), padding="same", activation="linear", name="high_res_perf_output")(combined)
    output = Conv2D(1, (3, 3), padding="same", activation="tanh", name="high_res_perf_output")(combined)

    # Create the model
    model = Model(inputs=[low_res_input, high_res_anat_input], outputs=output)
    
    return model


import matplotlib.pyplot as plt

# Function to visualize intermediate outputs
def visualize_intermediate_outputs(model, inputs):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(inputs)
    
    for i, activation in enumerate(activations):
        if len(activation.shape) == 4:  # Only visualize 4D outputs (batch, height, width, channels)
            plt.figure(figsize=(15, 15))
            for j in range(min(activation.shape[-1], 16)):  # Visualize up to 16 feature maps
                plt.subplot(4, 4, j + 1)
                plt.imshow(activation[0, :, :, j], cmap='viridis')
                plt.axis('off')
            plt.suptitle(f'Layer {i + 1} - {model.layers[i].name}')
            plt.show(block=False)  # Block execution until the plot window is closed
