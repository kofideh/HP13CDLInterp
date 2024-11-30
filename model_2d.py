import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Concatenate, Add, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import graphviz
import pydot


def residual_block_2d(x, filters=64, kernel_size=(3, 3)):
    skip = x
    x = Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = Add()([x, skip])  # Skip connection
    return x

def build_superresolution_model(output_size=(64, 64)):
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
    
    # Final output layer for super-resolved perfusion image
    high_res_perf_output = Conv2D(1, (3, 3), padding="same", activation="linear", name="high_res_perf_output")(combined)
    
    # Model definition
    model = Model(inputs=[low_res_input, high_res_anat_input], outputs=high_res_perf_output)
    return model

