import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Add, Concatenate
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", activation="relu")(x)
    x = Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = Add()([shortcut, x])
    return x

def build_superresolution_model(input_size=(16, 16, 1), anatomical_size=(256, 256, 1), output_size=(64, 64)):
    # Low-resolution input
    low_res_input = Input(shape=input_size, name="low_res_input")
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(low_res_input)

    # Add residual blocks to low-res input
    for _ in range(4):  # Adding 4 residual blocks
        x = residual_block(x, filters=64)

    # Upsample low-res input progressively to match the anatomical input size (256x256)
    x = UpSampling2D((4, 4))(x)  # Upsample from (16, 16) to (64, 64)
    x = UpSampling2D((4, 4))(x)  # Upsample from (64, 64) to (256, 256)

    # Anatomical input
    anatomical_input = Input(shape=anatomical_size, name="anatomical_input")
    y = Conv2D(64, (3, 3), padding="same", activation="relu")(anatomical_input)
    y = Conv2D(64, (3, 3), padding="same", activation="relu")(y)

    # Combine the two branches
    combined = Concatenate()([x, y])

    # After concatenation, bring down to target output size (64, 64)
    z = Conv2D(64, (3, 3), padding="same", activation="relu")(combined)
    z = Conv2D(64, (3, 3), strides=(4, 4), padding="same", activation="relu")(z)  # Downsample to (64, 64)

    # Final output layer
    high_res_output = Conv2D(1, (3, 3), padding="same", activation="sigmoid", name="high_res_output")(z)

    # Build the model
    model = Model(inputs=[low_res_input, anatomical_input], outputs=high_res_output)
    return model

# Create model instance
# model_with_residuals = build_superresolution_model_with_residual_blocks()
# model_with_residuals.compile(optimizer="adam", loss="mean_squared_error")
# model_with_residuals.summary()
