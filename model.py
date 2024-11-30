import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Input, Concatenate, GlobalAveragePooling3D, Dense, Reshape, UpSampling3D

def build_superresolution_model():
    # Define the low-resolution perfusion input
    perf_input = Input(shape=(24, 24, 24, 1), name='low_res_perf_input')

    # Define the high-resolution anatomical input with arbitrary shape
    anat_input = Input(shape=(None, None, None, 1), name='high_res_anat_input')

    # Downsample the anatomical input using global average pooling
    anat_pooled = GlobalAveragePooling3D()(anat_input)  # Produces a 1D vector
    anat_features = Dense(64, activation='relu')(anat_pooled)  # Fully connected layer to produce feature vector
    anat_features = Dense(24 * 24 * 24, activation='relu')(anat_features)  # Match size for concatenation
    anat_features = Reshape((24, 24, 24, 1))(anat_features)  # Reshape to (24, 24, 24, 1)

    # Process low-res perfusion input with Conv3D layers
    x = Conv3D(64, kernel_size=3, padding='same', activation='relu')(perf_input)
    x = Conv3D(64, kernel_size=3, padding='same', activation='relu')(x)

    # Concatenate the anatomical features with the low-res perfusion input
    x = Concatenate()([x, anat_features])

    # Additional convolution layers for superresolution
    x = Conv3D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv3D(128, kernel_size=3, padding='same', activation='relu')(x)

    # Final Conv3D layer to generate the high-resolution perfusion output
    high_res_output = Conv3D(1, kernel_size=3, padding='same', activation='linear')(x)

    # Upsample the output to match the high-resolution target shape (96, 96, 24)
    high_res_output = UpSampling3D(size=(4, 4, 1), name='high_res_perf_output')(high_res_output)

    # Build and return the model
    model = tf.keras.models.Model(inputs=[perf_input, anat_input], outputs=high_res_output)

    return model

# Build and display the model
# model = build_superresolution_model()
# model.summary()
