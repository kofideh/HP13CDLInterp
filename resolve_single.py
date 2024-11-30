import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model256_perfFocus import build_superresolution_model, visualize_intermediate_outputs
import nibabel as nib
import time
import argparse




def superresolve_image(low_res_img, anat_img, model_path):


    # Preprocess the images to match model input expectations
    low_res_img = np.expand_dims(low_res_img, axis=-1)  # Add channel dimension
    anat_img = np.expand_dims(anat_img, axis=-1)        # Add channel dimension

    # Expand batch dimension
    low_res_img = np.expand_dims(low_res_img, axis=0)
    anat_img = np.expand_dims(anat_img, axis=0)

    # Load model and weights
    # Initialize the model and load weights
    model = build_superresolution_model()
    model.load_weights(model_path)  # Replace with your HDF5 file path

    # Run model prediction
    superresolved_perf = model.predict([low_res_img, anat_img])[0, :, :, :]  # Remove batch and channel dimensions

    return superresolved_perf
    

def superresolve_perfusion(low_res_path, anat_path, model_path, output_path):
    """
    Super-resolve perfusion data given low-resolution and anatomical images.
    
    Parameters:
    low_res_path (str): Path to the low-resolution perfusion image (NIfTI file).
    anat_path (str): Path to the high-resolution anatomical image (NIfTI file).
    model_path (str): Path to the saved model weights file.
    output_path (str): Path to save the output superresolved perfusion image.
    """
    # Load images
    low_res_img = nib.load(low_res_path).get_fdata()
    anat_img = nib.load(anat_path).get_fdata()

    # Preprocess the images to match model input expectations
    low_res_img = np.expand_dims(low_res_img, axis=-1)  # Add channel dimension
    anat_img = np.expand_dims(anat_img, axis=-1)        # Add channel dimension

    # Expand batch dimension
    low_res_img = np.expand_dims(low_res_img, axis=0)
    anat_img = np.expand_dims(anat_img, axis=0)
    
    sr_perf = superresolve_image(low_res_img, anat_img, model_path, output_path)
    
        # Save output as NIfTI
    output_nifti = nib.Nifti1Image(sr_perf, affine=np.eye(4))
    nib.save(output_nifti, output_path)
    print(f"Superresolved perfusion image saved at {output_path}")




if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Superresolve perfusion data using low-res and anatomical images.")
    parser.add_argument("--low_res_path", type=str, required=True, help="Path to the low-resolution perfusion NIfTI file.")
    parser.add_argument("--anat_path", type=str, required=True, help="Path to the high-resolution anatomical NIfTI file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the superresolved perfusion image.")

    args = parser.parse_args()

    # Run the superresolve function
    superresolve_perfusion(args.low_res_path, args.anat_path, args.model_path, args.output_path)
# Perform superresolution
# superresolved_image = superresolve_perfusion(new_low_res_perf_data, new_high_res_anat_data)
# sr_nifti = nib.Nifti1Image(superresolved_image, np.eye(4))  # Create a NIfTI object for HR
# idx = time.time()
# nib.save(sr_nifti, f"SR_image_{idx}.nii.gz")  # Save HR image
# # Display or save the result
# print("Superresolved perfusion image shape:", superresolved_image.shape)