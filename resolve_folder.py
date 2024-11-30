import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data import load_datasets2d, dispImgs
from resolve_single import superresolve_image
import nibabel as nib
import time
import os
from model256_perfFocus import build_superresolution_model, visualize_intermediate_outputs

# Load the dataset
# anat_dirs = ["C:\Python\PerfSR\HPData\Anat"]
# perf_dirs = ["C:\Python\PerfSR\HPData\Perf"]
perf_dirs = ["C:\Python\TamasAllCases_1"]
anat_dirs = ["C:\Python\\100307_registered_T1w_1"]
low_res_perf_images = []
high_res_perf_images = []
high_res_anat_images = []
file_paths = []
snr_level = None
for p, a in zip(perf_dirs, anat_dirs):
    lrp, hrp, hra, paths = load_datasets2d(p, a, lr=(16, 16, 1), hr=(256, 256, 1), ar=(256,256,1), degrad=None)
    low_res_perf_images.extend(lrp)
    high_res_perf_images.extend(hrp)
    high_res_anat_images.extend(hra)
    file_paths.extend(paths)
    # snr_level += 5

# output_path = "C:\Python\PerfSR\PerfSR_2D_Healthy"
model_path = 'PerfSR256_perffocusSRGAN_generator_weights_100.h5'
# model_path = 'CNN_model_256_lowSNR_Healthy.h5'
output_path = f"{os.path.splitext(model_path )[0]}"
# output_path = 'CNN_model_256_perfFocus'
os.makedirs(f"{output_path}/SR", exist_ok=True)
os.makedirs(f"{output_path}/HR", exist_ok=True)
os.makedirs(f"{output_path}/LR", exist_ok=True)
os.makedirs(f"{output_path}/Anat", exist_ok=True)
os.makedirs(f"{output_path}/Plot", exist_ok=True)

low_res_perf_images = np.array(low_res_perf_images)  # Placeholder for low-resolution perfusion data
high_res_anat_images = np.array(high_res_anat_images)   # Placeholder for high-resolution anatomical data
high_res_perf_images = np.array(high_res_perf_images)     # Placeholder for target high-resolution perfusion data

# Example inputs for visualization
low_res_input_example = low_res_perf_images[10,:,:]  # Your low-res input example
high_res_anat_input_example = high_res_anat_images[10,:,:]  # Your high-res anatomical input example

low_res_input_example = low_res_input_example.reshape((1, 16, 16, 1))
high_res_anat_input_example = high_res_anat_input_example.reshape((1, 256, 256, 1))
# # Visualize intermediate outputs
# model_path = "generator_weights_900.h5"
model = build_superresolution_model(output_size=(256, 256))
model.load_weights(model_path)  # Replace with your HDF5 file path
visualize_intermediate_outputs(model, [low_res_input_example, high_res_anat_input_example])

for l,h, a, f in zip(low_res_perf_images, high_res_perf_images, high_res_anat_images, file_paths):
    sr_perf = superresolve_image(l, a, model_path=model_path)
    
    dispImgs(l, h, a, sr_perf)
    
    # Save output as NIfTI
    slice = f["slice"] + 1
    sr_nifti = nib.Nifti1Image(sr_perf, affine=np.eye(4))
    file_name = os.path.splitext(os.path.basename(f["perfusion"]))[0]
    nib.save(sr_nifti, f"{output_path}\SR\{file_name}_slice{slice}.nii.gz")  # Save SR image
    
    # Save HR image as NIfTI file
    # Create a NIfTI object for HR
    hr_nifti = nib.Nifti1Image(h, np.eye(4))
    nib.save(hr_nifti, f"{output_path}\HR\{file_name}_slice{slice}.nii.gz")  # Save HR image

    # Save LR image as NIfTI file
    lr_nifti = nib.Nifti1Image(l, np.eye(4))  # Create a NIfTI object for LR
    nib.save(lr_nifti, f"{output_path}\LR\{file_name}_slice{slice}.nii.gz")  # Save LR image

    # Save Anat image as NIfTI file
    anat_nifti = nib.Nifti1Image(a, np.eye(4))  # Create a NIfTI object for Anat
    file_name = os.path.splitext(os.path.basename(f["perfusion"]))[0]
    nib.save(anat_nifti, f"{output_path}\Anat\{file_name}_slice{slice}.nii.gz")  # Save Anat image
    




