import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data import load_datasets2d
from resolve_single import superresolve_image
import nibabel as nib
import time
import os
import skimage.transform

# from model256 import build_superresolution_model, visualize_intermediate_outputs

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Switch to Agg backend, which doesn't require GUI dependencies

import matplotlib.pyplot as plt

def line_profile(image, height):
    """
    Produce the line profile for an image at a specified height.

    Parameters:
    image (numpy.ndarray): The input image.
    height (int): The height at which to extract the line profile.

    Returns:
    numpy.ndarray: The line profile.
    """
    if height < 0 or height >= image.shape[0]:
        raise ValueError("Height is out of bounds of the image dimensions.")
    
    return image[height, :]



def dispImgs(LR, HR, Anat, SR=None, title=None, savePath=None, cmap='gray'):    

    # Save all images in HR, LR, and Anat
    pt = 1
    if savePath is None:
        savePath = os.getcwd()

    fig, axes = plt.subplots(2, 4, figsize=(15, 5))
    
     # Display the LR image
    # Display the first channel (e.g., channel 0)
    if LR.ndim == 2:
        axes[0,0].imshow(LR, cmap=cmap)
    else:
        axes[0].imshow(LR[:, :, int(LR.shape[2]/2)], cmap=cmap)
    if title is None:
        axes[0].set_title("Low-Resolution Image")
    else:
        axes[0,0].set_title(title)
    axes[0,0].axis('off')
    
    height = LR.shape[0] // 2
    profile = line_profile(LR, height)
    # Plot the line profile
    axes[1,0].plot(profile)
    axes[1,0].set_title(f'Line Profile at Height {height}')
    axes[1,0].set_xlabel('Width')
    axes[1,0].set_ylabel('Intensity')
    
    
    # Display the HR image
    # Display the first channel (e.g., channel 0)
    if HR.ndim == 2:
        axes[0,1].imshow(HR, cmap=cmap)
    else:
        axes[1].imshow(HR[:, :, int(HR.shape[2]/2)], cmap=cmap)
    if title is None:
        axes[0,1].set_title("High-Resolution Image")
    else:
        axes[0,1].set_title(title)
    axes[0,1].axis('off')
    
    height = HR.shape[0] // 2
    profile = line_profile(HR, height)
    # Plot the line profile
    axes[1,1].plot(profile)
    axes[1,1].set_title(f'Line Profile at Height {height}')
    axes[1,1].set_xlabel('Width')
    axes[1,1].set_ylabel('Intensity')

    if SR is not None:
        # Display the SR image
        # Display the first channel (e.g., channel 0)
        if SR.ndim == 2:
            axes[0,2].imshow(SR, cmap='gray')
        else:
            axes[0,2].imshow(SR[:, :, int(SR.shape[2]/2)], cmap=cmap)
        if title is None:
            axes[0,2].set_title("Super-Resolved Image")  
            
        height = SR.shape[0] // 2
        profile = line_profile(SR, height)
        # Plot the line profile
        axes[1,2].plot(profile)
        axes[1,2].set_title(f'Line Profile at Height {height}')
        axes[1,2].set_xlabel('Width')
        axes[1,2].set_ylabel('Intensity')


    # Display the Anatomical image
    # Display the first channel (e.g., channel 0)
    if Anat.ndim == 2:
        axes[0,3].imshow(Anat, cmap='gray')
    else:
        axes[3].imshow(Anat[:, :, int(Anat.shape[2]/2)], cmap='gray')
    if title is None:
        axes[0,3].set_title("Anatomical Image")
    else:
        axes[0,3].set_title(title)
    axes[0,3].axis('off')
    height = Anat.shape[0] // 2
    profile = line_profile(Anat, height)
    # Plot the line profile
    axes[1,3].plot(profile)
    axes[1,3].set_title(f'Line Profile at Height {height}')
    axes[1,3].set_xlabel('Width')
    axes[1,3].set_ylabel('Intensity')

    hr_nifti = nib.Nifti1Image(HR, np.eye(4))
    nib.save(hr_nifti, f"{savePath}/HR/HR_image_{title}.nii.gz")  # Save HR image

    # Save LR image as NIfTI file
    lr_nifti = nib.Nifti1Image(LR, np.eye(4))  # Create a NIfTI object for LR
    nib.save(lr_nifti, f"{savePath}/LR/LR_image_{title}.nii.gz")  # Save LR image

    # Save Anat image as NIfTI file
    anat_nifti = nib.Nifti1Image(Anat, np.eye(4))  # Create a NIfTI object for Anat
    nib.save(anat_nifti, f"{savePath}/Anat/Anat_image_{title}.nii.gz")  # Save Anat image
    
    
    # Save SR image as NIfTI file
    if SR is not None:
        sr_nifti = nib.Nifti1Image(SR, np.eye(4))  # Create a NIfTI object for Anat
        nib.save(sr_nifti, f"{savePath}/SR/SR_image_{title}.nii.gz")  # Save Anat image
    
    plt.savefig(f"{savePath}/Plot/plot_image_{title}.png")
    plt.close(fig)



# Load the dataset
perf_dirs = ["InputPerf"]
anat_dirs = ["InputAnat"]
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


model_path = "C:\Python\HP13CDLInterp\weights\CNNInterp_16to256_perfWeighted_epoch_250.h5" # Path to SRGAN model weights file
model_path = "C:\Python\HP13CDLInterp\weights\SRGAN_generator_weights_800.h5" # Path to SRGAN model weights file

model_name = f"{os.path.splitext( os.path.basename(model_path))[0]}"
output_path = f"results/{model_name}"
# output_path = 'CNN_model_256_perfFocus'
os.makedirs(f"{output_path}/SR", exist_ok=True)
os.makedirs(f"{output_path}/HR", exist_ok=True)
os.makedirs(f"{output_path}/LR", exist_ok=True)
os.makedirs(f"{output_path}/Anat", exist_ok=True)
os.makedirs(f"{output_path}/Plot", exist_ok=True)

low_res_perf_images = np.array(low_res_perf_images)  # Placeholder for low-resolution perfusion data
high_res_anat_images = np.array(high_res_anat_images)   # Placeholder for high-resolution anatomical data
high_res_perf_images = np.array(high_res_perf_images)     # Placeholder for target high-resolution perfusion data


for l,h, a, f in zip(low_res_perf_images, high_res_perf_images, high_res_anat_images, file_paths):
    sr_perf = superresolve_image(l, a, model_path=model_path)
    
    slice = f["slice"] + 1
    file_name = os.path.splitext(os.path.basename(f["perfusion"]))[0]
    
      # Scale the image to 256 by 256
    target_shape = (256, 256)
    b = skimage.transform.resize(l, target_shape, order = 3)
    dispImgs(np.rot90(l, k=-1), np.rot90(b, k=-1), np.rot90(a, k=-1), np.rot90(sr_perf, k=-1), title=f"{file_name}_{slice}", savePath=output_path, cmap='jet')

    


