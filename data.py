import os
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend like 'Agg' to prevent windows
import matplotlib.pyplot as plt
import random
import nibabel as nib
import time
from scipy.ndimage import rotate
import cv2
import image_degrade as degrade   
from mri_inhomogeneity_small import apply_geometric_distortion, create_field_inhomogeneity
import platform


random.seed(123456789)
epsilon = 1e-10  # Small value to avoid division by zero


import numpy as np
import cv2
from scipy.ndimage import sobel

def check_image_structure(image, edge_threshold=0.1, intensity_threshold=0.1):
    """
    Determine if a 2D MRI slice has structure or is just noise.
    
    Parameters:
    - image (2D array): Input MRI slice.
    - edge_threshold (float): Minimum fraction of edge pixels needed to consider the image structured.
    - intensity_threshold (float): Minimum standard deviation for pixel intensities to consider the image structured.
    
    Returns:
    - bool: True if the image likely has structure, False if it's likely noise.
    """
    
    # Check histogram for peaks (structured images usually have a few prominent intensity peaks)
    hist, _ = np.histogram(image, bins=50, range=(np.min(image), np.max(image)))
    peak_count = np.sum(hist > hist.max() * 0.1)
    return peak_count   # Returns True if structured; otherwise False



def get_kspace(phantom, noise_level):
    """Simulate MRI acquisition."""
    size = phantom.shape
    # Normalize the phantom to avoid large values
    if np.isnan(phantom).any() or np.isinf(phantom).any():
        phantom = np.nan_to_num(phantom, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Normalize the phantom array
    max_value = np.max(np.abs(phantom))
    phantom = phantom / (max_value + epsilon)

    # Compute k-space
    k_space = np.fft.fftshift(np.fft.fftn(phantom))

    # Add noise
    noise = np.random.normal(0, noise_level, size=size) + \
            1j * np.random.normal(0, noise_level, size=size)

    k_space += noise
    return k_space


def reconstruct_image(k_space):
    """Reconstruct image from k-space data using inverse Fourier transform."""
    return np.abs(np.fft.ifftn(np.fft.ifftshift(k_space)))

import numpy as np

def crop_kspace(k_space, target_size):
    """Crop or pad k-space data to specific dimensions."""
    size = k_space.shape
    
    # Determine if padding is needed for each dimension
    pad_width = [(0, 0)] * len(size)
    for i in range(len(size)):
        if size[i] < target_size[i]:  # Pad if smaller
            total_pad = target_size[i] - size[i]
            pad_width[i] = (total_pad // 2, total_pad - total_pad // 2)
    
    # Pad the k-space to the target size if needed
    if any(pad[0] > 0 or pad[1] > 0 for pad in pad_width):
        k_space = np.pad(k_space, pad_width, mode='constant', constant_values=0)
    
    # Calculate cropping indices if necessary
    crop_start = tuple((k_space.shape[i] - target_size[i]) // 2 for i in range(len(size)))
    crop_end = tuple(crop_start[i] + target_size[i] for i in range(len(size)))
    
    # Crop to target size
    if len(target_size) == 2 or target_size[2] == 1:
        return k_space[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1],:]
    return k_space[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]



def load_nifti_image(file_path):
    """
    Load a 3D MRI image from a NIfTI file.
    """
    nifti_image = nib.load(file_path)
    image_data = nifti_image.get_fdata()  # Get the image data as a NumPy array
    return image_data

def load_dataset_from_nifti(data_dir, lr=(24, 24, 24), hr=(96, 96, 96)):
    """
    Load a dataset of 3D MRI images from a directory containing NIfTI files.
    
    Args:
        data_dir (str): Directory containing the NIfTI files.
        image_size (tuple): The desired image size after resizing. 
                            Use (24, 24, 24) for low-res images or (96, 96, 96) for high-res images.
    
    Returns:
        np.array: Array of loaded 3D MRI images.
    """
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    lr_images = []
    hr_images = []
    
    for file_path in image_files:
        image = load_nifti_image(file_path)
        image[image < 0] = 0
        if lr is None and hr is None:
            lr_image = image
            hr_image = image
        else:
            lr_image = reconstruct_image(crop_kspace(get_kspace(image, 0), lr))
            hr_image = reconstruct_image(crop_kspace(get_kspace(image, 0), hr))
        # Resize the image to the desired shape (if needed). Here we use NumPy's resize.
        # For better performance, consider using a library like `scipy.ndimage.zoom` or `opencv`.

        lr_images.append(lr_image/np.max(lr_image))
        hr_images.append(hr_image/np.max(hr_image))
    return hr_images, lr_images



def dispImgs(LR, HR, Anat, SR=None, title=None, savePath=None, cmap='gray'):

    # Save all images in HR, LR, and Anat
    pt = 1
    if savePath is None:
        savePath = os.getcwd()
    
    os.makedirs(f"{savePath}\HR", exist_ok=True)
    os.makedirs(f"{savePath}\LR", exist_ok=True)
    os.makedirs(f"{savePath}\SR", exist_ok=True)
    os.makedirs(f"{savePath}\Anat", exist_ok=True)
    os.makedirs(f"{savePath}\Plot", exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    
     # Display the LR image
    # Display the first channel (e.g., channel 0)
    if LR.ndim == 2:
        axes[0].imshow(LR, cmap=cmap)
    else:
        axes[0].imshow(LR[:, :, int(LR.shape[2]/2)], cmap=cmap)
    if title is None:
        axes[0].set_title("Low-Resolution Image")
    else:
        axes[0].set_title(title)
    axes[0].axis('off')
    
    # Display the HR image
    # Display the first channel (e.g., channel 0)
    if HR.ndim == 2:
        axes[1].imshow(HR, cmap=cmap)
    else:
        axes[1].imshow(HR[:, :, int(HR.shape[2]/2)], cmap=cmap)
    if title is None:
        axes[1].set_title("High-Resolution Image")
    else:
        axes[1].set_title(title)
    axes[1].axis('off')

    if SR is not None:
        # Display the SR image
        # Display the first channel (e.g., channel 0)
        if SR.ndim == 2:
            axes[2].imshow(SR, cmap=cmap)
        else:
            axes[2].imshow(SR[:, :, int(SR.shape[2]/2)], cmap=cmap)
        if title is None:
            axes[2].set_title("Super-Resolved Image")  
    axes[2].axis('off') 

    # Display the Anatomical image
    # Display the first channel (e.g., channel 0)
    if Anat.ndim == 2:
        axes[3].imshow(Anat, cmap=cmap) 
    else:
        axes[3].imshow(Anat[:, :, int(Anat.shape[2]/2)], cmap=cmap)
    if title is None:
        axes[3].set_title("Anatomical Image")
    else:
        axes[3].set_title(title)
    axes[3].axis('off')

    # Show the plot
    # plt.show(block=False)
    # Pause for 3 seconds
    # plt.pause(pt)
    
    # Save HR image as NIfTI file
    # Create a NIfTI object for HR
    if title is None:
        title = time.time() 
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

    # print(f"Saved: HR_image_{idx}.nii.gz, LR_image_{idx}.nii.gz, Anat_image_{idx}.nii.gz")

    plt.savefig(f"{savePath}/Plot/plot_image_{title}.png")
    plt.close(fig)




def load_datasets2d(perf_dir, anat_dir, lr=(16, 16, 1), hr=(128, 128, 1), ar=(256,256,1), degrad=None):
    perf_files = sorted([os.path.join(perf_dir, f) for f in os.listdir(perf_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    anat_files = sorted([os.path.join(anat_dir, f) for f in os.listdir(anat_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    print(f"Perf files: {perf_files}")
    print(f"Anat files: {anat_files}")
    
    low_res_perf_images = []
    high_res_perf_images = []
    high_res_anat_images = []
    file_paths = []
    
    # Load the images using nibabel
    for perf_path, anat_path in zip(perf_files, anat_files):
        perf_img = np.squeeze(nib.load(perf_path).get_fdata())
        perf_img[perf_img < 0] = 0
        if degrad is not None:
            siz = random.choice([0, 10, 25, 50, 100])
            if siz > 0:
                perf_img = degrade.zero_out_irregular_blob(perf_img, blob_size=siz, sigma=10)
            
        low_res_img = reconstruct_image(crop_kspace(get_kspace(perf_img, 0), lr))
        high_res_img = reconstruct_image(crop_kspace(get_kspace(perf_img, 0), hr))
        anat_img = np.squeeze(nib.load(anat_path).get_fdata())
        anat_img[anat_img < 0] = 0
        anat_img = reconstruct_image(crop_kspace(get_kspace(anat_img, 0), ar))
        
        
        
        for i in range(low_res_img.shape[2]):
            
            lr_img = low_res_img[:,:,i]/(np.max(low_res_img[:,:,i]) + epsilon)
            an_img = anat_img[:,:,i]/(np.max(anat_img[:,:,i]) + epsilon)

            c = check_image_structure(an_img)

            if c < 15:
                hr_img = high_res_img[:,:,i]/(np.max(high_res_img[:,:,i]) + epsilon)
                

                     # Apply a random flip along any axis
                if degrad is not None:
                    snr = random.uniform(0.5, 5)
                    lr_img = degrade.add_rician_noise(lr_img, snr)
                    flip_axis = random.choice([0, 1, 2, 3, 4, 5])  # None means no flip
                    if flip_axis < 2:
                        lr_img = np.flip(lr_img, axis=flip_axis)
                        hr_img = np.flip(hr_img, axis=flip_axis)
                        an_img = np.flip(an_img, axis=flip_axis)
                    elif flip_axis < 5:
                        lr_img = np.rot90(lr_img, flip_axis-1)
                        hr_img = np.rot90(hr_img, flip_axis-1)
                        an_img = np.rot90(an_img, flip_axis-1)
                    else:
                        angle_range=(-10, 10)
                        angle = random.uniform(*angle_range)
                        lr_img = rotate(lr_img, angle=angle, reshape=False, mode='nearest')
                        hr_img = rotate(hr_img, angle=angle, reshape=False, mode='nearest')
                        an_img = rotate(an_img, angle=angle, reshape=False, mode='nearest')
                    inhomogeneity = random.choice([0, 1, 2])  
                    pixel_shift = random.uniform(1, 2)
                    fs = random.uniform(0.5, 1.5)
                    if inhomogeneity == 1:
                        field_map = create_field_inhomogeneity(size=lr_img.shape[0], strength=fs, direction='frequency')
                        lr_img = apply_geometric_distortion(lr_img, field_map, pixel_shift_scale=pixel_shift, direction='frequency')
                    elif inhomogeneity == 2:
                        field_map = create_field_inhomogeneity(size=lr_img.shape[0], strength=fs, direction='phase')
                        lr_img = apply_geometric_distortion(lr_img, field_map, pixel_shift_scale=pixel_shift, direction='phase')                
                
                if platform.system() == 'Windows':
                    file_name = os.path.splitext(os.path.basename(perf_path))[0]
                    dispImgs(lr_img, hr_img, an_img, title=f"{file_name}_{i}")

                low_res_perf_images.append(lr_img)
                high_res_perf_images.append(hr_img)
                high_res_anat_images.append(an_img)
                file_paths.append({"perfusion": perf_path, "anatomic": anat_path, "slice": i})


        
    return low_res_perf_images, high_res_perf_images, high_res_anat_images, file_paths






