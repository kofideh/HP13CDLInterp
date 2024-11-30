from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from data import load_nifti_image
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_laplace

import numpy as np
from numpy.fft import fftn, ifftn, fftshift

def low_pass_filter_3d(image, cutoff_freq=0.1):
    # Compute 3D Fourier transform
    fft_image = fftn(image)
    fft_image_shifted = fftshift(fft_image)
    
    # Create a low-pass filter mask
    x, y, z = np.meshgrid(
        np.fft.fftfreq(image.shape[0]),
        np.fft.fftfreq(image.shape[1]),
        np.fft.fftfreq(image.shape[2]),
        indexing="ij"
    )
    radius = np.sqrt(x**2 + y**2 + z**2)
    mask = radius < cutoff_freq
    
    # Apply the low-pass filter mask
    fft_image_shifted *= mask
    
    # Inverse transform to get filtered image
    filtered_image = ifftn(fftshift(fft_image_shifted)).real
    return filtered_image


def apply_mexican_hat(image, sigma=1.0):
    return -gaussian_laplace(image, sigma=sigma)  # Invert for a positive Mexican hat

# Assuming `carbon13_image` and `proton_image` are 2D numpy arrays
def estimate_psf(carbon13_image, sigma_guess=0.1):
    # Apply a Gaussian filter to simulate PSF; you can adjust `sigma_guess` based on visual inspection
    psf = gaussian_filter(carbon13_image, sigma=sigma_guess)
    return psf

def estimate_3d_psf(carbon13_image, sigma_guess=0.1):
    # Ensure sigma is either a single value or a tuple with three elements
    if isinstance(sigma_guess, (int, float)):
        sigma_guess = (sigma_guess, sigma_guess, sigma_guess)
    # Apply a Gaussian filter to simulate PSF in 3D
    psf = gaussian_filter(carbon13_image, sigma=sigma_guess)
    return psf

def unsharp_mask(image, amount=1.0, sigma=1.0):
    blurred = gaussian_filter(image, sigma=sigma)
    sharpened = image + amount * (image - blurred)
    return sharpened

def apply_psf_2d_slices(proton_image, psf_2d):
    # Apply the 2D PSF to each slice independently
    blurred_image = np.zeros_like(proton_image)
    for i in range(proton_image.shape[2]):  # Loop over slices
        blurred_image[:, :, i] = fftconvolve(proton_image[:, :, i], psf_2d, mode='same')
    return blurred_image


# Usage
carbon13_image = nib.load('D20P5_pyr_24x24.nii').get_fdata() # Replace with actual data
carbon13_image = np.squeeze(carbon13_image)
carbon13_psf = estimate_3d_psf(carbon13_image)

proton_image = nib.load('LR_image_1729906126.872631.nii.gz').get_fdata()  # Replace with actual data
# Apply low-pass filtering to the proton image
filtered_proton_image = low_pass_filter_3d(proton_image, cutoff_freq=0.05)

def apply_3d_psf(proton_image, psf):
    # Apply PSF in 3D using convolution to simulate blur
    blurred_proton_image = fftconvolve(proton_image, psf, mode='same')
    return blurred_proton_image


def normalize_3d_image(image, original):
    min_orig, max_orig = original.min(), original.max()
    min_image, max_image = image.min(), image.max()
    normalized_image = (image - min_image) / (max_image - min_image)
    normalized_image = normalized_image * (max_orig - min_orig) + min_orig
    return normalized_image


# Usage
blurred_proton_image = apply_3d_psf(proton_image, carbon13_psf)
blurred_proton_image = apply_3d_psf(filtered_proton_image, carbon13_psf)


# Estimate 2D PSF and apply it to each slice
psf_2d = estimate_psf(carbon13_image[:, :, carbon13_image.shape[2] // 2])  # Use a middle slice for PSF estimate
blurred_proton_image = apply_psf_2d_slices(proton_image, psf_2d)

# Apply Mexican hat filter instead of Gaussian blur
mexican_hat_psf = apply_mexican_hat(carbon13_image, sigma=1.0)
blurred_proton_image_mexican = fftconvolve(proton_image, mexican_hat_psf, mode='same')

final_proton_image = normalize_3d_image(blurred_proton_image, proton_image)
sharpened_proton_image = unsharp_mask(final_proton_image, amount=1.5, sigma=0.5)
bl_nifti = nib.Nifti1Image(sharpened_proton_image , np.eye(4))  
nib.save(bl_nifti, f"sharpened_proton_image.nii.gz")  # Save HR image
