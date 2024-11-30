import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter


def apply_random_signal_loss(image, loss_fraction=0.1):
    mask = np.ones(image.shape)
    # Randomly set a fraction of the image to zero
    num_loss_pixels = int(np.prod(image.shape) * loss_fraction)
    x_coords = np.random.randint(0, image.shape[0], num_loss_pixels)
    y_coords = np.random.randint(0, image.shape[1], num_loss_pixels)
    if image.ndim == 3:
        z_coords = np.random.randint(0, image.shape[2], num_loss_pixels)
        mask[x_coords, y_coords, z_coords] = 0  # Set to zero to simulate signal loss
    else:
        mask[x_coords, y_coords] = 0  # Set to zero to simulate signal loss
    return image * mask




import numpy.fft as fft

def apply_frequency_domain_signal_loss(image, mask_fraction=0.2):
    k_space = fft.fftn(image)
    # Generate a mask to zero out a fraction of k-space
    mask = np.random.rand(*k_space.shape) > mask_fraction
    k_space *= mask  # Apply mask
    degraded_image = np.abs(fft.ifftn(k_space))
    return degraded_image


def apply_coil_sensitivity_dropoff(image, dropoff_rate=0.8):
    # Create a 2D sensitivity map
    height, width = image.shape[:2]
    sensitivity_map = np.fromfunction(
        lambda i, j: dropoff_rate ** (np.sqrt((i - height // 2) ** 2 + (j - width // 2) ** 2) / max(height, width)),
        (height, width)
    )

    # Expand sensitivity map to match the image shape if necessary
    if image.ndim == 3:  # For 3D image (e.g., height x width x slices)
        sensitivity_map = np.expand_dims(sensitivity_map, axis=-1)

    # Apply coil sensitivity map to each slice/channel in the image
    degraded_image = image * sensitivity_map

    return degraded_image



def simulate_flow_void(image=(16, 16,24)):
    """
    Simulates a flow void in a 3D MRI perfusion image.

    Parameters:
    image (numpy.ndarray): 3D numpy array representing the MRI image.
    center (tuple): The (x, y, z) coordinates of the center of the flow void.
    radius (int): The radius of the flow void.

    Returns:
    numpy.ndarray: The modified image with the flow void.
    """
    center = random.randint(0, image.shape[0]), random.randint(0, image.shape[1]), random.randint(0, image.shape[2])
    x_center, y_center, z_center = center
    radius = random.randint(1, 5)
    x, y, z = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
    mask = (x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2 <= radius**2
    image[mask] = 0  # Assuming flow void is represented by zero intensity
    return image


from scipy.ndimage import gaussian_filter

def apply_gaussian_signal_loss(image, num_drops=5, intensity=0.7, sigma=5):
    degraded_image = image.copy()
    for _ in range(num_drops):
        x = np.random.randint(0, image.shape[0])
        y = np.random.randint(0, image.shape[1])
        drop_mask = np.zeros_like(image)
        drop_mask[x, y] = intensity  # Localized high-intensity drop
        blurred_drop = gaussian_filter(drop_mask, sigma=sigma)
        degraded_image -= blurred_drop  # Subtract the dropout
    return np.clip(degraded_image, 0, 1)  # Keep values in range


def simulate_motion(image, max_shift=5):
    # Shift each slice randomly up to max_shift pixels
    shifted_image = np.copy(image)
    for i in range(image.shape[2]):  # Assuming axial slices along the last dimension
        shift = np.random.randint(-max_shift, max_shift, size=2)
        shifted_image[..., i] = np.roll(image[..., i], shift, axis=(0, 1))
    return shifted_image


# def add_rician_noise(image, snr):
#     noise = np.random.rayleigh(scale=1/snr, size=image.shape)
#     noisy_image = np.sqrt((image + noise) ** 2 + noise ** 2)
#     return noisy_image

def add_rician_noise(image, snr):
    """
    Adds Rician noise to a 3D MRI image at a specified SNR level.

    Parameters:
    image (numpy array): 3D input image to which noise is to be added.
    snr (float): Desired signal-to-noise ratio (SNR) level.

    Returns:
    numpy array: 3D image with added Rician noise.
    """
    # Compute the signal power
    signal_power = np.mean(image ** 2)
    
    # Compute noise standard deviation to achieve desired SNR
    sigma = np.sqrt(signal_power / (2 * snr))

    # Generate Gaussian noise components
    noise_real = np.random.normal(0, sigma, image.shape)
    noise_imag = np.random.normal(0, sigma, image.shape)

    # Apply Rician noise model
    noisy_image = np.sqrt((image + noise_real) ** 2 + noise_imag ** 2)
    return np.clip(noisy_image, 0, 1)


def add_gaussian_noise(image, noise_std):
    """
    Adds Gaussian noise to a 3D MRI image with a specified standard deviation.

    Parameters:
    image (numpy array): 3D input image to which noise is to be added.
    noise_std (float): Standard deviation of the Gaussian noise to add.

    Returns:
    numpy array: 3D image with added Gaussian noise.
    """
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, image.shape)
    
    # Add noise to the image
    noisy_image = image + noise
    return noisy_image




def add_noise_to_2d_image(image, noise_std):
    """
    Adds Gaussian noise to a 2D image with a specified standard deviation.
    
    Parameters:
        input_filename (str): Path to the input 2D image.
        output_filename (str): Path where the noisy image will be saved.
        noise_std (float): Standard deviation of the Gaussian noise to add.
    """
    # Convert image to float32 for more precise noise addition
    image = image.astype(np.float32)
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, image.shape)
    
   # Add noise to the image and clip to stay within valid range (0-1)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    
    noisy_image = (noisy_image * 255).astype(np.uint8)


    return noisy_image



def apply_gaussian_blur(image, sigma=1):
    return gaussian_filter(image, sigma=sigma)



def zero_out_irregular_blob(image, blob_size=50, sigma=10):
    """
    Zero out an irregular blob-like region in an image.
    
    Parameters:
    image (numpy.ndarray): The input image.
    blob_size (int): Size of the blob.
    sigma (float): Standard deviation for Gaussian kernel.
    
    Returns:
    numpy.ndarray: The image with the zeroed-out region.
    """
    img_height, img_width, img_depth = image.shape[:3]
    
    # Create a random noise mask
    noise = np.random.rand(img_height, img_width, img_depth)
    
    # Apply Gaussian filter to create a smooth blob
    blob_mask = gaussian_filter(noise, sigma=sigma)
    
    # Threshold the blob mask to create a binary mask
    threshold = np.percentile(blob_mask, 100 - blob_size)
    binary_mask = blob_mask > threshold
    
    # Apply the mask to the image
    image[binary_mask] = 0
    
    return image

