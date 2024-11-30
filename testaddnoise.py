import cv2
import numpy as np
import nibabel as nib

# Read the image
#img = cv2.imread('image.jpg')
perf_path = "C:\Python\TamasAllCases\\040EPAD00001_healthy_perfusion_0.5x0.5x3.0.nii.gz"
img = np.squeeze(nib.load(perf_path).get_fdata())

for i in range(img.shape[2]):
    # Generate Gaussian noise
    mean = 0
    var = 100  # Adjust the variance to control the noise level
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img.shape)

    # Add the noise to the image
    noisy_image = img + gaussian

    # Clip the pixel values to be between 0 and 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Display the images
    cv2.imshow('Original Image', img)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()