import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import skimage.transform
import skimage.metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Function to calculate PSNR
def calculate_psnr(img1, img2):
    return skimage.metrics.peak_signal_noise_ratio(img1, img2)

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    return skimage.metrics.structural_similarity(img1, img2, multichannel=True)


lr_cnn_dir = "C:\Python\HP13CDLInterp\\results\CNNInterp_16to256_perfWeighted_epoch_190\LR"
hr_cnn_dir = "C:\Python\HP13CDLInterp\\results\CNNInterp_16to256_perfWeighted_epoch_190\HR"
sr_cnn_dir = "C:\Python\HP13CDLInterp\\results\CNNInterp_16to256_perfWeighted_epoch_190\SR"
anat_dir = "C:\Python\HP13CDLInterp\\results\CNNInterp_16to256_perfWeighted_epoch_190\Anat"

lr_gan_dir = "C:\Python\HP13CDLInterp\\results\CNNInterp_16to256_perfWeighted_epoch_190\LR"
hr_gan_dir = "C:\Python\HP13CDLInterp\\results\CNNInterp_16to256_perfWeighted_epoch_190\HR"
sr_gan_dir = "C:\Python\HP13CDLInterp\\results\CNNInterp_16to256_perfWeighted_epoch_190\SR"

lr_cnn_f =[]
lr_cnn_f["pyr"] = sorted([os.path.join(lr_cnn_dir, f) for f in os.listdir(lr_cnn_dir) if ('pyr' in f) and (f.endswith('.nii') or f.endswith('.nii.gz'))])
lr_cnn_f["lac"] = sorted([os.path.join(lr_cnn_dir, f) for f in os.listdir(lr_cnn_dir) if ('pyr' in f) and (f.endswith('.nii') or f.endswith('.nii.gz'))])
hr_cnn_f = sorted([os.path.join(hr_cnn_dir, f) for f in os.listdir(hr_cnn_dir) if (f.endswith('.nii') or f.endswith('.nii.gz'))])
sr_cnn_f = sorted([os.path.join(sr_cnn_dir, f) for f in os.listdir(sr_cnn_dir) if (f.endswith('.nii') or f.endswith('.nii.gz'))])

lr_gan_f = sorted([os.path.join(lr_gan_dir, f) for f in os.listdir(lr_gan_dir) if (f.endswith('.nii') or f.endswith('.nii.gz'))])
hr_gan_f = sorted([os.path.join(hr_gan_dir, f) for f in os.listdir(hr_gan_dir) if (f.endswith('.nii') or f.endswith('.nii.gz'))])

anat_f = sorted([os.path.join(anat_dir, f) for f in os.listdir(anat_dir) if (f.endswith('.nii') or f.endswith('.nii.gz'))])

pyr_images = []
lac_images = []
epsilon = 1e-10  # Small value to avoid division by zero
title = ['Acquired HP 13C', 'Bicubic', 'Dichromatic', 'CNN', 'SRGAN', 'Anatomical']

for lcf , hcf, scf, lgf, hgf, atf in zip(lr_cnn_f , hr_cnn_f, sr_cnn_f, lr_gan_f, hr_gan_f, anat_f):
        lcimg = np.squeeze(nib.load(lcf).get_fdata())
        lcimg = np.rot90(lcimg, k=4, axes=(0, 1))
        max_value = np.max(np.abs(lcimg))
        lcimg = lcimg / (max_value + epsilon)
        pyr_images.append(lcimg)  
        
        
        
        lac_img = np.squeeze(nib.load(lac_path).get_fdata())
        lac_img = np.rot90(lac_img, k=4, axes=(0, 1))
        max_value = np.max(np.abs(lac_img))
        lac_img = lac_img / (max_value + epsilon)
        lac_images.append(lac_img)    
    
# Resize the image using bicubic interpolation
target_shape = (256, 256)
bicubic_data = skimage.transform.resize(pyr_images[0], target_shape, order = 3)
pyr_images.insert(1, bicubic_data)  # Duplicate the first image to make the comparison easier

bicubic_data = skimage.transform.resize(lac_images[0], target_shape, order = 3)
lac_images.insert(1, bicubic_data)  # Duplicate the first image to make the comparison easier

fig, axes = plt.subplots(2, len(pyr_images)+1, figsize=(17, 5))

i = 0
for i in range(len(pyr_images)):
    im = axes[0,i].imshow(pyr_images[i], cmap='jet')
    axes[0,i].set_title(title[i])
    axes[0,i].axis('off')
    divider = make_axes_locatable(axes[0,i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    axes[1,i].imshow(lac_images[i], cmap='jet')
    axes[1,i].axis('off')
    divider = make_axes_locatable(axes[1,i])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    i += 1

# Plot the anatomical image
anat_path = sorted([os.path.join(perf_dir, f) for f in os.listdir(perf_dir) if ('Anat' in f) and (f.endswith('.nii') or f.endswith('.nii.gz'))])
anat_img = np.squeeze(nib.load(" ".join(anat_path)).get_fdata())
axes[0,i].imshow(anat_img, cmap='gray')
axes[0,i].set_title(title[i])
axes[0,i].axis('off')

axes[1,i].imshow(anat_img, cmap='gray')
axes[1,i].set_title(title[i])
axes[1,i].axis('off')

# Add vertical text labels
fig.text(0.008, 0.75, 'Pyruvate', va='center', ha='center', rotation='vertical', fontsize=12)
fig.text(0.008, 0.25, 'Lactate', va='center', ha='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.savefig(f"{perf_dir}/mainresults.tiff", dpi=600, format='tiff', bbox_inches='tight')
plt.show(block=True)
plt.close(fig)
