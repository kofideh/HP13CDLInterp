import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.io import imread, imsave
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from skimage.draw import disk
from skimage.transform import resize  # Add this import

# Define the folders and file patterns
folders = [os.path.join('results', f) for f in ['LR', 'HR', 'DC', 'CNN', 'SRGAN', 'Anat']]
folders = ['LR', 'HR', 'CNN', 'SRGAN', 'Anat']
patterns = ['pyr', 'lac']
title = ['Acquired HP 13C', 'Bicubic', 'CNN', 'SRGAN', 'Anatomical']
output_folder = 'montages'
epsilon = 1e-10  # Small value to avoid division by zero

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def extract_number(filename):
    match = re.search(r'_(\d+)\.nii', filename)
    return int(match.group(1)) if match else float('inf')

def get_files(folder, pattern):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if pattern.lower() in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz'))]
    return sorted(files, key=extract_number)

def calculate_mean_intensity(image, centers, radius):
    mean_intensities = []
    for center in centers:
        rr, cc = disk(center, radius, shape=image.shape)
        mean_intensity = np.mean(image[rr, cc])
        mean_intensities.append(mean_intensity)
    return mean_intensities

def plot_bar_chart(ax, values, labels, title):
    x = np.arange(3)
    bar_width = 0.25
    bars = ax.bar(x+0.4, values,  width=0.4, label=title)
    # ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean Intensity', fontsize=18)
    # ax.legend(bars, title, loc='upper right')

def plot_rois(ax, centers, radius, labels=None):
    for i, center in enumerate(centers):
        circle = plt.Circle(center, radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        if labels:
            ax.text(center[0], center[1], labels[i], color='white', fontsize=12, ha='center', va='center')

# Dictionary to store the NIfTI files
nifti_files = {pattern: {folder: [] for folder in folders} for pattern in patterns}

# Read and store the files
for folder in folders:
    for pattern in patterns:
        files = get_files(os.path.join('results',folder), pattern)
        nifti_files[pattern][folder] = [nib.load(f) for f in files]


for c in range(len(nifti_files[pattern][folder])):
    output_path = os.path.join(output_folder, f'montage_{c+1}.tiff')
    fig, axes = plt.subplots(3, len(folders), figsize=(18, 12))
    map = 'jet'
    
    # ctrs = [(69, 123), (135, 208), (217, 125)]  # Example centers of the circular ROIs
    # ctrs = [(123, 69), (208, 135), (125, 217)]  # Example centers of the circular ROIs
    ctrs = [(64, 102), (155, 59), (159, 194)]  # Example centers of the circular ROIs
    radius = 10  # Example radius of the circular ROIs
    centers = ctrs
    
    for i, f in enumerate(folders):
        pyr_image = nifti_files['pyr'][f][c].get_fdata()
        lac_image = nifti_files['lac'][f][c].get_fdata()
        
        pyr_image = np.rot90(pyr_image, k=1, axes=(0, 1))
        max_value = np.max(np.abs(pyr_image))
        pyr_image = pyr_image / (max_value + epsilon)

        lac_image = np.rot90(lac_image, k=1, axes=(0, 1))
        max_value = np.max(np.abs(lac_image))
        lac_image = lac_image / (max_value + epsilon)
        
        if i == 0:
            # Resize the images to match the anatomical image
            pyr_image = resize(pyr_image, (256, 256), order=0, preserve_range=True, anti_aliasing=False)  # Resize pyr_image
            lac_image = resize(lac_image, (256, 256), order=0, preserve_range=True, anti_aliasing=False)  # Resize lac_image


        if 'Anat' in f:
            map = 'gray'
            # Plot pyr images in the first row
        im = axes[0, i].imshow(pyr_image, cmap=map, vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0,i].set_title(title[i], fontsize=18)

        # Plot lac images in the second row
        axes[1, i].imshow(lac_image, cmap=map, vmin=0, vmax=1)
        axes[1, i].axis('off')

        # Calculate and plot mean intensity values for pyr and lac images
        if i < len(folders) - 1:
            pyr_mean_intensities = calculate_mean_intensity(pyr_image, centers, radius)
            lac_mean_intensities = calculate_mean_intensity(lac_image, centers, radius)
            ratio = [lac_mean_intensities[i] / (pyr_mean_intensities[i]) for i in range(len(pyr_mean_intensities))]
            
            x = np.arange(len(pyr_mean_intensities))
            bar_width = 0.25
            
            axes[2, i].bar(x, pyr_mean_intensities, width=bar_width, label='Pyr', alpha=0.7)
            axes[2, i].bar(x + bar_width, lac_mean_intensities, width=bar_width, label='Lac', alpha=0.7)
            axes[2, i].bar(x + 2 * bar_width, ratio, width=bar_width, label='Lac/Pyr', alpha=0.7)
            
            axes[2, i].set_xticks(x + bar_width)
            axes[2, i].set_xticklabels(['A', 'B', 'C'])
            axes[2, i].set_title('Mean ROI Intensity', fontsize=18)
            axes[2, i].legend(loc='upper left')
        elif i == len(folders) - 1:
            ct = [(y, x) for x, y in ctrs]
            labels = ['A', 'B', 'C']  # Example labels for the ROIs
            plot_rois(axes[0, i], ct, radius, labels)
            plot_rois(axes[1, i], ct, radius, labels)
            axes[2, i].axis('off')
        else:
            axes[2, i].axis('off')
            
        # Plot line plots in the third row
        if 'Anat' not in f:
            divider = make_axes_locatable(axes[0, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
            divider = make_axes_locatable(axes[1, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

    # Add vertical text labels
    fig.text(0.05, 0.82, 'Pyruvate', va='center', ha='center', rotation='vertical', fontsize=18)
    fig.text(0.05, 0.50, 'Lactate', va='center', ha='center', rotation='vertical', fontsize=18)
    plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust the left margin
    plt.savefig(output_path, dpi=300, format='tiff', bbox_inches='tight')
    plt.close()
    
# Generate montages
# for i, folder in enumerate(folders):
#     output_path = os.path.join(output_folder, f'montage_{i+1}.tiff')
#     plot_images_and_save(folders, output_path)