import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

def create_phantom(size=16):
    """Create a simple phantom image suitable for small matrix size"""
    phantom = np.zeros((size, size))
    
    # Create fewer, more visible grid lines
    # Add just 3 vertical and 3 horizontal lines
    line_positions = [size//4, size//2, 3*size//4]
    for i in line_positions:
        phantom[i, :] = 1.0  # Horizontal lines
        phantom[:, i] = 1.0  # Vertical lines
    
    # Add a single circle in the center
    center = size // 2
    y, x = np.ogrid[-center:size-center, -center:size-center]
    mask = x*x + y*y <= (size//4)**2
    phantom[mask] = 0.7
    
    return phantom

def create_field_inhomogeneity(size=16, strength=1.0, direction='frequency'):
    """Generate a simplified field inhomogeneity map for small matrix"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    if direction == 'frequency':
        # Simpler variation pattern suitable for small matrix
        field = strength * x
    elif direction == 'phase':
        field = strength * y
    else:
        raise ValueError("direction must be 'frequency' or 'phase'")
    
    # Add minimal random variations with larger spatial scale
    random_variation = gaussian_filter(np.random.randn(size, size), sigma=size//2)
    field += 0.1 * strength * random_variation
    
    return field

def apply_geometric_distortion(phantom, field_map, pixel_shift_scale=3.0, direction='frequency'):
    """Apply geometric distortion with parameters adjusted for small matrix"""
    size = phantom.shape[0]
    y_coords, x_coords = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    
    # Reduced pixel shift scale for small matrix
    pixel_shifts = pixel_shift_scale * field_map
    
    if direction == 'frequency':
        distorted_x_coords = x_coords + pixel_shifts
        distorted_y_coords = y_coords
    else:
        distorted_x_coords = x_coords
        distorted_y_coords = y_coords + pixel_shifts
    
    coords = np.stack([distorted_y_coords, distorted_x_coords])
    distorted_image = map_coordinates(phantom, coords, order=1, mode='reflect')
    
    return distorted_image

def simulate_geometric_distortion(size=16, field_strength=1.0, pixel_shift_scale=3.0, direction='frequency'):
    """Simulate complete MRI acquisition with parameters optimized for small matrix"""
    phantom = create_phantom(size)
    field_map = create_field_inhomogeneity(size, field_strength, direction)
    distorted_image = apply_geometric_distortion(phantom, field_map, pixel_shift_scale, direction)
    
    # Create larger figure size relative to matrix size for better visibility
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use nearest neighbor interpolation for display to show actual pixels
    axes[0].imshow(phantom, cmap='gray', interpolation='nearest')
    axes[0].set_title('Original Phantom')
    axes[0].grid(True, which='major', color='r', linestyle='-', linewidth=0.5)
    
    axes[1].imshow(field_map, cmap='RdBu', interpolation='nearest')
    axes[1].set_title(f'Field Inhomogeneity Map\n({direction}-encode direction)')
    axes[1].grid(True, which='major', color='r', linestyle='-', linewidth=0.5)
    
    axes[2].imshow(distorted_image, cmap='gray', interpolation='nearest')
    axes[2].set_title('Distorted Image')
    axes[2].grid(True, which='major', color='r', linestyle='-', linewidth=0.5)
    
    # Add pixel indices for reference
    for ax in axes:
        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_axis_on()
    
    plt.tight_layout()
    return phantom, field_map, distorted_image

# Example usage
if __name__ == "__main__":
    # Simulate distortion in frequency-encode direction
    phantom_freq, field_map_freq, distorted_freq = simulate_geometric_distortion(
        size=16,
        field_strength=1.0,
        pixel_shift_scale=3.0,  # Reduced from 20.0 for small matrix
        direction='frequency'
    )
    plt.show()
