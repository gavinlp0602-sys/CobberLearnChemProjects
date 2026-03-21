import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# 1. Create the 5x5 image array
image = np.zeros((5, 5), dtype=int)
image[1:4, 2] = 1  # Vertical line
image[2, 1:4] = 1  # Horizontal line

# 2. Define the Kernels
# Horizontal Edge Detector (Detects changes along the Y-axis)
h_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Vertical Edge Detector (Detects changes along the X-axis)
v_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Diagonal Edge Detector (Detects 45-degree edges)
d_kernel = np.array([
    [ 0,  1,  2],
    [-1,  0,  1],
    [-2, -1,  0]
])

# 3. Apply Convolution
h_feature = convolve2d(image, h_kernel, mode='same', boundary='fill', fillvalue=0)
v_feature = convolve2d(image, v_kernel, mode='same', boundary='fill', fillvalue=0)
d_feature = convolve2d(image, d_kernel, mode='same', boundary='fill', fillvalue=0)

# Print the numerical results
print("Vertical Kernel:")
print(v_kernel)
print("\nVertical Feature Map:")
print(v_feature)
print("\nDiagonal Kernel:")
print(d_kernel)
print("\nDiagonal Feature Map:")
print(d_feature)

# 4. Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original
axes[0, 0].imshow(image, cmap='gray', interpolation='nearest')
axes[0, 0].set_title('Original Image (5x5 Square)')

# Horizontal
im_h = axes[0, 1].imshow(h_feature, cmap='RdBu', interpolation='nearest')
axes[0, 1].set_title('Horizontal Edge Detection')
plt.colorbar(im_h, ax=axes[0, 1])

# Vertical
im_v = axes[1, 0].imshow(v_feature, cmap='RdBu', interpolation='nearest')
axes[1, 0].set_title('Vertical Edge Detection')
plt.colorbar(im_v, ax=axes[1, 0])

# Diagonal
im_d = axes[1, 1].imshow(d_feature, cmap='RdBu', interpolation='nearest')
axes[1, 1].set_title('Diagonal Edge Detection')
plt.colorbar(im_d, ax=axes[1, 1])

# Add grid lines
for ax in axes.flat:
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('edge_detection_results.png')
plt.show()