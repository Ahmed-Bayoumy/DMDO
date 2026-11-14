import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 1. Define your unique categories
x_centers = np.array([1, 2, 3])  # 3 categories
y_centers = np.array([1, 2])     # 2 categories

# 2. Create the grid coordinates
x_grid, y_grid = np.meshgrid(x_centers, y_centers)

# 3. CRITICAL: Flatten them to 1D arrays (Length becomes 3 * 2 = 6)
x_pos = x_grid.flatten()
y_pos = y_grid.flatten()

# Total number of boxes we are drawing is 6
num_boxes = len(x_pos) 

# 4. CRITICAL: Ensure ALL other arrays match 'num_boxes' (6 elements)
z_base = np.array([10, 12, 11, 15, 14, 13])      # Must be exactly length 6
z_top = np.array([25, 28, 22, 30, 29, 27])       # Must be exactly length 6
z_height = z_top - z_base                        # Automatically length 6

# 5. Box dimensions (Can be a single scalar OR a 1D array of length 6)
width = np.full(num_boxes, 0.3)                  # Length 6 array filled with 0.3
depth = np.full(num_boxes, 0.3)                  # Length 6 array filled with 0.3

# 6. Plotting - Shift x and y so bars are centered on their ticks
ax.bar3d(
    x_pos - 0.15, 
    y_pos - 0.15, 
    z_base, 
    width, 
    depth, 
    z_height, 
    color='skyblue', 
    edgecolor='navy'
)

ax.set_xlabel('X Categories')
ax.set_ylabel('Y Categories')
ax.set_zlabel('Values')

plt.show()