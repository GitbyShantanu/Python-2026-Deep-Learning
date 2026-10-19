# --------------------------------------
# 6 * 6 GrayScale Image
# Top Half = Black (0)
# Bottom Half = White (255) 
# Apply 3*3 Kernel for edge detection
# Show Feature map
# --------------------------------------

import numpy as np 

# Step 1 : Creare 6 * 6 Image 

image = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255] 
])

print("\nOriginal 6 x 6 Image") 
print(image) 

# Step 2 : 3*3 Kernel for Horizontal Edge Detection 

kernel = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1] 
])

print("\n3 x 3 Kernel")
print(kernel) 

# Step 3: Convolution Operation
# Output Size = (6-3+1) x (6-3+1) = 4 x 4 

feature_map = np.zeros((4,4)) 

for i in range(4):
    for j in range(4):

        # Extract 3x3 region 
        region = image[i:i+3, j:j+3]

        # Multiply and Sum 
        result = np.sum(region * kernel) 

        # Store result
        feature_map[i][j] = result 

# Step 4 : Show Feature Map 

print("\nFeature Map (Detected Edge)") 
print(feature_map) 