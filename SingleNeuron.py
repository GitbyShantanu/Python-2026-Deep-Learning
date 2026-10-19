import numpy as np

# Step 1 : Define input features ie X
#                 [X1, X2, X3]
input = np.array([2.0,3.0,4.0]) 
print("X : ", input) 

# Step 2 : Define weights ie W 
#                 [W1, W2, W3]
weights = np.array([0.5,0.3,0.2]) 
print("W : ", weights) 

# Step 3 : Define bias ie b 
#       b 
bias = 1.0 
print("b : ", bias) 

# Step 4 : Calculate Weighted Sum ie Z 
# Z = x1w1 + x2w2 + x3w3 + b
# z = (2.0*0.5) + (3.0*0.3) + (4.0)*(0.2) + 1.0 

z = np.dot(input, weights) + bias 
print("Z : ", z) 

# Step 5 : Activation function (ReLU)
def ReLU(X):
    return max(0, X) 

# Step 6 : Final Output ie Y 
Y = ReLU(z)
print("Y : ", Y)    