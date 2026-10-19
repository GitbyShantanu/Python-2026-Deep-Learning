import cv2 

# Read image
img = cv2.imread("sample.png", 0) # grayscale 

# Canny Edge Detection 
edges = cv2.Canny(img, 100, 200) 

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

