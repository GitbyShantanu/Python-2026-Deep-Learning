from PIL import Image

img = Image.open("Digit.png")

print(img.size) 

img = img.convert("L")
img = img.resize((28,28))

img.save("digit_28x28.png")

print(img.size)