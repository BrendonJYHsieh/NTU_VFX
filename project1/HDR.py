import numpy as np
from PIL import Image

image = Image.open("image0.png")
image_array = np.array(image)
print(image_array[0][0][0])  
print(image_array.shape)  