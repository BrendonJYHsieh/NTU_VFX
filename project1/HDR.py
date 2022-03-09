from cmath import log
from turtle import width
import numpy as np
import random
from PIL import Image

samble_num = 50
picture_num = 13;
Zmax = 255;
Zmin = 0;
images = list()
B = [13,10,4,3.2,1,0.8,0.33333333333333,0.25, 0.01666666666666,0.0125,0.003125,0.0025,0.001]

for i in range(0,np.size(B)):
    B[i] = log(B[i])

for i in range(1,picture_num):
    image = Image.open("img" + str(i) +".jpg")
    image_array = np.array(image)
    images.append(image_array)

width = image_array.shape[1]
height = image_array.shape[0]
Z = np.zeros((samble_num,picture_num,3))
for i in range(0,samble_num):
    sample_x = random.randrange(width)
    sample_y = random.randrange(height)
    for j in range(0,picture_num-1):
        Z[i][j][0] = images[j][sample_y][sample_x][0]
        Z[i][j][1] = images[j][sample_y][sample_x][1]
        Z[i][j][2] = images[j][sample_y][sample_x][2]