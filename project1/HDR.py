from cmath import exp, log
from turtle import pen, width
import numpy as np
import random
import math
from PIL import Image
from numba import jit,njit
import time
import json
import cv2
import matplotlib.pyplot as plt

n = 50
l = 40
images = list()
images1 = list()
B = list()
with open('info.json', newline='') as jsonfile:
    data = json.load(jsonfile)
for i in data:
    image = Image.open(i["path"])
    image_array = np.array(image)
    images.append(image_array)
    width, height = image.size
    B.append(i["t"])
    
    image1 = cv2.imread(i["path"],cv2.IMREAD_COLOR)
    images1.append(image1)

A = np.zeros((3,n*len(images)+256+1,256+n),dtype = 'float')
b = np.zeros((3,A.shape[1]),dtype = 'float')

w = np.zeros((256))
for z in range(256): 
    if z <= 127:
        w[z] = z + 1
    else: 
        w[z] = 255 - z + 1
            

for i in range(0,np.size(B)):
    B[i] = math.log(B[i])

Z = np.zeros((n,len(images),3)) 
for i in range(n):
    sample_x = random.randrange(width)
    sample_y = random.randrange(height)
    for j in range(0,len(images)):
        Z[i][j][0] = images[j][sample_y][sample_x][0]
        Z[i][j][1] = images[j][sample_y][sample_x][1]
        Z[i][j][2] = images[j][sample_y][sample_x][2]

# Color
for a in range(3):
    k = 0
    # Sample Point
    for i in range(n):
        # Picture
        for j in range(0,len(images)):
            wij = w[int(Z[i][j][a])]
            A[a][k][int(Z[i][j][a])] = wij
            A[a][k][256+i] = -wij
            b[a][k] = wij * B[j]
            k = k + 1
    A[a][k][129] = 1
    k = k +1
    for i in range(1,254):
        A[a][k][i] = l * w[i+1]
        A[a][k][i+1] = -2 * l * w[i+1]
        A[a][k][i+2] = l * w[i+1]
        k = k + 1


Gr = np.linalg.lstsq(A[0],b[0],rcond=None)[0]
Gg = np.linalg.lstsq(A[1],b[1],rcond=None)[0]
Gb = np.linalg.lstsq(A[2],b[2],rcond=None)[0]



y = np.arange(0,256)
plt.subplot(2,2,1)
plt.plot(Gr[y],y)
plt.subplot(2,2,2)
plt.plot(Gg[y],y)
plt.subplot(2,2,3)
plt.plot(Gb[y],y)
plt.show()
HDR = np.zeros((height,width,3),dtype = 'uint8')
# for i in range(height):
#     print(i)
#     for j in range(width):
#         Er = 0
#         Eg = 0
#         Eb = 0
#         Wr = 0.0001
#         Wg = 0.0001
#         Wb = 0.0001
#         for a in range(len(images)):
#             R = images[a][i][j][0]
#             G = images[a][i][j][1]
#             BB = images[a][i][j][2]
#             Er += w[R] * (Gr[R] - B[a])
#             Wr += w[R]
#             Eg += w[G]* (Gg[G] - B[a])
#             Wg += w[G]
#             Eb += w[BB] * (Gb[BB] - B[a])
#             Wb += w[BB]
#         HDR[i][j][0] = (math.exp(Er/Wr))
#         HDR[i][j][1] = (math.exp(Eg/Wg))
#         HDR[i][j][2] = (math.exp(Eb/Wb))

start_time = time.time()
for i in range(height):
    print(i)
    for j in range(width):
        Er = 0
        Eg = 0
        Eb = 0
        Wr = 0.000001
        Wg = 0.000001
        Wb = 0.000001
        for a in range(len(images)):
            R = images[a][i][j][0]
            G = images[a][i][j][1]
            BB = images[a][i][j][2]
            Er += w[R]* (Gr[R] - B[a])
            Wr += w[R]
            Eg += w[G]* (Gg[G] - B[a])
            Wg += w[G]
            Eb += w[BB] * (Gb[BB] - B[a])
            Wb += w[BB]
        HDR[i][j][0] = (math.exp(Er/Wr))
        HDR[i][j][1] = (math.exp(Eg/Wg))
        HDR[i][j][2] = (math.exp(Eb/Wb))


pil_image=Image.fromarray(np.uint8(HDR))
print('Time used: {} sec'.format(time.time()-start_time))
pil_image.show()
