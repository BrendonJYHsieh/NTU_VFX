from cmath import exp, log
from turtle import pen, width
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
from numba import jit,njit
import time
import sys

n = 50
p = 13
l = 20

images = list()
B = [13,10,4,3.2,1,0.8,0.33333333333333,0.25, 0.01666666666666,0.0125,0.003125,0.0025,0.001]
A = np.zeros((3,n*p+254+1,256+n),dtype = 'float')
b = np.zeros((3,n*p+254+1,1),dtype = 'float')

def w(z):
    return (1-(abs(z-127.5)/127.5))

for i in range(0,np.size(B)):
    B[i] = math.log(B[i])

for i in range(1,p+1):
    image = Image.open("img" + str(i) +".jpg")
    image_array = np.array(image)
    images.append(image_array)

width, height = image.size

Z = np.zeros((n,p,3)) 
for i in range(0,n):
    sample_x = random.randrange(width)
    sample_y = random.randrange(height)
    for j in range(0,p):
        Z[i][j][0] = images[j][sample_y][sample_x][0]
        Z[i][j][1] = images[j][sample_y][sample_x][1]
        Z[i][j][2] = images[j][sample_y][sample_x][2]

# Color
for a in range(0,3):
    k = 0
    # Sample Point
    for i in range(0,n):
        # Picture
        for j in range(0,p):
            wij = w(Z[i][j][a])
            A[a][k][int(Z[i][j][a])] = wij
            A[a][k][256+i] = -wij
            b[a][k][0] = wij * B[j]
            k = k + 1
    A[a][k][127] = 1
    k = k +1
    for i in range(0,254):
        A[a][k][i] = l * w(i)
        A[a][k][i+1] = -2 * l * w(i)
        A[a][k][i+2] = l * w(i)
        k = k + 1


Gr = np.linalg.lstsq(A[0],b[0],rcond=None)[0]
Gg = np.linalg.lstsq(A[1],b[1],rcond=None)[0]
Gb = np.linalg.lstsq(A[2],b[2],rcond=None)[0]



y = np.arange(1,256)
plt.subplot(2,2,1)
plt.plot(Gr[y],y)
plt.subplot(2,2,2)
plt.plot(Gg[y],y)
plt.subplot(2,2,3)
plt.plot(Gb[y],y)
plt.show()
HDR = np.zeros((height,width,3),dtype = 'uint8')
# @njit 
# def recover():
# # HDR = np.zeros((height,width,3),dtype = 'uint8')
# for i in range(0,height):
#     print(i)
#     for j in range(0,width):
#         Er = 0
#         Eg = 0
#         Eb = 0
#         Wr = 0.0001
#         Wg = 0.0001
#         Wb = 0.0001
#         for a in range(0,p):
#             R = images[a][i][j][0]
#             G = images[a][i][j][1]
#             BB = images[a][i][j][2]
#             Er += w(R) * (Gr[0][R] -B[a])
#             Wr += w(R)
#             Eg += w(G) * (Gg[0][G] - B[a])
#             Wg += w(G)
#             Eb += w(BB) * (Gb[0][BB] - B[a])
#             Wb += w(BB)
#         HDR[i][j][0] = np.uint8(math.exp(Er/Wr))
#         HDR[i][j][1] = np.uint8(math.exp(Eg/Wg))
#         HDR[i][j][2] = np.uint8(math.exp(Eb/Wb))
    # return HDR

start_time = time.time()
for i in range(0,height):
    print(i)
    for j in range(0,width):
        Er = 0
        Eg = 0
        Eb = 0
        Wr = 0.0001
        Wg = 0.0001
        Wb = 0.0001
        for a in (0,p):
            R = images[a][i][j][0]
            G = images[a][i][j][1]
            BB = images[a][i][j][2]
            Er += w(R) * (Gr[R] -B[a])
            Wr += w(R)
            Eg += w(G) * (Gg[G] - B[a])
            Wg += w(G)
            Eb += w(BB) * (Gb[BB] - B[a])
            Wb += w(BB)
        HDR[i][j][0] = np.uint8(math.exp(Er/Wr))
        HDR[i][j][1] = np.uint8(math.exp(Eg/Wg))
        HDR[i][j][2] = np.uint8(math.exp(Eb/Wb))

pil_image=Image.fromarray(HDR)
print('Time used: {} sec'.format(time.time()-start_time))
pil_image.show()
