from cmath import log
import numpy as np
import random
from PIL import Image
import time
import json
import cv2
import matplotlib.pyplot as plt

n = 50
l = 40
images = list()
B = list()

with open('info.json', newline='') as jsonfile:
    data = json.load(jsonfile)

for i in data:
    image = Image.open(i["path"])
    image_array = np.array(image)
    images.append(image_array)
    width, height = image.size
    B.append(i["t"])
    
flattenImage = np.zeros((len(images), 3, width*height),dtype=np.uint8)    
for i in range(len(images)):
    for c in range(3):
        flattenImage[i,c] = np.reshape(images[i][:,:,c], (width*height,))

A = np.zeros((3,n*len(images)+254+1,256+n),dtype = 'float')
b = np.zeros((3,A.shape[1]),dtype = 'float')
w = np.zeros((256))

for z in range(256): 
    if z <= 127:
        w[z] = z + 1
    else: 
        w[z] = 255 - z + 1      

for i in range(0,np.size(B)):
    B[i] = log(B[i])

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
        for j in range(len(images)):
            wij = w[int(Z[i][j][a])]
            A[a][k][int(Z[i][j][a])] = wij
            A[a][k][256+i] = -wij
            b[a][k] = wij * B[j]
            k = k + 1
    A[a][k][128] = 1
    k = k +1
    for i in range(0,254):
        A[a][k][i] = l * w[i]
        A[a][k][i+1] = -2 * l * w[i]
        A[a][k][i+2] = l * w[i]
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

start_time = time.time()

e = np.zeros((flattenImage.shape[1:]))
wsum = np.zeros((flattenImage.shape[1:]))
hdr = np.zeros((flattenImage.shape[1:]))

for i in range(len(images)):
    wij_R = w[flattenImage[i,0]]
    wij_G = w[flattenImage[i,1]]
    wij_B = w[flattenImage[i,2]]

    wsum[0] +=wij_R
    wsum[1] +=wij_G
    wsum[2] +=wij_B
    
    e[0] = np.subtract(Gr[flattenImage[i,0]] ,B[i])
    e[1] = np.subtract(Gg[flattenImage[i,1]] ,B[i])
    e[2] = np.subtract(Gb[flattenImage[i,2]] ,B[i])
    
    hdr[0] += np.multiply(e[0] , wij_R)
    hdr[1] += np.multiply(e[1] , wij_G)
    hdr[2] += np.multiply(e[2] , wij_B)

hdr = np.divide(hdr,wsum)
hdr = np.exp(hdr)
hdr = np.reshape(np.transpose(hdr), (height,width,3))

imgf32 = (hdr/np.amax(hdr)*255).astype(np.float32)
plt.figure(constrained_layout=False,figsize=(10,10))
plt.title("fused HDR radiance map", fontsize=20)
plt.imshow(imgf32)
print('Time used: {} sec'.format(time.time()-start_time))  
plt.show()

# pil_image=Image.fromarray(np.uint8(hdr))
# pil_image.show()
