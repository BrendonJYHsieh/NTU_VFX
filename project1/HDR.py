from cmath import log,sqrt
import math
from turtle import width
from cv2 import sqrt
import numpy as np
import random
from PIL import Image
import time
import json
import matplotlib.pyplot as plt

n = 50
l = 40

# Read File
def readfile(filename):
    images = list()
    B = list()
    with open(filename, newline='') as jsonfile:
        data = json.load(jsonfile)
    for i in data:
        image = Image.open(i["path"])
        image_array = np.array(image)
        images.append(image_array)
        width, height = image.size
        B.append(i["t"])
    flatten = np.zeros((len(images), 3, width*height),dtype=np.uint8)    
    for i in range(len(images)):
        for c in range(3):
            flatten[i,c] = np.reshape(images[i][:,:,c], (width*height,))
    return images,B,flatten,width,height
# Sample
def sampling(images,width,height):
    Z = np.zeros((3,n,len(images))) 
    for i in range(n):
        sample_x = random.randrange(width)
        sample_y = random.randrange(height)
        for j in range(0,len(images)):
            Z[0][i][j] = images[j][sample_y][sample_x][0]
            Z[1][i][j] = images[j][sample_y][sample_x][1]
            Z[2][i][j] = images[j][sample_y][sample_x][2]
    return Z
# Calculate Response Curve
def response_curve(images,Z,B):
    A = np.zeros((n*len(images)+254+1,256+n),dtype = 'float')
    b = np.zeros((A.shape[0]),dtype = 'float')
    w = np.zeros((256))
    for z in range(256): 
        if z <= 127:
            w[z] = z + 1
        else: 
            w[z] = 255 - z + 1
    k = 0
    # Sample Point
    for i in range(n):
        # Picture
        for j in range(len(images)):
            wij = w[int(Z[i][j])]
            A[k][int(Z[i][j])] = wij
            A[k][256+i] = -wij
            b[k] = wij * math.log(B[j])
            k = k + 1
    A[k][128] = 1
    k = k +1
    for i in range(0,254):
        A[k][i] = l * w[i]
        A[k][i+1] = -2 * l * w[i]
        A[k][i+2] = l * w[i]
        k = k + 1
    return np.linalg.lstsq(A,b,rcond=None)[0]
# Simplest tone mapping
def recover(Gr,Gg,Gb,flattenImage,B,width,height):
    w = np.zeros((256))
    for z in range(256): 
        if z <= 127:
            w[z] = z + 1
        else: 
            w[z] = 255 - z + 1
    
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
        
        e[0] = np.subtract(Gr[flattenImage[i,0]] ,math.log(B[i]))
        e[1] = np.subtract(Gg[flattenImage[i,1]] ,math.log(B[i]))
        e[2] = np.subtract(Gb[flattenImage[i,2]] ,math.log(B[i]))
        
        hdr[0] += np.multiply(e[0] , wij_R)
        hdr[1] += np.multiply(e[1] , wij_G)
        hdr[2] += np.multiply(e[2] , wij_B)

    hdr = np.divide(hdr,wsum)
    hdr = np.exp(hdr)
    hdr = np.reshape(np.transpose(hdr), (height,width,3))
    return hdr

start_time = time.time()

images, B, flattenImage, width, height = readfile("info.json")
Z = sampling(images,width,height)
Gr = response_curve(images,Z[0],B)
Gg = response_curve(images,Z[1],B)
Gb = response_curve(images,Z[2],B)
HDR = recover(Gr,Gg,Gb,flattenImage,B,width,height)


y = np.arange(0,256)
plt.plot(Gr[y],y,color = 'r')
plt.plot(Gg[y],y,color = 'g')
plt.plot(Gb[y],y,color = 'b')

imgf32 = (HDR/np.amax(HDR)*255).astype(np.float32)
plt.figure(constrained_layout=False,figsize=(10,10))
plt.title("fused HDR radiance map", fontsize=20)
plt.imshow(imgf32)
print('Time used: {} sec'.format(time.time()-start_time))  
plt.show()

# pil_image=Image.fromarray(np.uint8(hdr))
# pil_image.show()
