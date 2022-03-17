import math
import cv2
from cv2 import cvtColor
from matplotlib import image
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt
from scipy import ndimage

# Read File
def readfile(filename):
    images = list()
    B = list()
    with open(filename, newline='') as jsonfile:
        data = json.load(jsonfile)
    for i in data:
        image = cv2.imread(i["path"],cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        height, width, channel = image.shape
        B.append(i["t"])
    flatten = np.zeros((len(images), 3, width*height),dtype=np.uint8)    
    for i in range(len(images)):
        for c in range(3):
            flatten[i,c] = np.reshape(images[i][:,:,c], (width*height,))
    return images,B,flatten,width,height
# Sample
def sampling(images,width,height,n):
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
def response_curve(images,Z,B,n,l):
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

def tone_mapping(hdr):
    a = 0.18
    ww = pow(4.5,2)
    Lw = (hdr[0]*0.27)+(hdr[1]*0.67)+(hdr[2]*0.06)
    wr = np.exp(np.sum(np.log(Lw+0.001))/hdr.shape[1])
    Lm = a * (Lw / wr)
    Ld = (Lm*(1+(Lm/ww))/(Lm+1))
    hdr[0:3] = hdr[0:3] *Ld/Lw
    return hdr

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
        wij_R = w[flattenImage[i][0]]
        wij_G = w[flattenImage[i][1]]
        wij_B = w[flattenImage[i][2]]
        
        wsum[0] +=wij_R
        wsum[1] +=wij_G
        wsum[2] +=wij_B
        
        e[0] = np.subtract(Gr[flattenImage[i][0]] ,math.log(B[i]))
        e[1] = np.subtract(Gg[flattenImage[i][1]] ,math.log(B[i]))
        e[2] = np.subtract(Gb[flattenImage[i][2]] ,math.log(B[i]))
        
        hdr[0] += np.multiply(e[0] , wij_R)
        hdr[1] += np.multiply(e[1] , wij_G)
        hdr[2] += np.multiply(e[2] , wij_B)

    hdr = np.divide(hdr,wsum)
    hdr = np.exp(hdr)
    
    hdr = tone_mapping(hdr)
    
    hdr = np.reshape(np.transpose(hdr), (height,width,3))
    return hdr

def ComputeBitmaps(img):
    img = cvtColor(img,cv2.COLOR_RGB2GRAY)
    height, width = img.shape 
    flatten = np.reshape(img, (width*height))
    median_value = np.percentile(flatten, 83)
    tb = np.where(flatten>median_value,255,0)
    tb = np.reshape((tb), (height,width))
    eb = np.where((flatten>median_value+4) | (flatten<median_value-4),255,0)
    eb = np.reshape((eb), (height,width))
    return tb,eb

def ComputeShift(img1,img2):
    for a in range(8,-1,-1):
        if(a==8):
            shift_x=0
            shift_y=0
        else:
            shift_x *=2
            shift_y *=2
        height, width, channel = img1.shape
        sml_img1 = cv2.resize(img1, (int(width/math.pow(2,a)), int(height/math.pow(2,a))), interpolation=cv2.INTER_AREA)
        sml_img2 = cv2.resize(img2, (int(width/math.pow(2,a)), int(height/math.pow(2,a))), interpolation=cv2.INTER_AREA)
    
        min_err = height*width
        tb1,eb1 = ComputeBitmaps(sml_img1)
        tb2,eb2 = ComputeBitmaps(sml_img2)
        current_shift_y = 0
        current_shift_x = 0
        for i in range(-1,2):
            for j in range(-1,2):
                ys = shift_y +i
                xs = shift_x +j
                shifted_tb2 = ndimage.shift(tb2, shift=(ys, xs), mode='constant',cval=255)
                shifted_eb2 = ndimage.shift(eb2, shift=(ys, xs), mode='constant',cval=255)
                diff_b = np.bitwise_xor(tb1,shifted_tb2)
                diff_b = np.bitwise_and(diff_b,eb1)
                diff_b = np.bitwise_and(diff_b,shifted_eb2)
                err = np.sum(diff_b)/255
                print(i,j,err)
                if(err<min_err):
                    min_err = err
                    current_shift_y = ys
                    current_shift_x = xs
        shift_x = current_shift_x
        shift_y = current_shift_y
        #print("end")
        if(a==0):
            print(shift_y,shift_x)
            return shift_y, shift_x

def draw_responseCurve(Gr,Gg,Gb):
    y = np.arange(0,256)
    plt.plot(Gr[y],y,color = 'r')
    plt.plot(Gg[y],y,color = 'g')
    plt.plot(Gb[y],y,color = 'b')

# img1 = cv2.imread("./NTU/361448.jpg",cv2.IMREAD_COLOR)
# img2 = cv2.imread("./NTU/361449.jpg",cv2.IMREAD_COLOR)
# shift_y, shift_x = ComputeShift(img1,img2)
# adjust = ndimage.shift(img2, shift=(shift_y, shift_x,0), mode='constant', cval=255)
# cv2.imshow('adjust' , np.array(adjust, dtype = np.uint8 ) )
# cv2.imshow('img1' , np.array(img1, dtype = np.uint8 ) )
# cv2.imshow('img2' , np.array(img2, dtype = np.uint8 ) ) 

start_time = time.time()
images, B, flattenImage, width, height = readfile("./info.json")
# for i in range(len(images)):
#     print(i)
#     if(i!=5):
#         shift_y, shift_x = ComputeShift(images[5],images[i])
#         images[i] = ndimage.shift(images[i], shift=(shift_y, shift_x,0), mode='constant', cval=255)

n = 50
l = 40
Z = sampling(images,width,height,n)
Gr = response_curve(images,Z[0],B,n,l)
Gg = response_curve(images,Z[1],B,n,l)
Gb = response_curve(images,Z[2],B,n,l)
HDR = recover(Gr,Gg,Gb,flattenImage,B,width,height)

np.save('HDR.hdr',HDR)

print('Time used: {} sec'.format(time.time()-start_time))  

draw_responseCurve(Gr,Gg,Gb)

imgf32 = (HDR).astype(np.float32)
plt.figure(constrained_layout=False,figsize=(10,10))
plt.title("fused HDR radiance map", fontsize=20)
plt.imshow(imgf32)
plt.show()
