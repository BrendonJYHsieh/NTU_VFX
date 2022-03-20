import math
from operator import truth
import cv2
from cv2 import cvtColor
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage

# Read File
def readfile(folder,filename,image_align = False):
    images = list()
    B = list()
    with open(folder+filename, newline='') as jsonfile:
        data = json.load(jsonfile)
    for i in data:
        image = cv2.imread(folder + i["path"],cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        height, width, channel = image.shape
        B.append(i["t"])
    flatten = np.zeros((len(images), 3, width*height),dtype=np.uint8)    
    for i in range(len(images)):
        for c in range(3):
            flatten[i,c] = np.reshape(images[i][:,:,c], (width*height,))
    if(image_align):
        middle = int(len(images)/2)
        for i in range(0,len(images)):
            if(i!=middle):
                shift_y, shift_x = ComputeShift(images[middle],images[i])
                images[i] = ndimage.shift(images[i], shift=(shift_y, shift_x,0), mode='constant', cval=255)
    
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
    a =  0.45
    ww = pow(3,2)
    Lw = (hdr[0]*0.27)+(hdr[1]*0.67)+(hdr[2]*0.06)
    wr = np.exp(np.sum(np.log(Lw+0.001))/hdr.shape[1])
    Lm = a * (Lw / wr)
    Ld = (Lm*(1+(Lm/ww))/(Lm+1))
    hdr[0:3] = hdr[0:3] *Ld/Lw
    hdr = np.reshape(np.transpose(hdr), (height,width,3))
    return hdr

# Simplest tone mapping
def hdr_recover(Gr,Gg,Gb,flattenImage,B):
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
        
        e[0] = Gr[flattenImage[i][0]] - math.log(B[i])
        e[1] = Gg[flattenImage[i][1]] - math.log(B[i])
        e[2] = Gb[flattenImage[i][2]] - math.log(B[i])
        
        hdr[0] += e[0] * wij_R
        hdr[1] += e[1] * wij_G
        hdr[2] += e[2] * wij_B

    hdr = np.divide(hdr,wsum)
    hdr = np.exp(hdr)
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
                if(err<min_err):
                    min_err = err
                    current_shift_y = ys
                    current_shift_x = xs
        shift_x = current_shift_x
        shift_y = current_shift_y
        #print("end")
        if(a==0):
            return shift_y, shift_x

def draw_responseCurve(Gr,Gg,Gb):
    y = np.arange(0,256)
    plt.plot(Gr[y],y,color = 'r')
    plt.plot(Gg[y],y,color = 'g')
    plt.plot(Gb[y],y,color = 'b')

def save_radiance(image):
    
    image = np.reshape(np.transpose(image), (height,width,3))
    
    f = open("recovered_HDR.hdr", "wb")
    f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
    f.write("-Y {0} +X {1}\n".format(image.shape[0], image.shape[1]).encode())

    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()
    
# img1 = cv2.imread("./Images/NTNU1/1.jpg",cv2.IMREAD_COLOR)
# img2 = cv2.imread("./Images/NTNU1/5.jpg",cv2.IMREAD_COLOR)
# shift_y, shift_x = ComputeShift(img1,img2)
# print(shift_y,shift_x)
# adjust = ndimage.shift(img2, shift=(shift_y, shift_x,0), mode='constant', cval=255)
# cv2.imshow('adjust' , np.array(adjust, dtype = np.uint8 ) )
# cv2.imshow('img1' , np.array(img1, dtype = np.uint8 ) )
# cv2.imshow('img2' , np.array(img2, dtype = np.uint8 ) ) 

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="../data/", type=str)
parser.add_argument('--align', default=False, type=bool)

args = parser.parse_args()


start_time = time.time()
# "./images/NTNU1/"
images, B, flattenImage, width, height = readfile(args.path,"info.json", args.align)
n = 49
l = 100
Z = sampling(images,width,height,n)
Gr,Gg,Gb = response_curve(images,Z[0],B,n,l),response_curve(images,Z[1],B,n,l), response_curve(images,Z[2],B,n,l)
HDR = hdr_recover(Gr,Gg,Gb,flattenImage,B)
save_radiance(HDR)
HDR = tone_mapping(HDR)


print('Time used: {} sec'.format(time.time()-start_time))  

draw_responseCurve(Gr,Gg,Gb)

imgf32 = (HDR).astype(np.float32)
cv2.imwrite('result.png', cvtColor(imgf32,cv2.COLOR_RGB2BGR)*255)
plt.figure(constrained_layout=False,figsize=(10,10))
plt.title("Tone-mapped image", fontsize=20)
plt.imshow(imgf32)
plt.show()
