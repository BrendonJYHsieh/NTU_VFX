from math import floor
import cv2
import sys
import numpy as np
import time
import math
import random
from tqdm import tqdm, trange
from cv2 import GaussianBlur,sqrt,resize
from matplotlib import pyplot as plt
from numpy import array, zeros,sqrt,log,subtract,all
from numpy.linalg import lstsq, norm

def Dog_visualization(layers):
    total_height = 0
    max_width = 0
    for i in layers:
        total_height +=i[0].shape[0]
        if(i[0].shape[1]>max_width):
            max_width= i[0].shape[1]
    height = total_height
    width = max_width
    newimg = np.zeros((height, width*5, 3), np.uint8)
    amount = 0
    for l in layers:
        for i in range(0,len(l)):
            for j in range(3):
                h, w = l[i].shape
                newimg[amount:h+amount,w*i:w*(i+1),j]=l[i] 
        amount += l[0].shape[0]
    plt.imshow(newimg)
    plt.show()

def DoG(image):
    # Generate Gaussian sigmas

    sigma, s = 1.6, 3
    k = 2 ** (1. / s)
    
    layers_of_images_in_octave = s + 3
    gaussian_sigmas  = zeros(layers_of_images_in_octave-1)

    for i in range(1,layers_of_images_in_octave):
        s_pre  = (k**(i-1)) * sigma; 
        s_post = (k**i)     * sigma; 
        gaussian_sigmas[i-1] = sqrt(s_post**2 - s_pre**2)

    # Generate Gaussian Images

    num_of_octave = floor((log(min(image.shape)) / log(2)) - 2)
    gaussian_images = []
    for i in range(num_of_octave):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)
        for sigma in gaussian_sigmas:
            image = GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            gaussian_images_in_octave.append(image)
        next_octave_image = gaussian_images_in_octave[-3] # k^3 sigma
        image = resize(next_octave_image, (int(next_octave_image.shape[1] / 2), int(next_octave_image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        gaussian_images.append(gaussian_images_in_octave)
     
    # Generate DOG

    dog_images = []
    for i in gaussian_images:
        subtract_images = []
        for j in range(1,len(i)):
            subtract_images.append(subtract(i[j],i[j-1]))
        dog_images.append(subtract_images)

    return array(gaussian_images, dtype=object), dog_images

def Find_Keypoints(gaussian_images, dogs, image_border = 15):
    keypoints = []
    
    print("finding keypoints...")
    progress = tqdm(total = len(dogs))
    for octave_index, dog in enumerate(dogs):
        progress.update(1)
        progress.refresh()
        for image_index in range(1,len(dog)-1):
            image0, image1,image2 = dog[image_index-1:image_index+2]
            height , width = image0.shape
            for i in range(image_border, height - 1 - image_border):
                for j in range(image_border, width - 1 - image_border):
                    center_pixel = image1[i,j]
                    extremum_exist = False
                    if (abs(center_pixel)>1):
                        if(center_pixel>0):
                            if(all(center_pixel >= image0[i-1:i+2, j-1:j+2]) and all(center_pixel >= image1[i-1:i+2, j-1:j+2]) and all(center_pixel >= image2[i-1:i+2, j-1:j+2])):
                                extremum_exist = True
                        else:
                            if(all(center_pixel <= image0[i-1:i+2, j-1:j+2]) and all(center_pixel <= image1[i-1:i+2, j-1:j+2]) and all(center_pixel <= image2[i-1:i+2, j-1:j+2])):
                                extremum_exist = True

                    if(extremum_exist):
                        ii = i
                        jj = j
                        _image_index = image_index
                        for iteration in range(5):

                            _image0, _image1, _image2 = dog[_image_index-1:_image_index+2]
                            _center_pixel = _image1[ii, jj]

                            # Gradient
                            g = [((_image1[ii  , jj+1] - _image1[ii  , jj-1])/2)/255, 
                                 ((_image1[ii+1, jj  ] - _image1[ii-1, jj  ])/2)/255, 
                                 ((_image2[ii  , jj  ] - _image0[ii  , jj  ])/2)/255]

                            # Hessian
                            h = array([
                                [(_image1[ii  , jj+1] - 2 * _center_pixel + _image1[ii  , jj-1])/255,
                                 (_image1[ii+1, jj+1] - _image1[ii+1, jj-1] - _image1[ii-1, jj+1] + _image1[ii-1, jj-1])/4/255,
                                 (_image2[ii  , jj+1] - _image2[ii  , jj-1] - _image0[ii  , jj+1] + _image0[ii  , jj-1])/4/255], 
                                [(_image1[ii+1, jj+1] - _image1[ii+1, jj-1] - _image1[ii-1, jj+1] + _image1[ii-1, jj-1])/4/255,
                                 (_image1[ii+1, jj  ] - 2 * _center_pixel + _image1[ii-1, jj  ])/255,
                                 (_image2[ii+1, jj  ] - _image2[ii-1, jj  ] - _image0[ii+1, jj  ] + _image0[ii-1, jj  ])/4/255],
                                [(_image2[ii  , jj+1] - _image2[ii  , jj-1] - _image0[ii  , jj+1] + _image0[ii  , jj-1])/4/255, 
                                 (_image2[ii+1, jj  ] - _image2[ii-1, jj  ] - _image0[ii+1, jj  ] + _image0[ii-1, jj  ])/4/255, 
                                 (_image2[ii  , jj  ] - 2 * _center_pixel + _image0[ii  , jj  ])/255]])

                            approximation = -lstsq(h, g, rcond=None)[0]

                            # Good enough can, so no need to find
                            if all(abs(approximation)<0.5): 
                                break

                            jj += round(approximation[0])
                            ii += round(approximation[1])
                            _image_index += round(approximation[2])

                            # extremum_existing point is inside or not convergence
                            if iteration == 4 or ii < image_border or jj < image_border or ii > height - 1 - image_border  or jj > width - 1 -image_border  or _image_index < 1 or _image_index > 3: 
                                extremum_exist = False
                                break

                        if extremum_exist:
                            response = _center_pixel + 0.5 * np.dot(g, approximation)

                            # delete low constrast
                            if np.abs(response * 3) > 0.04:
                                # eliminate edge effect
                                if ((h[0,0] + h[1,1]) ** 2) / (h[0,0] * h[1,1] - h[0,1] * h[0,1]) < 12.1:
                                    # Keypoint 
                                    keypoint = cv2.KeyPoint(
                                    (jj) , # X
                                    (ii) # Y
                                    , 1.6 * (2 ** ((_image_index + approximation[2] - 1) / 3.)) * (2 ** (octave_index + 1))  #size
                                    , -1 # angle
                                    , abs(response) # response
                                    , octave_index + _image_index * (2 ** 8)) # octave

                                    gaussian_image = gaussian_images[octave_index][_image_index]
                                    height, width = gaussian_image.shape
                                    scale = 1.5 * keypoint.size  
                                    radius = round(3 * scale)
                                    histogram = zeros(36)

                                    for a in range(-radius, radius + 1):
                                        for b in range(-radius, radius + 1):
                                            y = round(keypoint.pt[1]) + a
                                            x = round(keypoint.pt[0]) + b
                                            if y > image_border and y < height - 1 - image_border and x > image_border and x < width - image_border - 1:
                                                Lx = gaussian_image[y, x + 1] - gaussian_image[y, x - 1]
                                                Ly = gaussian_image[y - 1, x] - gaussian_image[y + 1, x]
                                                m = sqrt(Lx * Lx + Ly * Ly)
                                                theta =  np.rad2deg(np.arctan2(Ly, Lx))
                                                histogram[round(theta / 10.) % 36] += np.exp(-0.5 / (scale ** 2) * (a ** 2 + b ** 2)) * m            
                                    
                                    
                                    for smooth_index in range(72):
                                        histogram[smooth_index%36] = histogram[(smooth_index-1)%36] * 0.25 + histogram[smooth_index%36] * 0.5 + histogram[(smooth_index+1)%36] * 0.25

                                    orientation_max = max(histogram)
                                    for histogram_index in range(len(histogram)):
                                        if histogram[histogram_index] >= 0.8 * orientation_max and histogram[histogram_index]>histogram[histogram_index-1] and histogram[histogram_index]>histogram[(histogram_index+1)%36]: # Make description more reliable
                                            orientation = 360. - (histogram_index + 0.5 * (histogram[(histogram_index - 1) % 36] - histogram[(histogram_index + 1) % 36]) 
                                                        / (histogram[(histogram_index - 1) % 36] - 2 * histogram[histogram_index] + histogram[(histogram_index + 1) % 36])) % 36 * 10.
                                            new_keypoint = cv2.KeyPoint(
                                              (keypoint.pt[0] + approximation[0]) * 2  ** ((keypoint.octave & 255))
                                            , (keypoint.pt[1] + approximation[1]) * 2  ** ((keypoint.octave & 255))
                                            , keypoint.size / 2
                                            , orientation
                                            , keypoint.response
                                            , keypoint.octave)
                                            keypoints.append(new_keypoint)
    return keypoints

def Trilinear_interpolation(r,c,o,m,d,obins=8):
    histogram = zeros(128)   
    for r, c, o, m  in zip(r, c, o, m):
            
            r0 = floor(r)
            c0 = floor(c)
            o0 = floor(o)
            d_r = r - r0
            d_c = c - c0
            d_o = o - o0

            for i in range(0,2):
                r_index = r0 + i
                if (r_index >= 0 and r_index < d):
                    for j in range(0,2):
                        c_index = c0 + j
                        if (c_index >=0 and c_index < d):
                            for k in range(0,2):
                                o_index = ( (o0+k) % obins)
                                value = m * ( 0.5 + (d_r - 0.5)*(2*i-1) )* ( 0.5 + (d_c - 0.5)*(2*j-1) ) * ( 0.5 + (d_o - 0.5)*(2*k-1) )
                                histogram[r_index*d*obins + c_index*obins + o_index] += value

    return histogram.flatten()

def Generate_Descriptors(keypoints, gaussian_images, descr_hist_d=4):
    print("generate descriptors...")
    progress = tqdm(total = len(keypoints))
    descriptors = []

    for keypoint in keypoints:
        progress.update(1)
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        scale = 1 / np.float32(1 << octave)

        gaussian_image = gaussian_images[octave, layer]

        height, width = gaussian_image.shape
        point = np.round(scale * array(keypoint.pt))

        cos_angle = np.cos(np.deg2rad(-keypoint.angle))
        sin_angle = np.sin(np.deg2rad(-keypoint.angle))

        row_list,col_list,m_list,orientation_list = [], [], [], []

        hist_width = 1.5 * scale * keypoint.size
        radius = round(hist_width * (descr_hist_d + 1) * sqrt(2) / 2)

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                i_rot = j * sin_angle + i * cos_angle
                j_rot = j * cos_angle - i * sin_angle
                r_bin = (i_rot / hist_width) + 0.5 * descr_hist_d - 0.5
                c_bin = (j_rot / hist_width) + 0.5 * descr_hist_d - 0.5
                if r_bin > -1 and r_bin < descr_hist_d and c_bin > -1 and c_bin < descr_hist_d:
                    window_row, window_col = round(point[1] + i), round(point[0] + j)
                    if window_row > 0 and window_row < height - 1 and window_col > 0 and window_col < width - 1:
                        Lx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        Ly = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        m = sqrt(Lx * Lx + Ly * Ly)
                        theta = np.rad2deg(np.arctan2(Ly, Lx)) % 360
                        row_list.append(r_bin)
                        col_list.append(c_bin)
                        m_list.append(np.exp( -(j_rot*j_rot+i_rot*i_rot) / (2*(0.5*descr_hist_d*hist_width) ** 2) ) * m)
                        orientation_list.append((theta - keypoint.angle) * 8 / 360.)

        descriptor_128 = Trilinear_interpolation(row_list, col_list, orientation_list, m_list,descr_hist_d)
        threshold = norm(descriptor_128) * 0.2
        descriptor_128[descriptor_128 > threshold] = threshold
        descriptor_128 = cv2.normalize(descriptor_128, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        for index in range(len(descriptor_128)):
            if descriptor_128[index] < 0:
                descriptor_128[index] = 0
            elif descriptor_128[index] > 255:
                descriptor_128[index] = 255
        descriptors.append(descriptor_128)
        
    return array(descriptors, dtype='float32')

def cylindricalWarpImage(image, focal_length):
    height, width = image.shape[0:2]

    K = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]])

    cyl = np.zeros_like(image)

    for i in np.arange(0,height):
        for j in np.arange(0,width):
            theta = (j - float(width) / 2.0) / focal_length
            h     = (i - float(height) / 2.0) / focal_length

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            jj = X[0] / X[2]
            if jj < 0 or jj >= width:
                continue

            ii = X[1] / X[2]
            if ii < 0 or ii >= height:
                continue

            cyl[int(i),int(j)] = image[int(ii),int(jj)]

    return (cyl)

def SIFT(image):
    _image = image.copy()
    image = GaussianBlur(image, (0, 0), sigmaX=1.24, sigmaY=1.24) #1.6
    gaussian, dog = DoG(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY).astype('float32'))
    Dog_visualization(dog)
    keypoints = Find_Keypoints(gaussian,dog)
    descriptors = Generate_Descriptors(keypoints, gaussian)
    return keypoints,descriptors,cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

def matcher(kp1, des1, kp2, des2, threshold = 0.5,crossCheck = True):
    # CrossCheck
    print("matching image ...")
    matches = []
    for i in trange(len(kp1)):
        first_min = sys.maxsize
        second_min = first_min
        jj = 0
        for j in range(len(kp2)):
            sum = norm(des1[i] - des2[j])
            if(sum < first_min):
                second_min = first_min
                first_min = sum
                jj = j
                
        if first_min / second_min < threshold:
            if(crossCheck):
                ii = 0
                first_minn = sys.maxsize
                second_minn = first_minn
                for index in range(len(kp1)):
                    sumn = norm(des1[index] - des2[jj])
                    if(sumn < first_minn):
                        second_minn = first_minn
                        first_minn = sumn
                        ii = index
                
                if first_minn / second_minn < threshold:
                    if(ii == i):
                        matches.append(list(kp1[i].pt + kp2[jj].pt))
                        np.delete(des2, jj)
            else:
                np.delete(des2, jj)
                matches.append(list(kp1[i].pt + kp2[jj].pt))

    return np.array(matches)

def homography(matches):
    matrix = []
    for match in matches:
        matrix.append([0, 0, 0, match[0], match[1], 1., -match[3]*match[0], -match[3]*match[1], -match[3]])
        matrix.append([match[0], match[1], 1., 0, 0, 0, -match[2]*match[0], -match[2]*match[1], -match[2]])
    U, s, V = np.linalg.svd(np.array(matrix))
    H = V[8].reshape(3, 3)
    H = H/H[2, 2]
    return H

def ransac(matches,threshold = 0.5):
    
    best_inliers = []
    for index in range(2000):

        H = homography(np.array([matches[i] for i in random.sample(range(len(matches)), 4) ]))

        inliers = []
        for matche in matches:
            p1 = np.append(matche[0:2],1)
            p2 = np.dot(H,p1)
            p2 = (p2/p2[2])[0:2]
            error = np.sum((matche[2:4] - p2)**2)
            if( error < threshold):
                inliers.append(error)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
            
    print("inliers/matches: {}/{} = {}%".format(len(best_inliers), len(matches),len(best_inliers)/len(matches)))
    return best_H

def transform(src_pts, H):
    src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
    pts = np.dot(H, src.T).T
    pts = (pts / pts[:, 2].reshape(-1, 1))[:, 0:2]
    return pts

def warpPerspective(image, H, dsize):
    width,height = dsize
    image = cv2.normalize(image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX) 
    idx_pts = np.mgrid[0:width, 0:height].reshape(2, -1).T
    map_pts = transform(idx_pts, np.linalg.inv(H))
    map_pts = map_pts.reshape(width, height, 2).astype(np.float32)
    warped = cv2.remap(image, map_pts, None, cv2.INTER_CUBIC).transpose(1, 0, 2)
    return warped

def stitch_img(left, right, H):
    print("stiching image ...")
    height, width = left.shape[0:2]
    corners = [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]
    _corners = np.array([np.dot(H, corner) for corner in corners]).T 
    y_offset = min(_corners[1] / _corners[2])
    x_offset = min(_corners[0] / _corners[2])

    translation = np.array([[1, 0, -x_offset], [0, 1, -y_offset], [0, 0, 1]])
    translation_H = np.dot(translation, H)

    left = cv2.normalize(left.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX) 
    right = cv2.normalize(right.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX) 

    left_warped = warpPerspective(left, translation_H, (round(abs(x_offset) + left.shape[1]),round(abs(y_offset) + left.shape[0])))
    right_warped = warpPerspective(right, translation, (round(abs(x_offset) + right.shape[1]),round(abs(y_offset) + right.shape[0])))

    for i in trange(right_warped.shape[0]):
        for j in range(right_warped.shape[1]):
            start = abs(round(x_offset))+5
            end = width-5
            weight = 1-((j-start)/(end-start))
            if np.sum(left_warped[i, j])<0.2 and np.sum(right_warped[i, j]):
                left_warped[i, j] = right_warped[i, j]
            elif np.sum(left_warped[i, j]) and np.sum(right_warped[i, j]) > 0.2:
                left_warped[i, j] = (left_warped[i, j]*weight + right_warped[i, j]*(1-weight))
                if(np.sum(left_warped[i, j]) > np.sum(right_warped[i, j]) and np.sum(right_warped[i, j])/np.sum(left_warped[i, j])<0.95):
                    pass
                elif(np.sum(left_warped[i, j]) < np.sum(right_warped[i, j]) and np.sum(left_warped[i, j])/np.sum(right_warped[i, j])<0.95):
                    left_warped[i, j] = right_warped[i, j]
                else:
                    left_warped[i, j] = (left_warped[i, j]*weight + right_warped[i, j]*(1-weight))
                    
    stitch_image = left_warped[:right_warped.shape[0], :right_warped.shape[1], :]
    return stitch_image


def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8'))
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()

def plot_keypoint(kp_left,left_rgb,kp_right,right_rgb):
    for i in kp_left:
        cv2.circle(left_rgb, (int(i.pt[0]),int(i.pt[1])), radius=2, color=(255, 0, 0), thickness=-1)
    for j in kp_right:
        cv2.circle(right_rgb, (int(j.pt[0]),int(j.pt[1])), radius=2, color=(255, 0, 0), thickness=-1)
    total_img = np.concatenate((left_rgb, right_rgb), axis=1)
    plt.imshow(total_img)
    plt.show()
start = time.time()

kp_left, des_left, left_rgb = SIFT(cylindricalWarpImage(cv2.imread("./7533.jpg"),1015))
kp_right, des_right, right_rgb = SIFT(cylindricalWarpImage(cv2.imread("./7511.jpg"),1027))

 
plot_keypoint(kp_left,left_rgb.copy(),kp_right,right_rgb.copy())

matches = matcher(kp_left, des_left, kp_right, des_right)
total_img = np.concatenate((left_rgb, right_rgb), axis=1)

plot_matches(matches, total_img) # Good mathces

reuslt = stitch_img(left_rgb, right_rgb, ransac(matches))

end = time.time()
print(end - start)

plt.imshow(reuslt)
plt.show()



