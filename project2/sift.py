from math import floor
from operator import le
import cv2
import sys
import numpy as np
import time
import random
from tqdm.notebook import tqdm
from cv2 import GaussianBlur,sqrt,resize
from matplotlib import pyplot as plt
from numpy import array, zeros,sqrt,log,subtract,all
from numpy.linalg import lstsq, norm

float_tolerance = 1e-7

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

def Find_Keypoints(gaussian_images, dogs):
    keypoints = []
    for octave_index, dog in enumerate(dogs):
        for image_index in range(1,len(dog)-1):
            image0, image1,image2 = dog[image_index-1:image_index+2]
            height , width = image0.shape
            for i in range(1, height - 2):
                for j in range(1, width - 2):
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
                            if iteration == 4 or ii < 1 or jj < 1 or ii > height - 1  or jj > width - 1 or _image_index < 1 or _image_index > 3: 
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
                                    , 1.6 * (2 ** ((_image_index + approximation[2] -1) / 3.)) #size
                                    , -1 # angle
                                    , abs(response) # response
                                    , octave_index + _image_index * (2 ** 8)) # octave

                                    gaussian_image = gaussian_images[octave_index][_image_index]
                                    height, width = gaussian_image.shape
                                    scale = 1.5 * keypoint.size  
                                    radius = round(3 * scale)
                                    histogram = zeros(36)
                                    smooth_histogram = zeros(36)
                                    for a in range(-radius, radius + 1):
                                        for b in range(-radius, radius + 1):
                                            y = round(keypoint.pt[1]) + a
                                            x = round(keypoint.pt[0]) + b
                                            if y > 0 and y < height - 1 and x > 0 and x < width - 1:
                                                Lx = gaussian_image[y, x + 1] - gaussian_image[y, x - 1]
                                                Ly = gaussian_image[y - 1, x] - gaussian_image[y + 1, x]
                                                m = sqrt(Lx * Lx + Ly * Ly)
                                                theta =  np.rad2deg(np.arctan2(Ly, Lx))
                                                histogram[round(theta / 10.) % 36] += np.exp(-0.5 / (scale ** 2) * (a ** 2 + b ** 2)) * m            
                                    
                                    orientation_max = max(histogram)
                                    for peak_index in range(len(histogram)):
                                        if histogram[peak_index] >= 0.8 * orientation_max: # Make description more reliable
                                            orientation = 360. - (peak_index + 0.5 * (histogram[(peak_index - 1) % 36] - histogram[(peak_index + 1) % 36]) 
                                                        / (histogram[(peak_index - 1) % 36] - 2 * histogram[peak_index] + histogram[(peak_index + 1) % 36])) % 36 * 10.
                                            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                                            keypoints.append(new_keypoint)
        
                                    # for n in range(36):
                                    #     smooth_histogram[n] = (6 * histogram[n] + 4 * (histogram[n - 1] + histogram[(n + 1) % 36]) + histogram[n - 2] + histogram[(n + 2) % 36]) / 16.
                                    # orientation_max = max(smooth_histogram)
                                    # orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
                                    # for peak_index in orientation_peaks:
                                    #     peak_value = smooth_histogram[peak_index]
                                    #     if peak_value >= 0.8 * orientation_max:
                                    #         # Quadratic peak interpolation
                                    #         # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
                                    #         left_value = smooth_histogram[(peak_index - 1) % 36]
                                    #         right_value = smooth_histogram[(peak_index + 1) % 36]
                                    #         interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % 36
                                    #         orientation = 360. - interpolated_peak_index * 360. / 36
                                    #         if abs(orientation - 360.) < float_tolerance:
                                    #             orientation = 0
                                    #         new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                                    #         keypoints.append(new_keypoint)
    print(len(keypoints))  
    return keypoints

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.size *= 1
        keypoint.octave = (keypoint.octave & 255)
        keypoint.pt = tuple(0.5 * array(keypoint.pt) * 2  ** keypoint.octave)
        converted_keypoints.append(keypoint)   
    return converted_keypoints


def Trilinear_interpolation(row_list, col_list, orientation_list, m_list):
    histogram = zeros((6, 6, 8))   
    for i, j, k, m  in zip(row_list, col_list, orientation_list, m_list):
            ii, jj, kk = np.floor([i, j, k]).astype(int)

            if kk < 0:
                kk += 8
            if kk >= 8:
                kk -= 8

            c0 = m * (1 - (i - ii))
            c1 = m * (i - ii)
            c00 = c0 * (1 - (j - jj))
            c01 = c0 * (j - jj)
            c10 = c1 * (1 - (j - jj))
            c11 = c1 * (j - jj)
            c000 = c00 * (1 - (k - kk))
            c001 = c00 * (k - kk)
            c010 = c01 * (1 - (k - kk))
            c011 = c01 * (k - kk)
            c100 = c10 * (1 - (k - kk))
            c101 = c10 * (k - kk)
            c110 = c11 * (1 - (k - kk))
            c111 = c11 * (k - kk)
            
            histogram[ii + 1, jj + 1, kk] += c000
            histogram[ii + 1, jj + 1, (kk + 1) % 8] += c001
            histogram[ii + 1, jj + 2, kk] += c010
            histogram[ii + 1, jj + 2, (kk + 1) % 8] += c011
            histogram[ii + 2, jj + 1, kk] += c100
            histogram[ii + 2, jj + 1, (kk + 1) % 8] += c101
            histogram[ii + 2, jj + 2, kk] += c110
            histogram[ii + 2, jj + 2, (kk + 1) % 8] += c111

    return histogram[1:-1, 1:-1, :].flatten()

def Generate_Descriptors(keypoints, gaussian_images, window_size=4):

    descriptors = []

    for keypoint in keypoints:
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)

        gaussian_image = gaussian_images[octave, layer]

        height, width = gaussian_image.shape
        point = np.round(scale * array(keypoint.pt))

        cos_angle = np.cos(np.deg2rad(-keypoint.angle))
        sin_angle = np.sin(np.deg2rad(-keypoint.angle))

        row_list,col_list,m_list,orientation_list = [], [], [], []

        oringe = 1.5 * scale * keypoint.size
        half = int(min(round(oringe * sqrt(2) * (window_size + 1) * 0.5) , sqrt(height ** 2 + width ** 2)))    

        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                row_orientation = j * sin_angle + i * cos_angle
                col_orientation = j * cos_angle - i * sin_angle
                row = (row_orientation / oringe) + 0.5 * window_size - 0.5
                col = (col_orientation / oringe) + 0.5 * window_size - 0.5
                if row > -1 and row < window_size and col > -1 and col < window_size:
                    window_row, window_col = round(point[1] + i), round(point[0] + j)
                    if window_row > 0 and window_row < height - 1 and window_col > 0 and window_col < width - 1:
                        Lx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        Ly = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        m = sqrt(Lx * Lx + Ly * Ly)
                        theta = np.rad2deg(np.arctan2(Ly, Lx)) % 360
                        row_list.append(row)
                        col_list.append(col)
                        m_list.append(np.exp(-0.5 / ((0.5 * window_size) ** 2) * ((row_orientation / oringe) ** 2 + (col_orientation / oringe) ** 2)) * m)
                        orientation_list.append((theta + keypoint.angle) * 8 / 360.)

        descriptor_128 = Trilinear_interpolation(row_list, col_list, orientation_list, m_list)
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

def SIFT(path):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE).astype('float32')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    image = GaussianBlur(image, (0, 0), sigmaX=1.24, sigmaY=1.24) #1.6
    gaussian, dog = DoG(image)
    keypoints = Find_Keypoints(gaussian,dog)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    #descriptors = Generate_Descriptors(keypoints, gaussian)
    #return keypoints,descriptors

    return keypoints

def matcher(kp1, des1, img1, kp2, des2, img2, threshold = 0.5):
    #BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    # matches = []
    # if(len(kp1)<len(kp2)):
    #     for i in range(len(kp1)):
    #         first_min = sys.maxsize
    #         second_min = first_min
    #         for j in range(len(kp2)):
    #             sum = np.sqrt(np.sum((des1[i] - des2[j]) ** 2))
    #             if(sum < first_min):
    #                 second_min = first_min
    #                 first_min = sum
    #                 matchpoint = kp2[j]
    #         if first_min / second_min < threshold and first_min:
    #             print(first_min)
    #             matches.append(list(kp1[i].pt + matchpoint.pt))
    # else:
    #     for i in range(len(kp2)):
    #         first_min = sys.maxsize
    #         second_min = first_min
    #         for j in range(len(kp1)):
    #             sum = np.sqrt(np.sum((des1[j] - des2[i]) ** 2))
    #             if(sum < first_min):
    #                 second_min = first_min
    #                 first_min = sum
    #                 matchpoint = kp1[j]
    #         if first_min / second_min < threshold and first_min:
    #             print(first_min)
    #             matches.append(list(matchpoint.pt + kp2[i].pt))

    matches = np.array(matches)

    return matches*2

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def stitch_img(left, right, H):
    print("stiching image ...")
    
    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)

    height_r, width_r, channel_r = right.shape
    
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)
    

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
     
    black = np.zeros(3)  # Black pixel.
    
    # Stitching procedure, store results in warped_l.
    for i in tqdm(range(warped_r.shape[0])):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()

start = time.time()

# left_rgb = cv2.imread("instance.png")
# left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)
# right_rgb = cv2.imread("NTUST2.png")
# right_rgb = cv2.cvtColor(right_rgb, cv2.COLOR_BGR2RGB)

# kp_left, des_left = SIFT("instance.png")
# kp_right, des_right = SIFT("NTUST2.png")



# matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb)

# total_img = np.concatenate((left_rgb, right_rgb), axis=1)
# plot_matches(matches, total_img) # Good mathces
# inliers, H = ransac(matches, 0.5, 2000)

# plt.imshow(stitch_img(left_rgb, right_rgb, H))


image0 = cv2.imread("box.png")
image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
keypoints = SIFT("box.png")
for i in keypoints:
    cv2.circle(image0, (int(i.pt[0]),int(i.pt[1])), radius=2, color=(255, 0, 0), thickness=-1)

plt.imshow(image0)
plt.show()


end = time.time()
print(end - start)

#Dog_visualization(dog)