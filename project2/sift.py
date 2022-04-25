import cv2
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
    newimg = np.zeros((height, width*6, 3), np.uint8)
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
    layers_of_images_in_octave = s + 3
    k = 2 ** (1. / s)
    gaussian_sigmas  = zeros(layers_of_images_in_octave-1)

    for i in range(1,layers_of_images_in_octave):
        s_pre  = (k**(i-1)) * sigma; 
        s_post = (k**i)     * sigma; 
        gaussian_sigmas[i-1] = sqrt(s_post**2 - s_pre**2)

    # Generate Gaussian Images

    num_of_octave = round(log(min(image.shape)) / log(2) - 1)
    gaussian_images = []
    for i in range(num_of_octave):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)
        for sigma in gaussian_sigmas:
            image = GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            gaussian_images_in_octave.append(image)
        blur_image = gaussian_images_in_octave[-3]
        image = resize(blur_image, (int(blur_image.shape[1] / 2), int(blur_image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        gaussian_images.append(gaussian_images_in_octave)
     
    # Generate DOG

    dog_images = []
    for i in gaussian_images:
        sub_images = []
        for j in range(1,len(i)):
            sub_images.append(subtract(i[j],i[j-1]))
        dog_images.append(sub_images)

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
                    check = False
                    if (abs(center_pixel)>1):
                        if(center_pixel>0):
                            if(all(center_pixel >= image0[i-1:i+2, j-1:j+2]) and all(center_pixel >= image1[i-1:i+2, j-1:j+2]) and all(center_pixel >= image2[i-1:i+2, j-1:j+2])):
                                check = True
                        else:
                            if(all(center_pixel <= image0[i-1:i+2, j-1:j+2]) and all(center_pixel <= image1[i-1:i+2, j-1:j+2]) and all(center_pixel <= image2[i-1:i+2, j-1:j+2])):
                                check = True
                    if(check):
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
                            if all(abs(approximation)<0.5):
                                break
                            jj += round(approximation[0])
                            ii += round(approximation[1])
                            _image_index += round(approximation[2])

                            # Checking point is inside
                            if ii < 1 or jj < 1 or ii > height - 1  or jj > width - 1 or _image_index < 1 or _image_index > 3:
                                check = False
                                break
                        if check:
                            response = _center_pixel + 0.5 * np.dot(g, approximation)
                            if ((h[0,0] + h[1,1]) ** 2) / (h[0,0] * h[1,1] - h[0,1] * h[0,1]) < 12.1 :
                                # Keypoint 
                                keypoint = cv2.KeyPoint(
                                (jj+approximation[0]) * (2 ** octave_index), # X
                                (ii+approximation[1]) * (2 ** octave_index) # Y
                                , 1.6 * (2 ** ((_image_index + approximation[2]) / np.float32(3))) #size
                                , -1 # angle
                                , abs(response) # response
                                , octave_index + _image_index * (2 ** 8) + int(round((approximation[2] + 0.5) * 255)) * (2 ** 16)) # octave

                                gaussian_image = gaussian_images[octave_index][_image_index]
                                height, width = gaussian_image.shape
                                scale = 1.5 * keypoint.size  
                                radius = int(round(3 * scale))
                                histogram = zeros(36)

                                for a in range(-radius, radius + 1):
                                    for b in range(-radius, radius + 1):
                                        y = round(keypoint.pt[1] /  np.float32(2 ** octave_index)) + a
                                        x = round(keypoint.pt[0] /  np.float32(2 ** octave_index)) + b
                                        if y > 0 and y < height - 1 and x > 0 and x < width - 1:
                                            Lx = gaussian_image[y, x + 1] - gaussian_image[y, x - 1]
                                            Ly = gaussian_image[y - 1, x] - gaussian_image[y + 1, x]
                                            m = sqrt(Lx * Lx + Ly * Ly)
                                            theta =  np.rad2deg( np.arctan2(Ly, Lx))
                                            histogram_index = int(round(theta / 10.))
                                            histogram[histogram_index % 36] += np.exp(-0.5 / (scale ** 2) * (a ** 2 + b ** 2)) * m

                                orientation_max = max(histogram)
                                for peak_index in range(len(histogram)):
                                    if histogram[peak_index] >= 0.8 * orientation_max: # Make description more reliable
                                        orientation = 360. - (peak_index + 0.5 * (histogram[(peak_index - 1) % 36] - histogram[(peak_index + 1) % 36]) 
                                                    / (histogram[(peak_index - 1) % 36] - 2 * histogram[peak_index] + histogram[(peak_index + 1) % 36])) % 36 * 10.
                                        new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                                        keypoints.append(new_keypoint)               
    return keypoints

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, descriptor_max_value=0.2):
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave, layer]
        height, width = gaussian_image.shape
        point = np.round(scale * array(keypoint.pt)).astype('int')
        angle = -keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width+2, window_width+2, 8))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = 1.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(height ** 2 + width ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_orientation = col * sin_angle + row * cos_angle
                col_orientation = col * cos_angle - row * sin_angle
                row_bin = (row_orientation / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_orientation / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < height - 1 and window_col > 0 and window_col < width - 1:
                        Lx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        Ly = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        m = sqrt(Lx * Lx + Ly * Ly)
                        theta = np.rad2deg(np.arctan2(Ly, Lx)) % 360
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(np.exp(weight_multiplier * ((row_orientation / hist_width) ** 2 + (col_orientation / hist_width) ** 2)) * m)
                        orientation_bin_list.append((theta - angle) * 8 / 360.)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # Smoothing via trilinear interpolation
            # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += 8
            if orientation_bin_floor >= 8:
                orientation_bin_floor -= 8

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            #print(row_bin_floor)
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % 8] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % 8] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % 8] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % 8] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector = cv2.normalize(descriptor_vector, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')

def SIFT(path):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE).astype('float32')
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    image = GaussianBlur(image, (0, 0), sigmaX=1.24, sigmaY=1.24) #1.6
    gaussian, dog = DoG(image)
    keypoints = Find_Keypoints(gaussian,dog)
    descriptors = generateDescriptors(keypoints, gaussian)
    return keypoints,descriptors

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
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

    matches = np.array(matches)
    return matches

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

start = time.time()

left_rgb = cv2.imread("prtn100.jpg")
left_rgb = cv2.cvtColor(left_rgb, cv2.COLOR_BGR2RGB)
right_rgb = cv2.imread("prtn110.jpg")
right_rgb = cv2.cvtColor(right_rgb, cv2.COLOR_BGR2RGB)

kp_left, des_left = SIFT("prtn100.jpg")
kp_right, des_right = SIFT("prtn110.jpg")

matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

# total_img = np.concatenate((left_rgb, right_rgb), axis=1)
# plot_matches(matches, total_img) # Good mathces
inliers, H = ransac(matches, 0.5, 2000)

plt.imshow(stitch_img(left_rgb, right_rgb, H))

# for i in keypoints_left:
#     cv2.circle(image0, (int(i.pt[0]),int(i.pt[1])), radius=2, color=(255, 0, 0), thickness=-1)
end = time.time()
print(end - start)
plt.show()
#Dog_visualization(dog)