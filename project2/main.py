import pysift
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('box.png',0)
image1 = cv2.imread('box.png')
image_ = cv2.imread('box_in_scene.png',0)
image1_ = cv2.imread('box_in_scene.png')
keypoints = pysift.computeKeypointsAndDescriptors(image)

for i in keypoints:
    cv2.circle(image1, (int(i.pt[0]),int(i.pt[1])), radius=0, color=(255, 0, 0), thickness=-1)
plt.imshow(image1)
plt.show()