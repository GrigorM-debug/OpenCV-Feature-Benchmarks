import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('image_left.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('image_rigth.jpg', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT.create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   
 
flann = cv.FlannBasedMatcher(index_params,search_params)
 
matches = flann.knnMatch(des1,des2,k=2)

good_matches = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()