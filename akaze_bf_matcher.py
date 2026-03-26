import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('images/image_left.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('images/image_rigth.jpg', cv.IMREAD_GRAYSCALE)

akaze = cv.AKAZE.create()

kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good_matches = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()