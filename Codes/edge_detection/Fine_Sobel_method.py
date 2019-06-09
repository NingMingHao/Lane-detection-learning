#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 20:56:53 2019

@author: minghao
"""

import cv2
import numpy as np


def warpFrame(frame, src_points, dst_points):
    frame_size = (frame.shape[1], frame.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_frame = cv2.warpPerspective(frame, M, frame_size, flags=cv2.INTER_LINEAR)
    
    return warped_frame, M, Minv


imgs_path = '/Users/mac/Documents/University/Github/Lane-detection-learning/imgs/'
img = cv2.imread(imgs_path+'um_000063.png') #00,32,63,81

#def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#    if event == cv2.EVENT_LBUTTONDOWN:
#        xy = "%d,%d" % (x, y)
#        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                    1.0, (0, 0, 0), thickness=1)
#        cv2.imshow("image", img)
#
#
#cv2.namedWindow("image")
#cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
#cv2.imshow("image", img)


src = np.float32([ [576,207], [645,207], [848,374], [420,374] ])
dst = np.float32([ [420,0], [848,0], [848,374], [420,374] ])
test_warp_image, M, Minv = warpFrame(img, src, dst)
cv2.imshow('warp',test_warp_image)
def absSobelThreshold(img, orient='x', min_thre=60, max_thre=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute( cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute( cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = cv2.inRange(scaled_sobel, min_thre, max_thre)/255
    return binary_output

c = absSobelThreshold(test_warp_image)
cv2.imshow('c',c)