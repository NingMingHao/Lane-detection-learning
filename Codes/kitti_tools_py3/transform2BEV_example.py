#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:16:19 2019

@author: minghao
"""


from BirdsEyeViewUsingCV2 import BirdsEyeView   #Using cv2 warpPerspective
#from BirdsEyeView import BirdsEyeView           #Don't use cv2 warpPerspective
import cv2
import numpy as np
import os

dataset_path = '/Users/mac/Documents/University/Github/data_road/training'
imgs_path = os.path.join(dataset_path, 'image_2')
calibs_path = os.path.join(dataset_path, 'calib')
all_image_names_list = os.listdir(imgs_path)

img_name = all_image_names_list[0]
calib_name = img_name.split('.')[0] + '.txt'

img_path = os.path.join(imgs_path,img_name)
calib_path = os.path.join(calibs_path, calib_name)

bev = BirdsEyeView(bev_zRange_minMax=(6,48))
bev.setup(calib_path)

img = cv2.imread(img_path)


#use function defined in BirdsEyeView
# =============================================================================
# img_bev = bev.compute(img)
# img_gen = bev.compute_reverse(img_bev, (370,1200))
# 
# cv2.imshow('img',img)
# cv2.imshow('bev',img_bev)
# cv2.imshow('img_gen',img_gen)
# =============================================================================


#use cv2 warpPerspective
M = np.matrix(bev.computeM())
M_inv = np.matrix(bev.computeM_reverse())

img_bev = cv2.warpPerspective(img, M, bev.bevParams.bev_size[::-1])

cv2.imshow('img',img)
cv2.imshow('bev',img_bev)


img_gen = cv2.warpPerspective(img_bev, M_inv, (1200,370))
cv2.imshow('img_gen',img_gen)

