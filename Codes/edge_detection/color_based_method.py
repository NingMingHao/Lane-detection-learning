#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:36:20 2019

@author: minghao
"""

import cv2
import numpy as np

def get_vertices(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    left_bottom = [220, height]
    right_bottom = [width-220,height]
    apex = [width/2, 180]
    vertices = np.array([left_bottom, right_bottom, apex], np.int32)
    return vertices

def get_roi(frame, vertices):
    mask = np.zeros_like(frame)
    if len(frame.shape) > 2:
        channel_count = frame.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image


def warpFrame(frame, ratio, src_points, dst_points):
    frame_size = (int(ratio[0]*frame.shape[1]), int(ratio[1]*frame.shape[0]))
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_frame = cv2.warpPerspective(frame, M, frame_size, flags=cv2.INTER_LINEAR)
    
    return warped_frame, M, Minv


imgs_path = '/Users/mac/Documents/University/Github/Lane-detection-learning/imgs/'
img = cv2.imread(imgs_path+'um_000032.png') #00,32,63,81

vertices = get_vertices(img)
masked_img = get_roi(img, vertices)
cv2.imshow('mask',masked_img)

src = np.float32([ [553,234], [679,234], [848,374], [420,374] ])
dst = np.float32([ [420,0], [848,0], [848,750], [420,750] ])
test_warp_image, M, Minv = warpFrame(masked_img, (1,2), src, dst)
cv2.imshow('warp',test_warp_image)

def hlsLSelect(img, thre=(195,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    l_channel = l_channel*(255/np.max(l_channel))
    binary_output = cv2.inRange(l_channel,thre[0],thre[1])/255
    return binary_output

def labBSelect(img, thre=(210,255)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_b = lab[:,:,2]
    if np.max(lab_b) > 100:
        lab_b = lab_b*(255/np.max(lab_b))
    binary_output = cv2.inRange(lab_b,thre[0],thre[1])/255
    return binary_output

hlsL_binary = hlsLSelect(test_warp_image)
labB_binary = labBSelect(test_warp_image)
combined_binary = cv2.bitwise_or(hlsL_binary,labB_binary)
#cv2.imshow('combined',combined_binary)

def find_lane_pixels(combined_binary, nwindows, margin, minpix):
    all_height = combined_binary.shape[0]
    hist_start = int(all_height*0.7)
    histogram = np.sum(combined_binary[hist_start:,:], axis=0)
    out_img = np.dstack((combined_binary,)*3)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint
    
    window_height = all_height//nwindows
    nonzero = combined_binary.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = all_height - (window+1)*window_height
        win_y_high = all_height - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        cv2.rectangle(out_img, (win_xleft_low,win_y_low),
                      (win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img, (win_xright_low,win_y_low),
                      (win_xright_high,win_y_high),(0,255,0),2)
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx,lefty,rightx,righty,out_img

def fit_polynomial(combined_binary, nwindows=9, margin=100, minpix=50):
    leftx,lefty,rightx,righty,out_img = find_lane_pixels(combined_binary,
                                                         nwindows, margin, minpix)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_f = np.poly1d(left_fit)
    right_f = np.poly1d(right_fit)
    
    all_height = combined_binary.shape[0]
    ploty = np.linspace(0, all_height -1, all_height)
    try:
        left_fitx = left_f(ploty)
        right_fitx = right_f(ploty)
    except TypeError:
        print('The function failed to fit a line!')
    
    out_img[lefty,leftx] = [255,0,0]
    out_img[righty,rightx] = [0,0,255]
    
    out_img[np.int32(ploty),np.int32(left_fitx)] = [255,255,255]
    out_img[np.int32(ploty),np.int32(right_fitx)] = [255,255,255]
    
    return out_img, left_fit, right_fit, ploty

out_img, left_fit, right_fit, ploty = fit_polynomial(combined_binary)
cv2.imshow('out',out_img)