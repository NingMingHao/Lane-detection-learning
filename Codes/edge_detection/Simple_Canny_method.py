#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:36:14 2019

@author: minghao
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def do_canny(frame, low_thre, high_thre):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15,15), 0)
    canny = cv2.Canny(blur, low_thre, high_thre)
    return canny

def get_vertices(canny_frame):
    height = canny_frame.shape[0]
    width = canny_frame.shape[1]
    
    left_bottom = [0, height]
    right_bottom = [width,height]
    apex = [width/2, 200]
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

def draw_lines(frame, lines, color=[255,0,0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(frame, (x1,y1), (x2,y2), color, thickness)
    return frame

def hough_transform(roi_canny_frame):
    rho = 2 #distance resolution in pixels of the Hough grid
    theta = np.pi / 90 #angular resolution in radians of the Hough grid
    threshold = 10 #minimum number of votes
    min_line_length = 50
    max_line_gap = 20 #maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(roi_canny_frame, rho, theta, threshold,
                            np.array([]), min_line_length, max_line_gap)
    return lines
    
def detection(frame):
    canny_frame = do_canny(frame, 40, 150)
    cv2.imshow('canny_frame', canny_frame)
    vertices = get_vertices(canny_frame)
    roi_canny_frame = get_roi(canny_frame, vertices)
    cv2.imshow('roi', roi_canny_frame)
    lines = hough_transform(roi_canny_frame)
    frame = draw_lines(frame, lines)
    cv2.imshow('detection',frame)


def main():
    imgs_path = '/Users/mac/Documents/University/Github/Lane-detection-learning/imgs/'
    frame = cv2.imread(imgs_path+'um_000081.png')
    detection(frame)
    
if __name__ == '__main__':
    main()