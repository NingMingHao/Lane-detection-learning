#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:48:21 2019

@author: minghao
"""

import cv2
import os


to_do_file_list = []
folder_name = 'fail'
all_file = os.listdir(folder_name)
for each_file in all_file:
    if each_file.split('.')[-1]=='jpg' and not(each_file.split('.')[0][-2:]=='my'):
        to_do_file_list.append(each_file)

for to_do_file in to_do_file_list:
    to_do_name = to_do_file.split('.')[0]
    img_my_path = folder_name+'/'+to_do_file
    
    img_my = cv2.imread(img_my_path)
    
    label = img_my[26:,:224]
    my = img_my[26:,236:]
    
    cv2.imwrite(os.path.join(folder_name,to_do_name+'__my.jpg'),my)
    cv2.imwrite(os.path.join(folder_name,to_do_name+'__label.jpg'),label)
    
    
#to_do_file_list = []
#folder_name = 'test'
#all_file = os.listdir(folder_name)
#for each_file in all_file:
#    if each_file.split('.')[-1]=='jpg' and not(each_file.split('.')[0][-2:]=='my'):
#        to_do_file_list.append(each_file)
#
#for to_do_file in to_do_file_list:
#    to_do_name = to_do_file.split('.')[0]
#    my_name = to_do_name+'_my.jpg'
#    img_my_path = folder_name+'/'+my_name
#    img_dreyeve_path = folder_name+'/'+to_do_file
#    
#    img_my = cv2.imread(img_my_path)
#    img_dreyeve = cv2.imread(img_dreyeve_path)
#    
#    label = img_my[26:,:224]
#    my = img_my[26:,236:]
#    dreyeve = img_dreyeve[26:,236:]
#    
#    cv2.imwrite(os.path.join(folder_name,to_do_name+'__my.jpg'),my)
#    cv2.imwrite(os.path.join(folder_name,to_do_name+'__label.jpg'),label)
#    cv2.imwrite(os.path.join(folder_name,to_do_name+'__dreyeve.jpg'),dreyeve)
#    
#    cv2.imshow('label',label)
#    cv2.imshow('my',my)
#    cv2.imshow('dreyeve',dreyeve)
#    cv2.waitKey(2000)