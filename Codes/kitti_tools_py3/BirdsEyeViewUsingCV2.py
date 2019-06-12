#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:47:12 2019

@author: minghao
"""

default_filename = '/Users/mac/Documents/University/Github/data_road/training/calib/um_000000.txt'

import os
import numpy as np
import cv2

class BevParams(object):
    def __init__(self, bev_res, bev_xLimits, bev_zLimits, imSize):
        bev_size = (round((bev_zLimits[1] - bev_zLimits[0]) / bev_res), \
                   round((bev_xLimits[1] - bev_xLimits[0]) / bev_res))
        self.bev_size = bev_size
        self.bev_res = bev_res
        self.bev_xLimits = bev_xLimits
        self.bev_zLimits = bev_zLimits
        self.imSize = imSize
        
    def px2meter(self, px_in):
        return px_in * self.bev_res
    
    def meter2px(self, meter_in):
        return meter_in / self.bev_res  #to_decide?
    
    def convertPositionMetric2Pixel(self, XZpointArrays):
        allX = XZpointArrays[:,0]
        allZ = XZpointArrays[:,1]
        allZconverted = self.bev_size[0] - self.meter2px(allZ - self.bev_zLimits[0])
        allXconverted = self.meter2px(allX - self.bev_xLimits[0])
        return np.float32( [allXconverted, allZconverted] ).T
    
    def convertPositionPixel2Metric(self, XYpointArrays):
        allX = XYpointArrays[:,0]
        allY = XYpointArrays[:,1]
        allYconverted = self.px2meter(self.bev_size[0] - allY) + self.bev_zLimits[0]
        allXconverted = self.px2meter(allX) + self.bev_xLimits[0]
        return np.float32( [allXconverted, allYconverted] ).T
    
    def convertPositionPixel2Metric2(self, inputTupleY, inputTupleX):
        result_arr = self.convertPositionPixel2Metric(np.array( [[inputTupleY],[inputTupleX]] ))
        return (result_arr[0,0], result_arr[0,1])
    
def readKittiCalib(filename, dtype = np.float32):
    out_dict = {}
    with open(filename,'rb') as f:
        allcontent = f.readlines()
    for contentRaw in allcontent:
        content = contentRaw.strip()
        if len(content) == 0:
            continue
        if content[0]!='#':
            tmp = content.decode().split(':')
            assert len(tmp)==2, 'wrong file format, only one : per line!'
            var = tmp[0].strip()
            values = np.array(tmp[-1].strip().split(' '),dtype)
            out_dict[var] = values
    return out_dict

class KittiCalibration(object):
    def __init__(self):
        pass
    
    def readFromFile(self,filename = default_filename):
        cur_calibStuff_dict = readKittiCalib(filename)
        self.setup(cur_calibStuff_dict)
    
    def setup(self, dictWithKittiStuff, useRect = False):
        dtype = np.float32
        self.P2 = np.matrix(dictWithKittiStuff['P2']).reshape((3,4))
        
        if useRect:
            R2_1 = self.P2
        else:
            R0_rect_raw = np.array(dictWithKittiStuff['R0_rect']).reshape((3,3))
            self.R0_rect = np.matrix(np.hstack((np.vstack((R0_rect_raw, np.zeros((1,3), dtype))), np.zeros((4,1), dtype))))
            self.R0_rect[3,3]=1.0
            R2_1 = np.dot(self.P2, self.R0_rect)
            
        Tr_cam_to_road_raw = np.array( dictWithKittiStuff['Tr_cam_to_road'] ).reshape(3,4)
        self.Tr_cam_to_road_raw = np.matrix( np.vstack((Tr_cam_to_road_raw, np.zeros((1,4), dtype))) )
        self.Tr_cam_to_road_raw[3,3] = 1.0
        
        self.Tr = np.dot( R2_1, self.Tr_cam_to_road_raw.I )
        self.Tr33 = self.Tr[:,[0,2,3]]
    
    def get_matrix33(self):
        assert not self.Tr33 is None
        return self.Tr33
    
class BirdsEyeView(object):
    imSize = None
    def __init__(self, bev_res= 0.05, bev_xRange_minMax = (-10, 10), bev_zRange_minMax = (6, 46)):
        self.calib = KittiCalibration()
        bev_res = bev_res
        self.bevParams = BevParams(bev_res, bev_xRange_minMax, bev_zRange_minMax, self.imSize)
        self.srcPoints = np.float32([ [0,0], [200,0], [200,200], [0,200] ])
            
    def world2image_uvMat(self, uv_mat):
        if uv_mat.shape[0] == 2:
            if len(uv_mat.shape) == 1:
                uv_mat = uv_mat[:,np.newaxis]
            uv_mat = np.vstack( (uv_mat, np.ones((1,uv_mat.shape[1]))))
        result = np.dot( self.Tr33, uv_mat )
        resultB = np.broadcast_arrays(result, result[-1, :])
        return resultB[0] / resultB[1]
    
    def image2world_uvMat(self, uv_mat):
        if uv_mat.shape[0] == 2:
            if len(uv_mat.shape)==1:
                uv_mat = uv_mat[:,np.newaxis]
            uv_mat = np.vstack((uv_mat, np.ones((1, uv_mat.shape[1]))))
        result = np.dot(self.Tr33.I, uv_mat)
        #w0 = -(uv_mat[0]* self.Tr_inv_33[1,0]+ uv_mat[1]* self.Tr_inv[1,1])/self.Tr_inv[1,2]
        resultB = np.broadcast_arrays(result, result[-1,:])
        return resultB[0] / resultB[1]
    
    def setup(self, calib_file):
        self.calib.readFromFile( filename=calib_file )
        self.set_matrix33(self.calib.get_matrix33())
    
    def set_matrix33(self, matrix33):
        self.Tr33 = matrix33
    
    def computeM_reverse(self):
        uvMat = np.vstack( (self.srcPoints.T, np.ones(4)) )
        xzMat = self.image2world_uvMat(uvMat)
        XZ = xzMat[[0,1],:]
        
        allXZ_Ind_reverse_all = self.bevParams.convertPositionMetric2Pixel(XZ.T)

        return cv2.getPerspectiveTransform(allXZ_Ind_reverse_all, self.srcPoints)
    
    def computeM(self):
        xyWorld = self.bevParams.convertPositionPixel2Metric(self.srcPoints)
        uvMat = np.vstack( (xyWorld.T, np.ones(4)) )
        xyMat = self.world2image_uvMat(uvMat)
        XY = np.float32(xyMat[[0,1],:])
        return cv2.getPerspectiveTransform(XY.T, self.srcPoints)
