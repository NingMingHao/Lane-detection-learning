#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:22:46 2019

@author: minghao
"""

default_filename = '/Users/mac/Documents/University/Github/data_road/training/calib/um_000000.txt'

import os
import numpy as np

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
    
    def convertPositionMetric2Pixel(self, YXpointArrays):
        allY = YXpointArrays[:,0]
        allX = YXpointArrays[:,1]
        allYconverted = self.bev_size[0] - self.meter2px(allY - self.bev_zLimits[0])
        allXconverted = self.meter2px(allX - self.bev_xLimits[0])
        return np.array( [allYconverted,allXconverted] ).T
    
    def convertPositionPixel2Metric(self, YXpointArrays):
        allY = YXpointArrays[:,0]
        allX = YXpointArrays[:,1]
        allYconverted = self.px2meter(self.bev_size[0] - allY) + self.bev_zLimits[0]
        allXconverted = self.px2meter(allX) + self.bev_xLimits[0]
        return np.array( [allYconverted,allXconverted] ).T
    
    def convertPositionPixel2Metric2(self, inputTupleY, inputTupleX):
        result_arr = self.convertPositionPixel2Metric(np.array( [[inputTupleY],[inputTupleX]] ))
        return (result_arr[0,0], result_arr[0,1])
    
def readKittiCalib(filename, dtype = np.float64):
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
        dtype = np.float64
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
    invalid_value = float('-INFINITY')
    def __init__(self, bev_res= 0.05, bev_xRange_minMax = (-10, 10), bev_zRange_minMax = (6, 46)):
        self.calib = KittiCalibration()
        bev_res = bev_res
        self.bevParams = BevParams(bev_res, bev_xRange_minMax, bev_zRange_minMax, self.imSize)
    
    def world2image(self, X_world, Y_world, Z_world):
        if not type(Y_world) == np.ndarray:
            Y_world = np.ones_like(Z_world)*Y_world
        y = np.vstack( (X_world, Y_world, Z_world, np.ones_like(Z_world)) )
        test = self.world2image_uvMat(np.vstack( (X_world, Z_world, np.ones_like(Z_world)) ))
        
        self.xi1 = test[0,:]
        self.yi1 = test[1,:]
        
        assert self.imSize != None
        condition = ~((self.yi1 >= 1) & (self.xi1 >= 1) & (self.yi1 <= self.imSize[0]) & (self.xi1 <= self.imSize[1]))
        if isinstance(condition, np.ndarray):
            self.xi1[condition] = self.invalid_value
            self.yi1[condition] = self.invalid_value
        elif condition == True:
            self.xi1 = self.invalid_value
            self.yi1 = self.invalid_value
            
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
    
    def compute(self, data):
        '''compute BEV'''
        self.imSize = data.shape
        self.computeBEVLookUpTable()
        return self.transformImage2BEV(data, out_dtype = data.dtype)
    
    def compute_reverse(self, data, imSize):
        self.imSize = imSize
        self.computeBEVLookUpTable_reverse()
        return self.transformBEV2Image(data, out_dtype = data.dtype)
    
    def computeBEVLookUpTable_reverse(self, imSize = None):
        mgrid = np.mgrid
        
        if imSize == None:
            imSize = self.imSize
        self.imSize_back = (imSize[0], imSize[1], )
        yx_im = (mgrid[1:self.imSize_back[0] + 1,  1:self.imSize_back[1] + 1]).astype(np.int32)
        
        y_im = yx_im[0]
        x_im = yx_im[1]
        
        dim = self.imSize_back[0] * self.imSize_back[1]
        uvMat = np.vstack( (x_im.flatten(), y_im.flatten(), np.ones((dim,) )))
        xzMat = self.image2world_uvMat(uvMat)
#        X = xzMat[0,:].reshape(x_im.shape)
#        Z = xzMat[1,:].reshape(x_im.shape)
        ZX = xzMat[[1,0],:]
        allZX_Ind_reverse_all = np.round(self.bevParams.convertPositionMetric2Pixel(ZX.T))
        XBevInd_reverse_all = allZX_Ind_reverse_all[:,1]
        ZBevInd_reverse_all = allZX_Ind_reverse_all[:,0]
        self.validMapIm_reverse = (XBevInd_reverse_all >= 1) & (XBevInd_reverse_all <= self.bevParams.bev_size[1]) & (ZBevInd_reverse_all >= 1) & (ZBevInd_reverse_all <= self.bevParams.bev_size[0])
        
        self.XBevInd_reverse = np.int32(XBevInd_reverse_all[self.validMapIm_reverse] - 1)
        self.ZBevInd_reverse = np.int32(ZBevInd_reverse_all[self.validMapIm_reverse] - 1)

        self.xImInd_reverse = x_im.flatten()[self.validMapIm_reverse] - 1
        self.yImInd_reverse = y_im.flatten()[self.validMapIm_reverse] - 1
    
    def computeBEVLookUpTable(self, cropping_ul = None, cropping_size = None):
        '''

        @param cropping_ul:
        @param cropping_size:
        '''

        # compute X,Z mesh from BEV params
        mgrid = np.mgrid
        
        res = self.bevParams.bev_res

        x_vec = np.arange(self.bevParams.bev_xLimits[0] + res / 2, self.bevParams.bev_xLimits[1], res)
        z_vec = np.arange(self.bevParams.bev_zLimits[1] - res / 2, self.bevParams.bev_zLimits[0], -res)
        XZ_mesh = np.meshgrid(x_vec, z_vec)


        assert XZ_mesh[0].shape == self.bevParams.bev_size


        Z_mesh_vec = (np.reshape(XZ_mesh[1], (self.bevParams.bev_size[0] * self.bevParams.bev_size[1]), order = 'F')).astype('f4')
        X_mesh_vec = (np.reshape(XZ_mesh[0], (self.bevParams.bev_size[0] * self.bevParams.bev_size[1]), order = 'F')).astype('f4')


        self.world2image(X_mesh_vec, 0, Z_mesh_vec)
        # output-> (y, x)
        if (cropping_ul is not None):
            valid_selector = np.ones((self.bevParams.bev_size[0] * self.bevParams.bev_size[1],), dtype = 'bool')
            valid_selector = valid_selector & (self.yi1 >= cropping_ul[0]) & (self.xi1 >= cropping_ul[1])
            if (cropping_size is not None):
                valid_selector = valid_selector & (self.yi1 <= (cropping_ul[0] + cropping_size[0])) & (self.xi1 <= (cropping_ul[1] + cropping_size[1]))
            # using selector to delete invalid pixel
            selector = (~(self.xi1 == self.invalid_value)).reshape(valid_selector.shape) & valid_selector  # store invalid value positions
        else:
            # using selector to delete invalid pixel
            selector = ~(self.xi1 == self.invalid_value)  # store invalid value positions

        y_OI_im_sel = self.yi1[selector]# without invalid pixel
        x_OI_im_sel = self.xi1[selector]# without invalid pixel


        # indices for bev positions for the LookUpTable
        ZX_ind = (mgrid[1:self.bevParams.bev_size[0] + 1, 1:self.bevParams.bev_size[1] + 1]).astype('i4')
        Z_ind_vec = np.reshape(ZX_ind[0], selector.shape, order = 'F')
        X_ind_vec = np.reshape(ZX_ind[1], selector.shape, order = 'F')

        # Select
        Z_ind_vec_sel = Z_ind_vec[selector] # without invalid pixel
        X_ind_vec_sel = X_ind_vec[selector] # without invalid pixel

        # Save stuff for LUT in BEVParams
        self.im_u_float = x_OI_im_sel
        self.im_v_float = y_OI_im_sel
        self.bev_x_ind = X_ind_vec_sel.reshape(x_OI_im_sel.shape)
        self.bev_z_ind = Z_ind_vec_sel.reshape(y_OI_im_sel.shape)


    def transformImage2BEV(self, inImage, out_dtype = 'f4'):
        '''
        
        :param inImage:
        '''
        assert not self.im_u_float is None
        assert not self.im_v_float is None
        assert not self.bev_x_ind is None
        assert not self.bev_z_ind is None
        
        
        if len(inImage.shape) > 2:
            outputData = np.zeros(self.bevParams.bev_size + (inImage.shape[2],), dtype = out_dtype)
            for channel in range(0, inImage.shape[2]):
                outputData[self.bev_z_ind-1, self.bev_x_ind-1, channel] = inImage[self.im_v_float.astype('u4')-1, self.im_u_float.astype('u4')-1, channel]
        else:
            outputData = np.zeros(self.bevParams.bev_size, dtype = out_dtype)
            outputData[self.bev_z_ind-1, self.bev_x_ind-1] = inImage[self.im_v_float.astype('u4')-1, self.im_u_float.astype('u4')-1]
        
        return  outputData
    
    def transformBEV2Image(self, bevMask, out_dtype = 'f4'):
        '''

        @param bevMask:
        '''
        assert not self.xImInd_reverse is None
        assert not self.yImInd_reverse is None
        assert not self.XBevInd_reverse is None
        assert not self.ZBevInd_reverse is None
        assert not self.imSize_back is None
        if len(bevMask.shape) > 2:
            outputData = np.zeros(self.imSize_back + (bevMask.shape[2],), dtype = out_dtype)
            for channel in range(0, bevMask.shape[2]):
                outputData[self.yImInd_reverse, self.xImInd_reverse, channel] = bevMask[self.ZBevInd_reverse, self.XBevInd_reverse, channel]
        else:
            outputData = np.zeros(self.imSize_back, dtype = out_dtype)
            outputData[self.yImInd_reverse, self.xImInd_reverse] = bevMask[self.ZBevInd_reverse, self.XBevInd_reverse]
        # Return result
        return outputData        
        