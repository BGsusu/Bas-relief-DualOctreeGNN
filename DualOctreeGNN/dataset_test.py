# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:07:41 2023

@author: 86186
"""
import os
filenames=[]
filelist='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/train.txt'
false_num=0
with open(filelist) as fid:
          lines = fid.readlines()
          for line in lines:
            filename = line.replace('\n', '')
            label = 0
            for i in range(0,3):
                for j in range(0,5):
                    '''
                    filename_pts='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/'+filename+'/BaseRelief{}_{}_pc.npz'.format(i,j)  
                    filename_out='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/'+filename+'/BaseRelief{}_{}_sdf.npz'.format(i,j)
                    if os.path.exists(filename_pts)==False:continue
                    if os.path.exists(filename_out)==False:continue#from 0_0-2_4
                    '''
                    filename_pts='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/'+filename+'/BaseRelief{}_{}_pc.npz'.format(i,j)
                    if os.path.exists(filename_pts)==False:
                        false_num+=1
                        continue
                    filenames.append(filename+'/BaseRelief{}_{}'.format(i,j) )

num = len(filenames)
print(num)
print(false_num)