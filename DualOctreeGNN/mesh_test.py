# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:17:32 2023

@author: 86186
"""
import argparse

import ocnn
import torch
import torch.autograd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
parser = argparse.ArgumentParser(description='INPUT PATH')
 
# 2. 添加命令行参数
parser.add_argument('--pth', type=str)
args = parser.parse_args()

sdf_values=np.load(args.pth)
def create_mesh(filename, size=256, max_batch=64**3, level=0,
                bbmin=-0.9, bbmax=0.9, mesh_scale=1.0, save_sdf=False, **kwargs):
  # marching cubes
  filename_sdf=filename+'.sdf'
  np.save(filename_sdf,sdf_values)
  vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
  try:
    vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level)
  except:
    pass
  if vtx.size == 0 or faces.size == 0:
    print('Warning from marching cubes: Empty mesh!')
    return

  # normalize vtx
  vtx = vtx * ((bbmax - bbmin) / size) + bbmin   # [0,sz]->[bbmin,bbmax]
  vtx = vtx * mesh_scale                         # rescale

  # save to ply and npy
  mesh = trimesh.Trimesh(vtx, faces)
  mesh.export(filename)
  if save_sdf:
    np.save(filename[:-4] + ".sdf.npy", sdf_values)
    
create_mesh('my_mesh.obj')