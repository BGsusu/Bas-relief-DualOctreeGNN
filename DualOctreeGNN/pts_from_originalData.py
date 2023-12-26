# -*- coding: utf-8 -*-
"""
Created on 6.20

@author: 铁血日常大王
"""

import torch
import numpy as np

import os
import trimesh
import mesh2sdf
from tqdm import tqdm
import ocnn

def gen_dataset():
  r''' Samples 10k points with normals from the ground-truth meshes.
  '''
  num_samples = 40000
  mesh_scale = 1

  print('-> Run sample_pts_from_mesh.')
  for i in tqdm(range(1,22)):
      for j in range(0,4):
                    filename_obj='./{}-{}.obj'.format(i,j)
                    filename_pts='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/relief/{}_{}_pc.npz'.format(i,j)   
                    
                    if os.path.exists(filename_obj)==False:continue
                    if os.path.exists(filename_pts):continue#from 0_0-2_4
                    mesh = trimesh.load(filename_obj, force='mesh')
                    if mesh==[]:continue
                    vertices = mesh.vertices
                    bbmin, bbmax = vertices.min(0), vertices.max(0)
                    center = (bbmin + bbmax) * 0.5
                    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
                    points, idx = trimesh.sample.sample_surface(mesh, num_samples)
                    points = (points - center) * scale
                    normals = mesh.face_normals[idx]

                    # save points
                    np.savez(filename_pts, points=points.astype(np.float16),
                             normals=normals.astype(np.float16))
                    
gen_dataset()