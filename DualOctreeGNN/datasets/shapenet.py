# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import ocnn
import torch
import numpy as np

from solver import Dataset
from .utils import collate_func

# 处理读取到的数据
class TransformShape:
  
  def __init__(self, flags):
    self.flags = flags
    self.point_sample_num = 3000
    self.sdf_sample_num = 5000
    self.points_scale = 0.5  # the points are in [-0.5, 0.5]
    self.noise_std = 0.005
    self.points2octree = ocnn.Points2Octree(**flags)

  def process_points_cloud(self, sample):
    # get the input
    points, normals = sample['points'], sample['normals']
    points = points / self.points_scale  # scale to [-1.0, 1.0]

    # transform points to octree
    points_gt = ocnn.points_new(
        torch.from_numpy(points).float(), torch.from_numpy(normals).float(),
        torch.Tensor(), torch.Tensor())
    points_gt, _ = ocnn.clip_points(points_gt, [-1.0]*3, [1.0]*3)
    octree_gt = self.points2octree(points_gt)

    if self.flags.distort:
      # randomly sample points and add noise
      # Since we rescale points to [-1.0, 1.0] in Line 24, we also need to
      # rescale the `noise_std` here to make sure the `noise_std` is always
      # 0.5% of the bounding box size.
      noise_std = self.noise_std / self.points_scale
      noise = noise_std * np.random.randn(self.point_sample_num, 3)
      rand_idx = np.random.choice(points.shape[0], size=self.point_sample_num)
      points_noise = points[rand_idx] + noise

      points_in = ocnn.points_new(
          torch.from_numpy(points_noise).float(), torch.Tensor(),
          torch.ones(self.point_sample_num).float(), torch.Tensor())
      points_in, _ = ocnn.clip_points(points_in, [-1.0]*3, [1.0]*3)
      octree_in = self.points2octree(points_in)
    else:
      points_in = points_gt
      octree_in = octree_gt

    # construct the output dict
    return {'octree_in': octree_in, 'points_in': points_in,
            'octree_gt': octree_gt, 'points_gt': points_gt}

  def sample_sdf(self, sample):
    sdf = sample['sdf']
    grad = sample['grad']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(points.shape[0], size=self.sdf_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    sdf = torch.from_numpy(sdf[rand_idx]).float()
    grad = torch.from_numpy(grad[rand_idx]).float()
    return {'pos': points, 'sdf': sdf, 'grad': grad}

  def sample_on_surface(self, points, normals):
    rand_idx = np.random.choice(points.shape[0], size=self.sdf_sample_num)
    xyz = torch.from_numpy(points[rand_idx]).float()
    grad = torch.from_numpy(normals[rand_idx]).float()
    sdf = torch.zeros(self.sdf_sample_num)
    return {'pos': xyz, 'sdf': sdf, 'grad': grad}
  #增加了相似的采样函数
  def sample_view_pos(self, sample):
    view=sample['view']
    return {'view_pos': view}

  def sample_off_surface(self, xyz):
    xyz = xyz / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(xyz.shape[0], size=self.sdf_sample_num)
    xyz = torch.from_numpy(xyz[rand_idx]).float()
    # grad = torch.zeros(self.sample_number, 3)  # dummy grads
    grad = xyz / (xyz.norm(p=2, dim=1, keepdim=True) + 1.0e-6)
    sdf = -1 * torch.ones(self.sdf_sample_num)  # dummy sdfs
    return {'pos': xyz, 'sdf': sdf, 'grad': grad}

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['point_cloud'])
    # sample ground truth sdfs
    if self.flags.load_sdf:
      sdf_samples = self.sample_sdf(sample['sdf'])
      output.update(sdf_samples)

    # sample on surface points and off surface points
    if self.flags.sample_surf_points:
      on_surf = self.sample_on_surface(sample['points'], sample['normals'])
      off_surf = self.sample_off_surface(sample['sdf']['points'])  # TODO
      sdf_samples = {
          'pos': torch.cat([on_surf['pos'], off_surf['pos']], dim=0),
          'grad': torch.cat([on_surf['grad'], off_surf['grad']], dim=0),
          'sdf': torch.cat([on_surf['sdf'], off_surf['sdf']], dim=0)}
      output.update(sdf_samples)
    return output

# 根据filelist读取文件
class ReadFile:
  def __init__(self, load_sdf=False, load_occu=False):
    self.load_occu = load_occu
    self.load_sdf = load_sdf

  def __call__(self, filename):
    #模型点云读取 读取浮雕模型路径下的pc.npz 暂时改为浮雕点云
    file_path=os.path.dirname(filename)
    pc_path=file_path
    #filename_pc = os.path.join(pc_path, 'pc.npz')
    filename_pc = filename+'_pc.npz'
    raw = np.load(filename_pc)
    
    # 数据处理阶段归一化到[-1,1]，实际使用时候是在[-0.5,0.5]，需要修改
    # raw_new=raw['points']*0.5#此处适应归一化而修改
    
    point_cloud = {'points': raw["points"], 'normals': raw['normals']}#此处适应归一化而修改
    output = {'point_cloud': point_cloud}
    if self.load_occu:
      filename_occu = os.path.join(filename, 'points.npz')
      raw = np.load(filename_occu)
      occu = {'points': raw['points'], 'occupancies': raw['occupancies']}
      output['occu'] = occu
    
    if self.load_sdf:
      filename_sdf = filename+'_sdf.npz'
      raw = np.load(filename_sdf)
      sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}
      output['sdf'] = sdf
    return output
  


def get_shapenet_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags.load_sdf, flags.load_occu)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
