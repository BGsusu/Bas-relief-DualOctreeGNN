# data reader for bas relief

import os
import ocnn
import torch
import numpy as np

from solver import Dataset
from .utils import collate_func
from ocnn.octree import Octree, Points

# 处理读取到的数据
class TransformShape:
  
  def __init__(self, flags):
    self.flags = flags
    self.point_sample_num = 3000
    self.sdf_sample_num = 5000
    self.points_scale = 0.5  # the points are in [-0.5, 0.5]
    self.noise_std = 0.005

  def process_points_cloud(self, m_sample, r_sample):
    # points_in,octree_in: origin model
    # points_gt,octree_gt: bas-relief model

    # get the input
    m_points, m_normals = m_sample['points'], m_sample['normals']
    m_points = m_points / self.points_scale  # scale to [-1.0, 1.0]

    # transform points to octree
    points_in = Points(torch.from_numpy(m_points).float(), torch.from_numpy(m_normals).float())
    points_in.clip(-1.0,1.0)
    octree_in = Octree(self.flags.depth, self.flags.full_depth)
    octree_in.build_octree(points_in)

    # get the input
    r_points, r_normals = r_sample['points'], r_sample['normals']
    r_points = r_points / self.points_scale  # scale to [-1.0, 1.0]

    # transform points to octree
    points_gt = Points(torch.from_numpy(r_points).float(), torch.from_numpy(r_normals).float())
    points_gt.clip(-1.0,1.0)
    octree_gt = Octree(self.flags.depth, self.flags.full_depth)
    octree_gt.build_octree(points_gt)

    # construct the output dict
    return {'octree_in': octree_in, 'points_in': points_in,
            'octree_gt': octree_gt, 'points_gt': points_gt}

  def sample_sdf(self, sample):
    # sdf: ba-relief sdf for GT
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

  def sample_off_surface(self, xyz):
    xyz = xyz / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(xyz.shape[0], size=self.sdf_sample_num)
    xyz = torch.from_numpy(xyz[rand_idx]).float()
    # grad = torch.zeros(self.sample_number, 3)  # dummy grads
    grad = xyz / (xyz.norm(p=2, dim=1, keepdim=True) + 1.0e-6)
    sdf = -1 * torch.ones(self.sdf_sample_num)  # dummy sdfs
    return {'pos': xyz, 'sdf': sdf, 'grad': grad}

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['model_point_cloud'], sample['relief_point_cloud'])
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

  def __call__(self, model_filename, relief_filename):
    # input: model pc, GT: relief pc and relief sdf
    output = {}
    # model pc
    model_pc_file = os.path.join(model_filename,'pointcloud.npz')
    raw = np.load(model_pc_file)
    point_cloud = {'points': raw["points"], 'normals': raw['normals']}#此处适应归一化而修改
    output['model_point_cloud'] = point_cloud

    # relief pc
    relief_pc_file = os.path.join(relief_filename,'pointcloud.npz')
    raw = np.load(relief_pc_file)
    point_cloud = {'points': raw["points"], 'normals': raw['normals']}#此处适应归一化而修改
    output['relief_point_cloud'] = point_cloud

    # if self.load_occu:
    #   filename_occu = os.path.join(filename, 'points.npz')
    #   raw = np.load(filename_occu)
    #   occu = {'points': raw['points'], 'occupancies': raw['occupancies']}
    #   output['occu'] = occu
    
    if self.load_sdf:
      sdf_file = os.path.join(relief_filename,'sdf.npz')
      raw = np.load(sdf_file)
      sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}
      output['sdf'] = sdf
    return output
  


def get_bas_relief_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags.load_sdf, flags.load_occu)
  dataset = Dataset(flags.location, flags.model_filelist, flags.relief_filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
