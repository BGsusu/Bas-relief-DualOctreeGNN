import os
import torch
import numpy as np

import builder
import utils
from solver import Solver, get_config
import torch.nn.functional as F

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')

# # CUDA debug
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class OcnnOUNetSolver(Solver):

  def get_model(self, flags):
    return builder.get_model(flags)

  def get_dataset(self, flags):
    return builder.get_dataset(flags)

  def batch_to_cuda(self, batch):
    keys = ['octree_in', 'octree_gt', 'pos', 'sdf', 'grad', 'weight', 'occu']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].cuda()
    batch['pos'].requires_grad_()

  def compute_loss(self, octree, model_out):
    flags = self.FLAGS.LOSS
    loss_func = builder.get_loss_function(flags)
    output = loss_func(octree, model_out)
    return output

  def model_forward(self, batch):
    octree_in = batch['octree_in'].cuda()
    octree_gt = batch['octree_gt'].cuda()
    model_out = self.model(octree_in, octree_gt, False)
    output = self.compute_loss(octree_gt, model_out)
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output

  def extract_mesh(self, neural_mpu, filename, bbox=None):
    # bbox used for marching cubes
    if bbox is not None:
      bbmin, bbmax = bbox[:3], bbox[3:]
    else:
      sdf_scale = self.FLAGS.SOLVER.sdf_scale
      bbmin, bbmax = -sdf_scale, sdf_scale

    # create mesh
    utils.create_mesh(neural_mpu, filename,
                      size=self.FLAGS.SOLVER.resolution,
                      bbmin=bbmin, bbmax=bbmax,
                      mesh_scale=self.FLAGS.DATA.test.point_scale,
                      save_sdf=self.FLAGS.SOLVER.save_sdf)

  # def eval_step(self, batch):
  #   # forward the model
  #   output = self.model.forward(batch['octree_in'].cuda())
    
  #   #self.batch_to_cuda(batch)
  #   #这里修改为未实装的样子
  #   #output = self.model(batch['octree_in'], batch['octree_gt'], batch['pos'],batch['view_pos'])
  #   #output = self.model(batch['octree_in'], batch['octree_gt'], batch['pos'])
    
  #   # extract the mesh
  #   filename = batch['filename'][0]
  #   pos = filename.rfind('.')
  #   if pos != -1: filename = filename[:pos]  # remove the suffix
  #   filename = os.path.join(self.logdir, filename + '.obj')
  #   folder = os.path.dirname(filename)
  #   if not os.path.exists(folder): os.makedirs(folder)
  #   bbox = batch['bbox'][0].numpy() if 'bbox' in batch else None
  #   self.extract_mesh(output['neural_mpu'], filename, bbox)
    
  #   # save the input point cloud
  #   filename = filename[:-4] + '.input.ply'
  #   utils.points2ply(filename, batch['points_in'][0].cpu(),
  #                    self.FLAGS.DATA.test.point_scale)
    
  def eval_step(self, batch):
    # forward the model
    octree_in = batch['octree_in'].cuda(non_blocking=True)
    output = self.model(octree_in, update_octree=True)
    points_out = self.octree2pts(output['octree_out'])

    # save the output point clouds
    # points_in = batch['points']
    filenames = batch['model_filenames']
    for i, filename in enumerate(filenames):
      pos = filename.rfind('o')
      if pos != -1: filename = filename[:pos-1]  # remove the suffix
      filename_in = os.path.join(self.logdir, filename + '.in.xyz')
      filename_out = os.path.join(self.logdir, filename + '.out.xyz')
      os.makedirs(os.path.dirname(filename_in), exist_ok=True)

      # NOTE: it consumes much time to save point clouds to hard disks
      # points_in[i].save(filename_in)
      np.savetxt(filename_out, points_out[i].cpu().numpy(), fmt='%.6f')
      
      # save the input point cloud
      ply_out = os.path.join(self.logdir, filename + '.out.ply')
      # utils.points2ply(ply_out, points_out[i].cpu(),self.FLAGS.DATA.test.point_scale)
      points = points_out[i].cpu()[:, 0:3]
      normal = points_out[i].cpu()[:, 3:6]
      utils.save_points_to_ply(ply_out, points,normal)

  def octree2pts(self, octree):
    depth = octree.depth
    batch_size = octree.batch_size

    signal = octree.features[depth]
    normal = F.normalize(signal[:, :3])
    displacement = signal[:, 3:]

    x, y, z, _ = octree.xyzb(depth, nempty=True)
    xyz = torch.stack([x, y, z], dim=1) + 0.5 + displacement * normal
    xyz = xyz / 2**(depth - 1) - 1.0  # [0, 2^depth] -> [-1, 1]
    point_cloud = torch.cat([xyz, normal], dim=1)

    batch_id = octree.batch_id(depth, nempty=True)
    points_num = [torch.sum(batch_id == i) for i in range(batch_size)]
    points = torch.split(point_cloud, points_num)
    return points

  def save_tensors(self, batch, output):
    iter_num = batch['iter_num']
    filename = os.path.join(self.logdir, '%04d.out.octree' % iter_num)
    output['octree_out'].cpu().numpy().tofile(filename)
    filename = os.path.join(self.logdir, '%04d.in.octree' % iter_num)
    batch['octree_in'].cpu().numpy().tofile(filename)
    filename = os.path.join(self.logdir, '%04d.in.points' % iter_num)
    batch['points_in'][0].cpu().numpy().tofile(filename)
    filename = os.path.join(self.logdir, '%04d.gt.octree' % iter_num)
    batch['octree_gt'].cpu().numpy().tofile(filename)
    filename = os.path.join(self.logdir, '%04d.gt.points' % iter_num)
    batch['points_gt'][0].cpu().numpy().tofile(filename)

  @classmethod
  def update_configs(cls):
    FLAGS = get_config()
    FLAGS.SOLVER.resolution = 512       # the resolution used for marching cubes
    FLAGS.SOLVER.save_sdf = False       # save the sdfs in evaluation
    FLAGS.SOLVER.sdf_scale = 0.9        # the scale of sdfs

    FLAGS.DATA.train.point_scale = 0.5  # the scale of point clouds
    FLAGS.DATA.train.load_sdf = True    # load sdf samples
    FLAGS.DATA.train.load_occu = False  # load occupancy samples
    FLAGS.DATA.train.point_sample_num = 10000
    FLAGS.DATA.train.sample_surf_points = False

    # FLAGS.MODEL.skip_connections = True
    FLAGS.DATA.test = FLAGS.DATA.train.clone()
    FLAGS.LOSS.loss_type = 'sdf_reg_loss'

    FLAGS.DATA.train.points_scale = 128


if __name__ == '__main__':
  OcnnOUNetSolver.main()
