# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import torch.nn

from . import mpu
from . import modules
from . import dual_octree


class GraphOUNet(torch.nn.Module):

  def __init__(self, depth, channel_in, nout, full_depth=2, depth_out=6,
               resblk_type='bottleneck', bottleneck=4):
    super().__init__()
    self.depth = depth
    self.channel_in = channel_in
    self.nout = nout
    self.full_depth = full_depth
    self.depth_out = depth_out
    self.resblk_type = resblk_type
    self.bottleneck = bottleneck
    self.neural_mpu = mpu.NeuralMPU(self.full_depth, self.depth_out)
    self._setup_channels_and_resblks()
    n_edge_type, avg_degree = 7, 7

    # encoder
    self.conv1 = modules.GraphConvBnRelu(
        channel_in, self.channels[depth], n_edge_type, avg_degree, depth-1)
    self.encoder = torch.nn.ModuleList(
        [modules.GraphResBlocks(self.channels[d], self.channels[d],
         self.resblk_num[d], bottleneck, n_edge_type, avg_degree, d-1, resblk_type)
         for d in range(depth, full_depth-1, -1)])
    self.downsample = torch.nn.ModuleList(
        [modules.GraphDownsample(self.channels[d], self.channels[d-1])
         for d in range(depth, full_depth, -1)])

    # decoder
    self.upsample = torch.nn.ModuleList(
        [modules.GraphUpsample(self.channels[d-1], self.channels[d])
         for d in range(full_depth+1, depth + 1)])
    self.decoder = torch.nn.ModuleList(
        [modules.GraphResBlocks(self.channels[d], self.channels[d],
         self.resblk_num[d], bottleneck, n_edge_type, avg_degree, d-1, resblk_type)
         for d in range(full_depth+1, depth + 1)])

    # header
    #此处修改
    #self.view_pos=modules.Predict_ViewPos()
    self.predict = torch.nn.ModuleList(
        [self._make_predict_module(self.channels[d], 2)
         for d in range(full_depth, depth + 1)])
    self.regress = torch.nn.ModuleList(
        [self._make_predict_module(self.channels[d], 4)
         for d in range(full_depth, depth + 1)])

  def _setup_channels_and_resblks(self):
    # self.resblk_num = [3] * 7 + [1] + [1] * 9
    self.resblk_num = [3] * 16
    self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24]
  
  def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
    return torch.nn.Sequential(
        modules.Conv1x1BnRelu(channel_in, num_hidden),
        modules.Conv1x1(num_hidden, channel_out, use_bias=True))
  
  '''def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
      return modules.Predict_ViewPos(channel_in, channel_out,num_hidden)'''
  def _get_input_feature(self, doctree):
    return doctree.get_input_feature()
  #此处定义了一系列图卷积运算，作为一个列表存储    
  def octree_encoder(self, octree, doctree):
    depth, full_depth = self.depth, self.full_depth
    data = self._get_input_feature(doctree)

    convs = dict()
    convs[depth] = data
    for i, d in enumerate(range(depth, full_depth-1, -1)):
      # perform graph conv
      convd = convs[d]  # get convd
      edge_idx = doctree.graph[d]['edge_idx']
      edge_type = doctree.graph[d]['edge_dir']
      node_type = doctree.graph[d]['node_type']
      if d == self.depth:  # the first conv
        convd = self.conv1(convd, edge_idx, edge_type, node_type)
      convd = self.encoder[i](convd, edge_idx, edge_type, node_type)
      #此处我增加了和视角分量的交互方式
      
      convs[d] = convd  # update convd

      # downsampleing
      if d > full_depth:  # init convd
        nnum = doctree.nnum[d]
        lnum = doctree.lnum[d-1]
        leaf_mask = doctree.node_child(d-1) < 0
        convs[d-1] = self.downsample[i](convd, leaf_mask, nnum, lnum)

    return convs

  # doctree_out:GT; doctree:encoder result
  def octree_decoder(self, convs, doctree_out, doctree, view_pos=None,update_octree=False):
    logits = dict()
    reg_voxs = dict()
    deconvs = dict()

    deconvs[self.full_depth] = convs[self.full_depth]
    for i, d in enumerate(range(self.full_depth, self.depth_out+1)):
        first_layer=d
        break
    batch_size=int(deconvs[first_layer].size()[0]/512)#1-16的整数
    for i, d in enumerate(range(self.full_depth, self.depth_out+1)):
      if d > self.full_depth:
        nnum = doctree_out.nnum[d-1]
        leaf_mask = doctree_out.node_child(d-1) < 0
        deconvd = self.upsample[i-1](deconvs[d-1], leaf_mask, nnum)
        skip = modules.doctree_align(
            convs[d], doctree.graph[d]['keyd'], doctree_out.graph[d]['keyd'])
        deconvd = deconvd + skip  # skip connections

        edge_idx = doctree_out.graph[d]['edge_idx']
        edge_type = doctree_out.graph[d]['edge_dir']
        node_type = doctree_out.graph[d]['node_type']
        deconvs[d] = self.decoder[i-1](deconvd, edge_idx, edge_type, node_type)

      # predict the splitting label  这里有修改
      #logit = self.predict[i](deconvs[d],view_pos)
      #广播次数 是修改后的 注意到还不能按照这个算，有些层是非2的倍数，差一些 没有办法的办法：batch_size=1
      duplicate_num=deconvs[d].shape[0]
      #duplicate_num=int(deconvs[d].shape[0]/batch_size)
      logit = self.predict[i](deconvs[d],view_pos,duplicate_num)
      nnum = doctree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # update the octree according to predicted labels 这里也有修改
      if update_octree:
        label = logits[d].argmax(1).to(torch.int32)
        octree_out = doctree_out.octree
        octree_out = ocnn.octree_update(octree_out, label, d, split=1)
        if d < self.depth_out:
          octree_out = ocnn.octree_grow(octree_out, target_depth=d+1)
        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

      # predict the signal
      #reg_vox = self.regress[i](deconvs[d],view_pos)
      reg_vox = self.regress[i](deconvs[d],view_pos,duplicate_num)
      # TODO: improve it
      # pad zeros to reg_vox to reuse the original code for ocnn
      node_mask = doctree_out.graph[d]['node_mask']
      shape = (node_mask.shape[0], reg_vox.shape[1])
      reg_vox_pad = torch.zeros(shape, device=reg_vox.device)
      reg_vox_pad[node_mask] = reg_vox
      reg_voxs[d] = reg_vox_pad

    return logits, reg_voxs, doctree_out.octree

  def forward(self, octree_in, octree_out=None, pos=None,view_pos=None):
    # generate dual octrees
    doctree_in = dual_octree.DualOctree(octree_in)
    doctree_in.post_processing_for_docnn()

    update_octree = octree_out is None
    if update_octree:
      octree_out = ocnn.create_full_octree(self.full_depth, self.nout)
    doctree_out = dual_octree.DualOctree(octree_out)
    doctree_out.post_processing_for_docnn()

    # run encoder and decoder
    convs = self.octree_encoder(octree_in, doctree_in)
    #此处增加参数view_pos
    #原代码 output没有logits这一键值对
    out = self.octree_decoder(convs, doctree_out, doctree_in,view_pos,update_octree)
    output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos, out[1], out[2])

    # create the mpu wrapper
    def _neural_mpu(pos):
      pred = self.neural_mpu(pos, out[1], out[2])
      return pred[self.depth_out][0]
    output['neural_mpu'] = _neural_mpu

    return output
