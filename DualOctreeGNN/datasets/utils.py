# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import numpy as np

def collate_func(batch):
  output = ocnn.collate_octrees(batch)

  if 'pos' in output:
    batch_idx = torch.cat([torch.ones(pos.size(0), 1) * i
                           for i, pos in enumerate(output['pos'])], dim=0)
    pos = torch.cat(output['pos'], dim=0)
    output['pos'] = torch.cat([pos, batch_idx], dim=1)
    
  for key in ['grad', 'sdf', 'occu', 'weight']:
    if key in output:
      output[key] = torch.cat(output[key], dim=0)
  
  if 'view_pos' in output:#这里按照八叉树解码器的维度，每个数据512维，给他们复制512份，512*3，然后拼接
      tmp=np.array(output['view_pos'],dtype=np.float32)
      tmp=torch.Tensor(tmp)
      output['view_pos']=tmp
      #output['view_pos']=torch.cat(tmp, dim=0)
  return output
