# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm


def read_file(filename):
  points = np.fromfile(filename, dtype=np.uint8)
  return torch.from_numpy(points)   # convert it to torch.tensor


class Dataset(torch.utils.data.Dataset):

  def __init__(self, root, filelist, transform, read_file=read_file,
               in_memory=False, take: int = -1):
    super(Dataset, self).__init__()
    self.root = root
    self.filelist = filelist
    self.transform = transform
    self.in_memory = in_memory
    self.read_file = read_file
    self.take = take

    self.filenames, self.labels = self.load_filenames()
    if self.in_memory:
      print('Load files into memory from ' + self.filelist)
      self.samples = [self.read_file(os.path.join(self.root, f))
                      for f in tqdm(self.filenames, ncols=80, leave=False)]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    path=os.path.join(self.root, self.filenames[idx])
    sample = self.samples[idx] if self.in_memory else \
             self.read_file(os.path.join(self.root, self.filenames[idx]))  # noqa
    output = self.transform(sample, idx)    # data augmentation + build octree
    output['label'] = self.labels[idx]
    output['filename'] = self.filenames[idx]
    #一对一
    return output

  # 从给定的文件夹列表中，查找符合要求鹅文件，并形成文件列表
  # 这个函数不负责文件读取，只是文件搜集和存在性验证的过程
  def load_filenames(self):
      filenames, labels = [], []
      with open(self.filelist) as fid:
          lines = fid.readlines()
          for line in lines:
            filename = line.replace('\n', '')
            label = 0
            #filenames.append(filename+'/pc')
            #labels.append(int(label))
            for i in range(0,3):#3
                for j in range(0,5):#5
                    '''
                    filename_pts='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/'+filename+'/BaseRelief{}_{}_pc.npz'.format(i,j)  
                    filename_out='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/'+filename+'/BaseRelief{}_{}_sdf.npz'.format(i,j)
                    if os.path.exists(filename_pts)==False:continue
                    if os.path.exists(filename_out)==False:continue#from 0_0-2_4
                    '''
                    filename_pts='/home/daipinxuan/bas_relief/AllData/BasRelief/pc_sdf/'+filename+'/BaseRelief{}_{}_pc.npz'.format(i,j)
                    if os.path.exists(filename_pts)==False:continue
                    filenames.append(filename+'/BaseRelief{}_{}'.format(i,j) )
                    labels.append(int(label))

      num = len(filenames)
      if self.take > num or self.take < 1:
          self.take = num
      return filenames[:self.take], labels[:self.take]