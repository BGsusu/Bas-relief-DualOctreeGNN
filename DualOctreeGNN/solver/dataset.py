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

  def __init__(self, root, model_filelist, relief_filelist, transform, read_file=read_file,
               in_memory=False, take: int = -1):
    super(Dataset, self).__init__()
    self.root = root
    self.model_filelist = model_filelist
    self.relief_filelist = relief_filelist
    self.transform = transform
    self.in_memory = in_memory
    self.read_file = read_file
    self.take = take

    self.model_filenames, self.relief_filenames, self.labels = self.load_filenames()
    if self.in_memory:
      print('Load files into memory from ' + self.filelist)
      self.samples = [self.read_file(os.path.join(self.root, m), os.path.join(self.root, r))
                      for m in tqdm(self.model_filenames, ncols=80, leave=False)
                      for r in enumerate(self.relief_filenames)]

  def __len__(self):
    return len(self.model_filenames)

  def __getitem__(self, idx):
    sample = self.samples[idx] if self.in_memory else \
             self.read_file(os.path.join(self.root, self.model_filenames[idx]), 
                            os.path.join(self.root, self.relief_filenames[idx]))  # noqa
    output = self.transform(sample, idx)    # data augmentation + build octree
    output['label'] = self.labels[idx]
    output['model_filenames'] = self.model_filenames[idx]
    output['relief_filenames'] = self.relief_filenames[idx]
    #一对一
    return output

  # 从给定的文件夹列表中，查找符合要求鹅文件，并形成文件列表
  # 这个函数不负责文件读取，只是文件搜集和存在性验证的过程
  def load_filenames(self):
      model_filenames, relief_filenames, labels = [], [], []
      # model filelist
      with open(self.model_filelist) as fid:
          lines = fid.readlines()
          for line in lines:
            filename = line.replace('\n', '')
            model_filenames.append(filename)
            label = 0
            labels.append(int(label))
      # bas-relief filelist
      with open(self.relief_filelist) as fid:
          lines = fid.readlines()
          for line in lines:
            filename = line.replace('\n', '')
            relief_filenames.append(filename)
            label = 0
            labels.append(int(label))
                    
      num = len(model_filenames)
      if self.take > num or self.take < 1:
          self.take = num
      return model_filenames[:self.take], relief_filenames[:self.take], labels[:self.take]