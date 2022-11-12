# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""CIFAR data."""

import os

import numpy as np
# from six.moves import cPickle
# import tensorflow as tf
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from data.augment import retrieve_augment
from data.data_util import BasicImageProcess

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def rotate_array_deterministic(data):
    """Rotate numpy array into 4 rotation angles.

  Args:
    data: data numpy array, B x C x H x W

  Returns:
    A concatenation of the original and 3 rotations.
  """
    return torch.cat(
        [data] + [torch.rot90(data, k=k, dims=(-2, -1)) for k in range(1, 4)], dim=0)  # (4*B)* C * H * W 


def hflip_array_deterministic(data):
    """Flips numpy array into 2 dierction.

  Args:
    data: data numpy array, B x C x H x W 

  Returns:
    A concatenation of the original data and its horizontally flipped one.
  """
    return torch.cat([data] + [torch.flip(data, dims=(-1,))], dim=0)


def vflip_array_deterministic(data):
    """Flips numpy array into 2 dierction.

  Args:
    data: data numpy array, B x C x H x W

  Returns:
    A concatenation of the original data and its vertically flipped one.
  """
    return torch.cat([data] + [torch.flip(data, dims=(-2,))], dim=0)


def geotrans_array_deterministic(data, num_aug):
    """Transforms numpy array with geometric transformations.

  Original
   + Horizontal flip
  rotation 90
   + Horizontal flip
  rotation 180
   + Horizontal flip
  rotation 270
   + Horizontal flip

  Args:
    data: data numpy array, B x C x H x W
    num_aug: number of distribution augmentations.

  Returns:
    A list of distortions.
  """
    rot_list = [data] + [torch.rot90(data, k=k, dims=(-2, -1)) for k in range(1, 4)]
    rot_flip_list = [[rdata] + [torch.flip(rdata, dims=(-1,))] for rdata in rot_list]

    return_list = []
    for sublist in rot_flip_list:
        for item in sublist:
            return_list.append(item)

    return torch.cat(return_list[:num_aug], dim=0)   # [rot0, rot0 + flip, rot90, rot90 + flip, ...]


class CIFAR(object):
    """CIFAR data loader."""
    def __init__(self, root, dataset='cifar10', input_shape=(3, 32, 32), device='cpu'):
        self.root = root
        label_mode = 'coarse' if dataset in ['cifar20', 'cifar20ood'
                                             ] else 'fine'
        if dataset in ['cifar10', 'cifar10ood']:
            dataset_raw = 'cifar10'
        else:
            dataset_raw = 'cifar100'
        train_dataset, test_dataset = self.load_data(
            dataset=dataset_raw, label_mode=label_mode)  ## numpy数据
        self.trainval_data = [
            # torch.transpose(torch.tensor(train_dataset.data, dtype=torch.float), axes=(0, 3, 1, 2)).to(device),
            torch.tensor(np.transpose(train_dataset.data, axes=(0, 3, 1, 2)), dtype=torch.float).to(device),
            torch.LongTensor(train_dataset.targets).to(device),
            torch.arange(len(train_dataset)).to(device)
            # torch.expand_dims(torch.arange(len(train_dataset)),
            #                axis=1)  # n * 1 index
        ]
        self.test_data = [
            # torch.array(test_dataset.data, dtype=torch.float),
            # torch.array(test_dataset.targets, dtype=torch.int),
            # torch.expand_dims(torch.arange(len(test_dataset)), axis=1)
            torch.tensor(np.transpose(test_dataset.data, axes=(0, 3, 1, 2)), dtype=torch.float).to(device),
            # torch.transpose(torch.tensor(test_dataset.data, dtype=torch.float), axes=(0, 3, 1, 2)).to(device),
            torch.LongTensor(test_dataset.targets).to(device),
            torch.arange(len(test_dataset)).to(device)
        ]
        self.dataset = dataset
        self.input_shape = input_shape
        self.device = device
        
        # print("max: ", self.trainval_data[0].max())

    
    def load_data(self, dataset='cifar10', label_mode='fine'):
        if dataset == 'cifar10':
            if not os.path.exists(self.root):
                os.makedirs(self.root)
            train_dataset = datasets.CIFAR10(root=self.root,
                                             train=True,
                                             download=True)
            test_dataset = datasets.CIFAR10(root=self.root,
                                            train=False,
                                            download=True)
            return train_dataset, test_dataset  ### numpy 数据

    def process_for_ood(self, category=0):
        """Process data for OOD experiment."""
        assert category in torch.unique(
            self.trainval_data[1]), 'category is not in a label set'
        train_neg_idx = torch.where(self.trainval_data[1] == category)[0]  ##neg 是唯一正常的类
        train_pos_idx = torch.where(self.trainval_data[1] != category)[0]  ## pos 其余异常的类
        test_neg_idx = torch.where(self.test_data[1] == category)[0]
        test_pos_idx = torch.where(self.test_data[1] != category)[0]
        self.trainval_data_pos = [  ## 正常的类
            self.trainval_data[0][train_pos_idx],
            self.trainval_data[1][train_pos_idx],
            self.trainval_data[2][train_pos_idx]
        ]
        self.trainval_data[0] = self.trainval_data[0][train_neg_idx]  #正常的类
        self.trainval_data[1] = self.trainval_data[1][train_neg_idx]
        self.trainval_data[2] = self.trainval_data[2][train_neg_idx]
        self.test_data[1][test_neg_idx] = torch.zeros_like(  #正常样本标签为0
            self.test_data[1][test_neg_idx])
        self.test_data[1][test_pos_idx] = torch.ones_like(  #异常样本标签为1
            self.test_data[1][test_pos_idx])

    def get_prefix(self, aug_list):
        """Gets naming prefix."""
        shape_str = 'x'.join(('%d' % s for s in self.input_shape))
        self.fname = f'{self.dataset}_c{self.category}_s{shape_str}'
        self.fname = os.path.join(self.fname, '_'.join(s for s in aug_list))

    def load_dataset(self,
                     is_validation=False,
                     aug_list=None,
                     aug_list_for_test=None,
                     batch_size=64,
                     num_batch_per_epoch=None,
                     distaug_type=''):
        """Load dataset."""

        # Constructs dataset for validation or test.
        # 训练数据只有一类（neg）
        if is_validation:
            np.random.seed(1)
            idx = np.random.permutation(len(self.trainval_data[0]))
            idx_train = idx[:int(len(idx) * 0.9)]
            idx_val_neg = idx[int(len(idx) * 0.9):]
            idx_val_pos = np.random.permutation(len(
                self.trainval_data_pos[0]))[:len(idx_val_neg)]
            train_data = [
                self.trainval_data[0][idx_train],
                self.trainval_data[1][idx_train],
                torch.arange(len(idx_train)).to(self.device)
            ]
            test_data = [
                torch.cat((self.trainval_data[0][idx_val_neg], self.trainval_data_pos[0][idx_val_pos]), dim=0),
                torch.cat((torch.zeros_like(self.trainval_data[1][idx_val_neg]), torch.ones_like(self.trainval_data_pos[1][idx_val_pos])), dim=0),
                torch.arange(len(idx_val_pos) + len(idx_val_neg)).to(self.device)
            ]
        else:
            train_data = self.trainval_data
            test_data = self.test_data

        # Sets aside unaugmented training data to construct a classifier.
        # We limit the number by 20000 for efficiency of learning classifier.
        indices = np.random.permutation(len(train_data[0]))[:20000] if len(
            train_data[0]) > 20000 else np.arange(len(train_data[0]))
        train_data_for_cls = [data[indices]
                              for data in train_data]  # shuffle data

        if distaug_type:
            # Applies offline distribution augmentation on train data.
            # Type of augmentation: Rotation (0, 90, 180, 270), horizontal or
            # vertical flip, combination of rotation and horizontal flips.
            assert distaug_type in ['rot', 'hflip', 'vflip'] + [
                1, 2, 3, 4, 5, 6, 7, 8
            ], f'{distaug_type} is not supported distribution augmentation type.'
            if distaug_type == 'rot':
                aug_data = rotate_array_deterministic(train_data[0])
                lab_data = torch.cat([train_data[1] for _ in range(4)], dim=0)  # (4 * B)
            elif distaug_type == 'hflip':
                aug_data = hflip_array_deterministic(train_data[0])
                lab_data = torch.cat([train_data[1] for _ in range(2)], dim=0)
            elif distaug_type == 'vflip':
                aug_data = vflip_array_deterministic(train_data[0])
                lab_data = torch.cat([train_data[1] for _ in range(2)], dim=0)
            elif distaug_type in [1, 2, 3, 4, 5, 6, 7, 8]:
                aug_data = geotrans_array_deterministic(train_data[0], distaug_type)
                lab_data = torch.cat([train_data[1] for _ in range(distaug_type)], dim=0)
            train_data = [
                aug_data,
                lab_data,
                torch.arange(len(aug_data)).to(self.device)  # (B*aug_num) * 1
            ]
            #在这里都是numpy数据
        train_len = len(train_data)
        train_set = BasicImageProcess(data=tuple(train_data),
                                      input_shape=self.input_shape)
        train_set_for_cls = BasicImageProcess(data=tuple(train_data_for_cls),
                                              input_shape=self.input_shape)
        test_set = BasicImageProcess(data=tuple(test_data),
                                     input_shape=self.input_shape)

        aug_args = {'size': self.input_shape[1]}
        # print(aug_list)
        augs = retrieve_augment(aug_list, **aug_args)  # augs: [[fun1, fun2,...], [...]]
        # print("augs:", augs)
        train_aug, test_aug = augs[:-1], augs[-1]
        # print("len(train_aug): ", len(train_aug))
        # print("augs: ", augs)
        if aug_list_for_test is not None:
            test_aug = retrieve_augment(aug_list_for_test, **aug_args)
            test_aug = test_aug[:-1]

        # [dataset, num_batch_per_epoch], dataset(batch * (images, label[1], image_id)), 有些可能会重复
        train_loader = train_set.input_fn(
            is_training=True,
            batch_size=batch_size,
            aug_list=train_aug,
            num_batch_per_epoch=num_batch_per_epoch,
            training_dataset_cache=True)
        train_loader_for_cls = train_set_for_cls.input_fn(
            is_training=False,
            batch_size=batch_size if aug_list_for_test is None else batch_size // len(aug_list_for_test),
            aug_list=test_aug,
            force_augment=aug_list_for_test is not None,
            training_dataset_cache=True)
        # print("test_loader")
        test_loader = test_set.input_fn(
            is_training=False,
            batch_size=batch_size if aug_list_for_test is None else batch_size // len(aug_list_for_test),
            aug_list=test_aug,
            force_augment=aug_list_for_test is not None,
            training_dataset_cache=True)

        self.get_prefix(aug_list)
        return train_loader, train_loader_for_cls, test_loader, train_len


class CIFAROOD(CIFAR):
    """CIFAR for OOD."""
    def __init__(self,
                 root,
                 dataset='cifar10',
                 input_shape=(3, 32, 32),
                 category=0, device="cpu"):
        super(CIFAROOD, self).__init__(root=root,
                                       dataset=dataset,
                                       input_shape=input_shape, device=device)
        if isinstance(category, str):
            try:
                category = int(float(category))
            except ValueError:
                msg = f'category {category} must be integer convertible.'
                raise ValueError(msg)
        self.category = category
        self.process_for_ood(category=self.category)
        
        
