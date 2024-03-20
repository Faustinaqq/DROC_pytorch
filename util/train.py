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
"""Base model trainer."""

import json
import os
import random
import shutil
import time

from model.model import SimModel
from absl import logging
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.svm import OneClassSVM
import tqdm
# import tensorflow as tf
import torch
from tqdm import trange, tqdm
from torchvision.models import resnet18
import torch.nn as nn
import logging
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# from data.celeba import CelebA
from data.cifar import CIFAROOD
# from data.dogvscat import DogVsCatOOD
# from data.fmnist import FashionMNISTOOD
import torch.optim as optim

# import model.model as model
# import util.metric as util_metric
# from util.scheduler import CustomLearningRateSchedule as CustomSchedule

_SUPPORTED_DATASET = frozenset([
    'cifar10ood', 'cifar20ood', 'cifar100ood', 'fashion_mnistood', 'fmnistood',
    'dogvscatood', 'dvcood', 'celeba'
])

# def setup():

#   logging.set_verbosity(logging.ERROR)
#   physical_devices = tf.config.experimental.list_physical_devices('GPU')
#   if not physical_devices:
#     logging.info('No GPUs are detected')
#   for dev in physical_devices:
#     tf.config.experimental.set_memory_growth(dev, True)
#   return tf.distribute.MirroredStrategy()


class BaseTrain(object):
    """Base model trainer.

  Model constructor:
    Parameters
    Data loader
    Model architecture
    Optimizer
  Model trainer:
    Custom train loop
    Evaluation loop
  """
    def __init__(self, args):
        # self.strategy = setup_tf() #GPU使用
        self.args = args
        # data
        self.is_validation = args.is_validation  #是否使用验证集
        self.data_root = args.data_root  #data根目录
        self.dataset = args.dataset  #数据集名称
        self.category = args.category  #数据集类别
        self.aug_list = args.aug_list.split(
            ',')  # eg: [hflip+jitter, hflip+jitter+cutout0.3]
        self.aug_list_for_test = args.aug_list_for_test.split(
            ',') if args.aug_list_for_test is not None else None  #同上类型 / None
        self.input_shape = tuple(args.input_shape)  # (32, 32, 3)
        try:
            self.distaug_type = int(args.distaug_type)  #分布增强的number
        except ValueError:
            self.distaug_type = args.distaug_type
        # network architecture
        self.net_type = args.net_type  # eg： ReNet
        self.in_channels = args.in_channels  # channels
        self.head_dims = tuple(args.head_dims) if args.head_dims not in [
            None, []
        ] else None  # MLP各层神经元数 eg: (512, 512, ..., 128)
        self.latent_dim = args.latent_dim  #FC的hidden layer神经元数，FC是线性网络
        self.num_class = args.num_class

        # optimizer
        self.seed = args.seed  # random seed
        self.force_init = args.force_init  #从头开始训练
        self.optim_type = args.optim_type  # eg: SGD
        self.sched_type = args.sched_type  # learning rate, eg: coss
        # self.sched_freq = args.sched_freq  # shedule的频率，epoch or step
        self.sched_step_size = args.sched_step_size  #
        self.sched_gamma = args.sched_gamma
        self.sched_min_rate = args.sched_min_rate
        # self.sched_level = args.sched_level
        self.sched_milestones = args.sched_milestones
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.regularize_bn = args.regularize_bn
        self.weight_decay_constraint = []
        if self.regularize_bn:
            self.weight_decay_constraint.append('bn')
        self.momentum = args.momentum
        self.nesterov = args.nesterov
        self.num_epoch = args.num_epoch
        self.num_batch = args.num_batch
        self.batch_size = args.batch_size
        # monitoring and checkpoint
        self.ckpt_prefix = os.path.join(args.model_dir, args.ckpt_prefix)
        self.ckpt_epoch = args.ckpt_epoch
        self.file_path = args.file_path
        # additional args
        self.set_args(args)
        self.set_metrics()

    def set_random_seed(self):
        seed = self.seed
        if seed > 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        #   tf.random.set_seed(seed)

    def config(self, device):
        """Config."""
        self.set_random_seed()
        
        self.device = device
        # Data loader.
        # Model architecture.
        self.model = SimModel(net=self.net_type, rep_dim=self.latent_dim, in_channels=self.in_channels, dims=self.head_dims, num_class=self.num_class).to(self.device)
        # Scheduler.
        # print("model: \n", self.model)
        
        self.get_dataloader()
        
        self.optimizer = self.get_optimizer(optim_type=self.optim_type, net=self.model, lr_init=self.learning_rate, weight_decay=self.weight_decay,
                                            **{'momentum': self.momentum, 'nesterov': self.nesterov})
        self.scheduler = self.get_scheduler(sched_type=self.sched_type, optimizer=self.optimizer, 
                                            **{
                                                'step_size': self.sched_step_size,
                                                'gamma': self.sched_gamma,
                                                'min_rate': self.sched_min_rate,
                                                'milestones': self.sched_milestones
                                            })
        self.get_file_path()

    def get_dataloader(self):
        """Gets the data loader."""
        dl = self.get_dataset(self.data_root, self.dataset.lower(), self.category,
                              self.input_shape)  #checked

        self.train_loader, self.cls_loader, self.test_loader, self.train_len = dl.load_dataset(is_validation=self.is_validation,
                                   aug_list=self.aug_list,
                                   aug_list_for_test=self.aug_list_for_test,
                                   batch_size=self.batch_size,
                                   num_batch_per_epoch=self.num_batch,
                                   distaug_type=self.distaug_type)

        # train_loader: train data for representation learning (augmentation)
        # cls_loader: train data for classifier learning (no augmentation)
        # test_loader: test data
        # self.train_loader, self.cls_loader, self.test_loader = data_loaders
        # self.db_name = dl.fname

        # if self.strategy:
        #     self.train_loader = self.strategy.experimental_distribute_dataset(
        #         self.train_loader)
        #     self.cls_loader[0] = self.strategy.experimental_distribute_dataset(
        #         self.cls_loader[0])
        #     self.test_loader[
        #         0] = self.strategy.experimental_distribute_dataset(
        #             self.test_loader[0])

    def get_dataset(self, root, dataset, category, input_shape):
        """Gets the dataset."""
        if dataset not in _SUPPORTED_DATASET:
            msg = (f'Unsupported dataset {dataset} is provided. Only '
                   f'{_SUPPORTED_DATASET} are available.')
            raise ValueError(msg)

        if dataset in ['cifar10ood', 'cifar20ood', 'cifar100ood']:
            dl = CIFAROOD(root=root,
                          dataset=dataset,
                          category=category,
                          input_shape=input_shape, device=self.device)
        elif dataset in ['fashion_mnistood', 'fmnistood']:
            dl = FashionMNISTOOD(root=root,
                                 dataset=dataset,
                                 category=category,
                                 input_shape=input_shape)
        elif dataset in ['dogvscatood', 'dvcood']:
            dl = DogVsCatOOD(root=root,
                             dataset=dataset,
                             category=category,
                             input_shape=input_shape)
        elif dataset == 'celeba':
            dl = CelebA(root=root,
                        dataset=dataset,
                        category=category,
                        input_shape=input_shape)
        return dl

    def get_optimizer(self, optim_type='adam', net=None, lr_init=1e-03, weight_decay=1e-03, **kwargs):
        if optim_type == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr_init, weight_decay=weight_decay, amsgrad=True)
        elif optim_type == 'sgd':
            momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9
            nesterov = kwargs['nesterov'] if 'nesterov' in kwargs else False
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr_init,  weight_decay=weight_decay, **{'momentum': momentum, 'nesterov': nesterov})
        elif optim_type == 'RMSprop':
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr_init, weight_decay=weight_decay)
        return optimizer

    def get_scheduler(self, sched_type='cosine', optimizer=None, **kwargs):
        if sched_type == 'cosine':
            eta_min = kwargs['min_rate'] if 'min_rate' in kwargs else 0
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **{'T_max': (self.train_len // self.batch_size + 1) * self.num_epoch, 'eta_min': eta_min})
        elif sched_type == 'step_lr':
            step_size = kwargs['step_size'] if 'step_size' in kwargs else 1
            gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.995
            scheduler = optim.lr_scheduler.StepLR(optimizer, **{'step_size': step_size, 'gamma': gamma})
        elif sched_type == 'multi_step_lr':
            gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.995
            milestones = kwargs['milestones'] if 'milestones' in kwargs else None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **{'gamma': gamma, 'milestones': milestones})
        else:
            scheduler = None
        return scheduler

    def get_file_path(self):
        """Gets the file path for saving."""
        if self.file_path:
            self.file_path = os.path.join(self.ckpt_prefix, self.file_path)  # ./model_save
        else:
            self.file_path = os.path.join(
                self.ckpt_prefix, '{}_seed{}'.format(self.dataset, self.seed),
                self.model.name, '{}_{}_{}_bs{}'.format(
                    self.__class__.__name__, self.optim_type, self.sched_type,
                    self.batch_size))
            # if self.file_suffix:
            #     self.file_path = '{}_{}'.format(self.file_path,
            #                                     self.file_suffix)
        self.file_path = self.file_path.replace('__', '_')
        # print("self.file_path: ", self.file_path)
        self.json_path = os.path.join(self.file_path, 'stats')

    def get_loss_l2(self):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5 * torch.sum(torch.pow(param, 2)))
        return l2_loss
    
    def train(self):
        self.train_data()
        self.eval_data()
        
    def train_step(self, data):
        x1, x2, _, _ = data
        x = torch.cat((x1, x2), dim=0)
        y = torch.cat((torch.zeros(x1.size(0)), torch.ones(x2.size(0))), dim=0)
        output = self.model(x)
        logits = output['logits']
        loss_xe = nn.CrossEntropyLoss()(logits, y)
        loss_l2 = self.weight_decay * self.get_loss_l2()
        loss = loss_xe + loss_l2
        acc = torch.sum(torch.max(logits, dim=-1).reshape(-1) == y).item() / len(y)
        return loss, loss_xe, loss_l2, x1.size(0), acc
    
    def train_data(self):
        lossAvg = dict()
        # lossAvg['loss_xe'] = AverageMeter()
        # lossAvg['loss_l2'] = AverageMeter()
        lossAvg['loss'] = AverageMeter()
        lossAvg['acc_train'] = AverageMeter()
        # clsLoss = nn.CrossEntropyLoss()
        # c_list = [torch.zeros(o_T), torch.arange(1, up_T+1) / up_T, torch.ones(p_T)]
        # coff = torch.cat(c_list).to(self.device)
        # T = up_T + p_T + o_T
        for i in range(1, self.num_epoch + 1):
            # lossAvg['loss_xe'].reset()
            # lossAvg['loss_l2'].reset()
            lossAvg['loss'].reset()
            lossAvg['acc_train'].reset()
            self.model.train()
            # lr = self.optimizer.param_groups[0]['lr']
            # print("epoch: [{i}/{self.num_epoch}]\t lr = {lr}")
            for data in tqdm(self.train_loader):  
                
                # batch = x1.size(0)

                # x = torch.cat((x1, x2), dim=0)
                # y = torch.cat((torch.zeros(x1.size(0)), torch.ones(x2.size(0))), dim=0)

                # output = self.model(x)
                # logits = output['logits']
                # loss_xe = clsLoss(logits, y)
                # loss_l2 = self.get_loss_l2()
                # loss = loss_xe + self.weight_decay * loss_l2
                loss, batch, acc = self.train_step(data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # acc = torch.sum(torch.max(logits, dim=-1).reshape(-1) == y).item() / len(y)
                
                # lossAvg['loss_xe'].update(loss_xe, batch)
                # lossAvg['loss_l2'].update(loss_l2, 1)
                lossAvg['loss'].update(loss, batch)
                lossAvg['acc_train'].update(acc, batch)
                
            print('epoch: [', i, ']/[', self.num_epoch, '] ',  'loss: ', lossAvg['loss'].average().item(), 'acc: ', lossAvg['acc_train'].average()) #'loss_xe: ',  lossAvg['loss_xe'].average().item(), 'loss_l2: ', lossAvg['loss_l2'].average().item())  # %f"%(lossAvg['loss'].average(), lossAvg['loss.xe'].average(),lossAvg['loss.l2'].average(), lossAvg['acc.train'].average()})
            self.eval_data()
            
               
    def extract(self, data_loader):
        outputs = {'logits': [], 'embeds': [], 'pools': [], 'dscore': [], 'labels': []}
        self.model.eval()
        # if self.aug_list_for_test is not None:
        #     num_aug = len(self.aug_list_for_test)
        # else:
        #     num_aug = 1
        with torch.no_grad():
            for data in tqdm(data_loader):
                x, y = data[:-2], data[-2]
                batch = y.size(0)
                output = self.model(torch.cat(x, dim=0))
                # if num_aug > 1:
                probs = nn.functional.softmax(output['logits'], dim=-1)
                # print("probs.size(): ", probs.size())
                probs = torch.split(probs, batch)
                dscore = torch.exp(torch.sum(torch.log(torch.cat([probs[i][:, i:i+1] for i in range(len(probs))], dim=-1)), dim=-1)) # entropy exp
                outputs['dscore'].append(dscore)
                outputs['logits'].append(nn.functional.softmax(torch.split(output['logits'], batch)[0], dim=-1))
                outputs['embeds'].append(torch.split(output['embeds'], batch)[0])
                outputs['pools'].append(torch.split(output['pools'], batch)[0])
                outputs['labels'].append(y)
            outputs['dscore'] = torch.cat(outputs['dscore'])
            outputs['logits'] = torch.cat(outputs['logits'])
            outputs['embeds'] = torch.cat(outputs['embeds'])
            outputs['pools'] = torch.cat(outputs['pools'])
            outputs['labels'] = torch.cat(outputs['labels'])   
            return outputs
    
    def cal_squared_differece(self, a, b, do_normalization=True):
        if do_normalization:
            a = nn.functional.normalize(a, p=2, dim=-1)
            b = nn.functional.normalize(b, p=2, dim=-1)
            return -2 * torch.matmul(a, b.T)
        else:
            return torch.norm(a, dim=-1, keepdim=True) ** 2 + torch.norm(b, dim=-1, keepdim=True).T ** 2 - 2 * torch.matmul(a, b.T)
            
    def eval_data(self):
        train_outputs = self.extract(self.cls_loader)
        train_embeds, train_pools = train_outputs['embeds'], train_outputs['pools']
        test_outputs = self.extract(self.test_loader)
        test_dscore, test_probs, test_embeds, test_pools, test_labels = test_outputs['dscore'], test_outputs['logits'], test_outputs['embeds'], test_outputs['pools'], test_outputs['labels'].cpu().numpy()
        sim_embed = -0.5 * self.cal_squared_differece(test_embeds, train_embeds, True)
        sim_pool = -0.5 * self.cal_squared_differece(test_pools, train_pools, True)
        dist_embed = 1.0 - torch.max(sim_embed, dim=-1)[0]  #torch.mean(1.0 - torch.topk(sim_embed, k=1)[0], dim=-1)
        dist_pool = 1.0 - torch.max(sim_pool, dim=-1)[0]  #torch.mean(1.0 - torch.topk(sim_pool, k=1)[0], dim=-1)
        for key in self.eval_metrics:
            if key.startswith('logit'):
                pred = 1.0 - test_probs[:, 0]
            elif key.startswith('dscore'):
                pred = 1.0 - test_dscore
            elif key.startswith('embed'):
                pred = dist_embed
                train_feats = train_embeds
                test_feats = test_embeds
                sim = sim_embed
            elif key.startswith('pool'):
                pred = dist_pool
                train_feats = train_pools
                test_feats = test_pools
                sim = sim_pool
                # np.save('./save/{}_train_pool.npy'.format(self.category), train_feats)
                # np.save('./save/{}_test_pool.npy'.format(self.category), test_feats)
                # np.save('./save/{}_test_labels'.format(self.category), test_labels)
            if 'auc' in key:
                self.eval_metrics[key] = roc_auc_score(test_labels, pred.cpu().numpy())
            elif 'locsvm' in key and key.startswith(('embed', 'pool')):
                clf = OneClassSVM(kernel='linear').fit(train_feats.cpu().numpy())
                scores = -clf.score_samples(test_feats.cpu().numpy())
                self.eval_metrics[key] = roc_auc_score(test_labels, scores)
            elif 'kocsvm' in key and key.startswith(('embed', 'pool')):
                train_feats = torch.nn.functional.normalize(train_feats, dim=-1)
                test_feats = torch.nn.functional.normalize(test_feats, dim=-1)
                # gamma = 10. / (torch.var(train_feats).item() * train_feats.size(1))
                clf = OneClassSVM(kernel='rbf', gamma='scale').fit(train_feats.cpu().numpy())
                scores = -clf.score_samples(test_feats.cpu().numpy())
                self.eval_metrics[key] = roc_auc_score(test_labels, scores)
            elif 'kde' in key and key.startswith(('embed', 'pool')):
                # train_feats = torch.nn.functional.normalize(train_feats, dim=-1)
                # test_feats = torch.nn.functional.normalize(test_feats, dim=-1)
                clf = KernelDensity().fit(sim.cpu().numpy())
                scores = clf.score_samples(sim.cpu().numpy())
                self.eval_metrics[key] = roc_auc_score(test_labels, scores)
            elif 'gde' in key and key.startswith(('embed', 'pool')):
                train_feats = torch.nn.functional.normalize(train_feats, dim=-1)
                test_feats = torch.nn.functional.normalize(test_feats, dim=-1)
                gmm = GaussianMixture(n_components=1, init_params='kmeans', covariance_type='full')
                gmm.fit(train_feats.cpu().numpy())
                scores = -gmm.score_samples(test_feats.cpu().numpy())
                self.eval_metrics[key] = roc_auc_score(test_labels, scores)
            
        print(self.eval_metrics)
                     

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        
    def average(self):
        if self.count == 0:
            return 0
        return self.sum / self.count