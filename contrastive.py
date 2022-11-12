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
"""Contrastive learning module."""

import torch
from util.train import BaseTrain


class Contrastive(BaseTrain):
    """Contrastive learning."""
    def __init__(self, args):
        super(Contrastive, self).__init__(args=args)

    def set_args(self, args):
        # Algorithm-specific parameter
        self.temperature = args.temperature

        # File suffix
        self.file_suffix = 'temp{:g}'.format(self.temperature)

    def set_metrics(self):
        # Metrics
        self.list_of_metrics = [
            'loss.train', 'loss.xe', 'loss.L2', 'acc.train'
        ]
        self.list_of_eval_metrics = [
            'embed.auc',
            # 'embed.kocsvm',
            # 'embed.locsvm',
            # 'embed.kde',
            # 'embed.gde',
            'pool.auc',
            # 'pool.kocsvm',
            # 'pool.locsvm',
            # 'pool.kde',
            # 'pool.gde',
        ]
        # self.metric_of_interest = [
        #     'embed.auc',
        #     'embed.kocsvm',
        #     'embed.locsvm',
        #     'embed.kde',
        #     'embed.gde',
        #     'pool.auc',
        #     'pool.kocsvm',
        #     'pool.locsvm',
        #     'pool.kde',
        #     'pool.gde',
        # ]
        # assert all([
        #     m in self.list_of_eval_metrics for m in self.metric_of_interest
        # ]), 'Some metric does not exist'
        
        self.eval_metrics = {}
        for metric in self.list_of_eval_metrics:
            self.eval_metrics[metric] = None
    
    def train_step(self, data):
        x1, x2, _, _ = data
        batch = x1.size(0)
        # print("x1.size()", x1.size())
        x = torch.cat((x1, x2), dim=0)
        y = torch.arange(0, batch).to(x1.device)
        output = self.model(x)
        embeds = output['embeds']
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
        embeds1, embeds2 = torch.split(embeds, batch)
        # ip = torch.exp(torch.matmul(embeds1, embeds2.T) / self.temperature)
        # loss = - torch.sum(torch.log(ip.diag() / torch.sum(ip, dim=-1)))
        
        ip = torch.matmul(embeds1, embeds2.T) / self.temperature
        # print("ip: ", ip)
        loss = torch.nn.CrossEntropyLoss()(ip, y)
        # loss_l2 = self.get_loss_l2()
        #loss = loss_xe  #+ self.weight_decay * loss_l2
        acc = torch.sum(torch.argmax(ip, axis=-1) == y).item() / len(y)
        return loss, batch, acc

