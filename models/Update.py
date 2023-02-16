#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from typing import OrderedDict
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

from opacus import PrivacyEngine


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, mal=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.mal = mal
        # print(len(DatasetSplit(dataset, idxs)), self.args.local_bs, len(self.ldr_train))

    def train(self, net):
        if self.mal < 1:
            w = OrderedDict()
            for k, v in net.state_dict().items():
                for i in range(len(v)):
                    v[i] = 0.05
                w[k] = v
            return w, 0
            
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        privacy_engine = PrivacyEngine()
        net, optimizer, self.ldr_train = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=self.ldr_train,
            noise_multiplier=0.1,
            max_grad_norm=1.0,
        )
        range_bound = 0.1

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(batch_idx,len(images), len(labels))
                if len(labels) == 0:
                    continue
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # net.zero_grad()
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                try:
                    loss.backward()
                except Exception as e:
                    # print(len(labels))
                    # print(batch_idx)
                    print(e)
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # opacus changes layer name, so rename it back
        w = OrderedDict()
        for k, v in net.state_dict().items():
            v = torch.clip(v, -range_bound, range_bound)
            if k[:8] == "_module.":
                w[k[8:]] = v
            else:
                w[k] = v
        # print(w)
        return w, sum(epoch_loss) / len(epoch_loss)

