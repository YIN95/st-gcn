import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN
from .group_gcn import Model as GCN
from net.utils.gat import GAT
import numpy as np


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        # print(**kwargs)
        # print("===")
        self.join_stream = ST_GCN(*args, **kwargs)
        self.group_stream = ST_GCN(*args, **kwargs)
        self.gcn = GCN(*args, **kwargs)
        # self.gcn = GCN(16, 2, self.graph_args, True )
        # self.gat = GAT(16, 8, nclass=2, dropout=0.5, nheads=2, alpha=0.2)
        # self.adj = np.array([[1, 1, 1, 1],
        #                      [1, 1, 0, 0],
        #                      [1, 0, 1, 0],
        #                      [1, 0, 0, 1]])
        # self.adj = Variable(torch.tensor(self.adj))

        # self.mlp3 = make_mlp([4*4, 2], activation='softmax')

    def forward(self, x):
        batch = x.shape[0]
        groupSize = 3

        jointInput = x[:, :, :, :, 0]
        N, C, T, V = jointInput.shape
        jointInput = jointInput.reshape((N, C, T, V, 1))
        groupInputs = x[:, :, :, :, 1:]

        joinBodyOutput = self.join_stream(jointInput)
        allBodyOutputs = []
        allBodyOutputs.append(joinBodyOutput)

        for i in range(groupSize):
            groupInput = groupInputs[:, :, :, :, i]
            N, C, T, V = groupInput.shape
            groupInput = groupInput.reshape((N, C, T, V, 1))
            groupBodyOutput = self.group_stream(groupInput)
            allBodyOutputs.append(groupBodyOutput)

        allGroupData = torch.stack(allBodyOutputs, dim=2)
        allGroupData = allGroupData.permute(0, 2, 1).contiguous()
        # print(allGroupData.shape)
        # device = torch.device(allGroupData.device)

        allGroupData = allGroupData.reshape((N, 16, 1, 4, 1))
        out = self.gcn(allGroupData)
        # print(out)

        # print(out.shape)
        # self.adj = self.adj.to(device=device)
        # self.out = self.gro
        # out = np.array(out)
        # print(out)

        return out
