import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.data as data_utils
import dgl
import networkx as nx
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN
from .group_gcn import Model as GCN
from net.utils.gat import GAT
import numpy as np
import scipy.spatial.distance as distance 


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
        # print(jointInput[:, :, -1:, 5])
        # print(jointInput[:, :, -1:, 5].shape)
        joinChest = jointInput[:, :, -1:, 5].data.cpu().numpy().reshape(batch, 3)

        jointInput = jointInput.reshape((N, C, T, V, 1))
        groupInputs = x[:, :, :, :, 1:]

        groupChest1 = groupInputs[:, :, -1:, 5, 0].data.cpu().numpy().reshape(batch, 3)
        groupChest2 = groupInputs[:, :, -1:, 5, 1].data.cpu().numpy().reshape(batch, 3)
        groupChest3 = groupInputs[:, :, -1:, 5, 2].data.cpu().numpy().reshape(batch, 3)

        dis1 = np.array([joinChest, groupChest1])
        dis2 = np.array([joinChest, groupChest2])
        dis3 = np.array([joinChest, groupChest3])
        # print(dis1.shape)
        dist1 = []
        dist2 = []
        dist3 = []
        for b in range(batch):
            dist1.append(1.0/distance.pdist(dis1[:, b, :]))
            dist2.append(1.0/distance.pdist(dis2[:, b, :]))
            dist3.append(1.0/distance.pdist(dis3[:, b, :]))

        # print(dist1)
        # dist1 = distance.pdist(points)
        dist = [dist1, dist2, dist3]
        # print(dist)
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
        device = torch.device(allGroupData.device)
        dist1 = torch.tensor(dist1, dtype=torch.float32).to(device=device)
        dist2 = torch.tensor(dist2, dtype=torch.float32).to(device=device)
        dist3 = torch.tensor(dist3, dtype=torch.float32).to(device=device)

        # dist1 = dist1
        # dist2 = dist2
        # dist3 = dist3
        for i in range(batch):
            allGroupData[i, 0, :] = allGroupData[i, 0, :]*dist1[i].data
            allGroupData[i, 1, :] = allGroupData[i, 1, :]*dist2[i].data
            allGroupData[i, 2, :] = allGroupData[i, 2, :]*dist3[i].data

        allGroupData = allGroupData.permute(0, 2, 1).contiguous()
        # print(allGroupData.shape)
        # device = torch.device(allGroupData.device)

        allGroupData = allGroupData.reshape((N, 16, 1, 4, 1))
        out = self.gcn(allGroupData, dist)
        # print(out)

        # print(out.shapeFloatTensor
        # self.adj = self.adj.to(device=device)
        # self.out = self.gro
        # out = np.array(out)
        # print(out)

        return out
