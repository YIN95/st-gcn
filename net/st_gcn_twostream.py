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
        self.join_stream = ST_GCN(*args, **kwargs)
        self.group_stream = ST_GCN(*args, **kwargs)
        self.gcn = GCN(*args, **kwargs)

    def forward(self, x):
        batch = x.shape[0]
        groupSize = 3
        device = torch.device(x.device)

        jointInput = x[:, :, :, :, 0]
        joinChest = jointInput[:, :, -1:, 5].data.cpu().numpy().reshape(batch, 3)

        N, C, T, V = jointInput.shape
        jointInput = jointInput.reshape((N, C, T, V, 1))
        groupInputs = x[:, :, :, :, 1:]

        joinBodyOutput = self.join_stream(jointInput)

        # print("jointChest")
        # print(joinChest.shape)

        groupChest1 = groupInputs[:, :, -1:, 5, 0].data.cpu().numpy().reshape(batch, 3)
        groupChest2 = groupInputs[:, :, -1:, 5, 1].data.cpu().numpy().reshape(batch, 3)
        groupChest3 = groupInputs[:, :, -1:, 5, 2].data.cpu().numpy().reshape(batch, 3)

        joinHead1 = jointInput[:, [0, 2], -1:, 8].data.cpu().numpy().reshape(batch, 2)
        joinHead2 = jointInput[:, [0, 2], -1:, 9].data.cpu().numpy().reshape(batch, 2)

        groupHead1_1 = groupInputs[:, [0, 2], -1:, 8, 0].data.cpu().numpy().reshape(batch, 2)
        groupHead1_2 = groupInputs[:, [0, 2], -1:, 9, 0].data.cpu().numpy().reshape(batch, 2)

        groupHead2_1 = groupInputs[:, [0, 2], -1:, 8, 1].data.cpu().numpy().reshape(batch, 2)
        groupHead2_2 = groupInputs[:, [0, 2], -1:, 9, 1].data.cpu().numpy().reshape(batch, 2)

        groupHead3_1 = groupInputs[:, [0, 2], -1:, 8, 2].data.cpu().numpy().reshape(batch, 2)
        groupHead3_2 = groupInputs[:, [0, 2], -1:, 9, 2].data.cpu().numpy().reshape(batch, 2)

        dis1 = np.array([joinChest, groupChest1])
        dis2 = np.array([joinChest, groupChest2])
        dis3 = np.array([joinChest, groupChest3])

        join_slope = (joinHead2[:, 1] - joinHead1[:, 1]) / (joinHead2[:, 0] - joinHead1[:, 0])

        join_slope = join_slope.reshape(batch, 1)

        group1_slope = (groupHead1_1[:, 1] - groupHead1_2[:, 1]) / (groupHead1_1[:, 0] - groupHead1_2[:, 0])
        group1_slope = group1_slope.reshape(batch, 1)

        group2_slope = (groupHead2_1[:, 1] - groupHead2_2[:, 1]) / (groupHead2_1[:, 0] - groupHead2_2[:, 0])
        group2_slope = group2_slope.reshape(batch, 1)

        group3_slope = (groupHead3_1[:, 1] - groupHead3_2[:, 1]) / (groupHead3_1[:, 0] - groupHead3_2[:, 0])
        group3_slope = group3_slope.reshape(batch, 1)

        # ang1 = (group1_slope - join_slope) / (1 + join_slope*group1_slope)
        # ang2 = (group2_slope - join_slope) / (1 + join_slope*group2_slope)
        # ang3 = (group3_slope - join_slope) / (1 + join_slope*group2_slope)
        ang1 = torch.tensor(group1_slope, dtype=torch.float32).to(device=device)
        ang1 = torch.sigmoid(ang1)
        ang2 = torch.tensor(group2_slope, dtype=torch.float32).to(device=device)
        ang2 = torch.sigmoid(ang2)
        ang3 = torch.tensor(group3_slope, dtype=torch.float32).to(device=device)
        ang3 = torch.sigmoid(ang3)

        ang0 = torch.tensor(join_slope, dtype=torch.float32).to(device=device)
        ang0 = torch.sigmoid(ang0)
        ang = [ang1, ang2, ang3]
        # print(ang)
        dist1, dist2, dist3 = [], [], []
        for b in range(batch):
            dist1.append(distance.pdist(dis1[:, b, :]))
            dist2.append(distance.pdist(dis2[:, b, :]))
            dist3.append(distance.pdist(dis3[:, b, :]))

        dist1 = torch.tensor(dist1, dtype=torch.float32).to(device=device)
        dist2 = torch.tensor(dist2, dtype=torch.float32).to(device=device)
        dist3 = torch.tensor(dist3, dtype=torch.float32).to(device=device)
        dist0 = (dist1 + dist2 + dist3) / 3.0

        joinBodyOutput = torch.cat([joinBodyOutput, dist0], dim=1)
        joinBodyOutput = torch.cat([joinBodyOutput, ang0], dim=1)

        allBodyOutputs = []
        allBodyOutputs.append(joinBodyOutput)

        # print("dist1")
        # print(dist1.shape)
        dist = [dist1, dist2, dist3]

        for i in range(groupSize):
            groupInput = groupInputs[:, :, :, :, i]
            N, C, T, V = groupInput.shape
            groupInput = groupInput.reshape((N, C, T, V, 1))
            groupBodyOutput = self.group_stream(groupInput)
            # print("allbodyoutputs")
            # print(groupBodyOutput.shape)
            groupBodyOutput = torch.cat([groupBodyOutput, dist[i]], dim=1)
            groupBodyOutput = torch.cat([groupBodyOutput, ang[i]], dim=1)
            allBodyOutputs.append(groupBodyOutput)

        allGroupData = torch.stack(allBodyOutputs, dim=2)
        allGroupData = allGroupData.permute(0, 2, 1).contiguous()

        allGroupData = allGroupData.reshape((N, 18, 1, 4, 1))
        out = self.gcn(allGroupData)

        return out
