import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN


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

        self.mlp1 = make_mlp([4*16, 4], activation='tanh')
        self.mlp2 = make_mlp([4, 4], activation='softmax')
        self.mlp3 = make_mlp([4*16, 2], activation='softmax')

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
        groupAttention = self.mlp1(allGroupData.reshape(batch, -1))
        groupAttention = self.mlp2(groupAttention)
        groupAttentionOutput = torch.mul(allGroupData, groupAttention.unsqueeze(dim=1))
        groupAttentionOutput = groupAttentionOutput.reshape(batch, -1)
        groupAttentionOutput = self.mlp3(groupAttentionOutput) 
        return groupAttentionOutput
        
        # N, C, T, V, M = x.size()
        # m = torch.cat((torch.cuda.FloatTensor(N, C, 1, V, M).zero_(),
        #                x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
        #                torch.cuda.FloatTensor(N, C, 1, V, M).zero_()), 2)

        # res = self.origin_stream(x) + self.motion_stream(m)