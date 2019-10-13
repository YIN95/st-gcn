import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

from .st_gcn import Model as ST_GCN
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

class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = Variable(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = Variable(torch.Tensor(out_feats))
        else:
            self.bias = None

    def forward(self, input, adj):
        '''
        input(torch.Tensor):
        adj(torch.Tensor): adjacency matrix
        '''
        hidden = torch.mm(input, weight)
        output = torch.mm(adj, hidden)
        if self.bias:
            return self.bias + output
        else:
            return output


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


class Model(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.join_stream = ST_GCN(*args, **kwargs)
        self.group_stream = ST_GCN(*args, **kwargs)

        self.num_node = 4
        self_link = [(i, i) for i in range(self.num_node)]
        self.edge = [(1, 0), (2, 0), (3, 0), (0, 0), (1, 1), (2, 2), (3, 3)]
        self.center = 0
        self.max_hop = 1
        self.dilation = 1

        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=self.max_hop)

        self.gc1 = GraphConvLayer(16, 8)
        self.gc2 = GraphConvLayer(8, 8)
        self.gc3 = GraphConvLayer(8, 4)
        self.mlp3 = make_mlp([4*4, 2], activation='softmax')
        self.get_adjacency_group()
    
    def get_adjacency_group(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node, self.num_node))
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == hop:
                        if self.hop_dis[j, self.center] == self.hop_dis[
                                i, self.center]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif self.hop_dis[j, self.
                                            center] > self.hop_dis[i, self.
                                                                    center]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        A = np.stack(A)
        self.adj = A

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
        x = F.relu(self.gc1(allGroupData, self.adj))
        x = F.relu(self.gc2(x, self.adj))
        x = self.gc3(x, self.adj)
        x = x.reshape(batch, -1)
        x = self.mlp3(x)
        
        return x