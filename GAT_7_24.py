################图注意力，GAT
import torch
import math
import dgl
import numpy as np
import scipy.sparse as spp
# from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import MaxPooling,AvgPooling
# from torch_geometric.datasets import Planetoid
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.gcn1 = GATConv(1280,128 ,4)
        self.gcn2 = GATConv(128*4, 128, 4)
        self.fc_g1 = torch.nn.Linear(128 * 4, 64)
        # self.ins = nn.InstanceNorm1d(64)
        self.maxpooling = MaxPooling()
        # self.gcn1 = GATConv(128 * 4, 64, 1)
        # self.gcn1 = GATConv(self.embedding_size, 128, 1,dropout=0.5)
        self.elu = nn.ELU()
    def forward(self,G,feature):
        # print(G)
        feature=feature.squeeze(0)
        out=self.gcn1(G,feature)               #G是接触图，feature是1280维
        out=self.elu(out)
        out = out.reshape(-1, 128*4)
        out=self.gcn2(G,out)
        out = self.elu(out)
        out = out.reshape(-1, 128 * 4)
        G.ndata['feat'] = out
        # out = self.maxpooling(G, out)
        out=self.fc_g1(out)
        out=out.unsqueeze(0)
        # out=self.ins(out)
        # out = out.squeeze(0)
        out= self.elu(out)
        return out
def default_loader(cmap_data,l):
# def default_loader(cpath):
    # cmap_data = np.load(cpath)
    # cmap_data = np.loadtxt(cpath)
    # nodenum = len(str(cmap_data['seq']))
    # cmap = cmap_data['contact']
    # g_embed = torch.tensor(embed_data[pid][:nodenum]).float().to(device)
    # l=len(cmap_data)
    # g_embed = torch.rand(l, 12)
    cmap = cmap_data
    for i in range(0, l):  # 自环
        if cmap[i][i] != 1:
            cmap[i][i] = 1
    adj = spp.coo_matrix(cmap)
    u=adj.row
    v=adj.col
    G = dgl.graph((u, v))
    # G = dgl.DGLGraph(adj,device=device)
    # G = G.to(torch.device('cuda'))
    # if len(cmap)==len(g_embed):
    #     g_embed=torch.tensor(g_embed)
    # G.ndata['feat'] = g_embed
    # else:
    #     print('len(cmap)!=len(g_embed)')
    # if nodenum > 1000:
    # textembed = g_embed[:1000]
    # elif nodenum < 1000:
    #     textembed = np.concatenate((embed_data[pid], np.zeros((1000 - nodenum, 1024))))

    # textembed = torch.tensor(textembed).float().to(device)
    return G


# a, b = default_loader('1B63.cmap')
#
# model=Net()
# out = model(a,b)
# out
