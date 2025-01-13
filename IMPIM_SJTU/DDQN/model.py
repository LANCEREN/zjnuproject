import os
import torch
import numpy as np
import random
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_add, scatter_softmax, scatter_max
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
# import utils.graph_utils as graph_utils
from collections import deque
from tqdm import tqdm
# from federatedscope.gfl.model import SAGE_Net
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import SAGEConv


class GNN_DDQN(nn.Module):
    ''' to check and verify with the design in the paper '''
    def __init__(self, reg_hidden, embed_dim, node_dim, edge_dim, T, w_scale, avg):
        '''w_scale=0.01, node_dim=2, edge_dim=4'''
        super(GNN_DDQN, self).__init__()
        # depth of structure2vector
        # 在初始化中，定义了一些参数和权重。这些参数和权重包括节点和边的嵌入维度、图嵌入的迭代次数、隐藏层的大小等。这些权重参数被初始化为正态分布的随机值。
        # 嵌入维度
        self.T = T
        # 默认64
        self.embed_dim = embed_dim 
        # 隐藏层大小，先用默认32
        self.reg_hidden = reg_hidden
        # 默认加而不是平均
        self.avg = avg
        # 参数初始化
        # 这些都是默认
        # input node to latent
        # node_dim+1是因为最后一维多了状态
        self.w_n2l = torch.nn.Parameter(torch.Tensor(node_dim+1, embed_dim))
        torch.nn.init.normal_(self.w_n2l, mean=0, std=w_scale)

        # input edge to latent
        self.w_e2l = torch.nn.Parameter(torch.Tensor(edge_dim, embed_dim))
        torch.nn.init.normal_(self.w_e2l, mean=0, std=w_scale)

        # linear node conv
        self.p_node_conv = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.p_node_conv, mean=0, std=w_scale)

        # trans node 1
        self.trans_node_1 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_1, mean=0, std=w_scale)

        # trans node 2
        self.trans_node_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_2, mean=0, std=w_scale)
        # 如果reg_hidden大于0，还会初始化两个额外的权重矩阵（h1_weight和h2_weight）用于正则化。所有这些权重都通过正态分布进行初始化。
        # 这些就是强化学习的参数dqn部分
        if self.reg_hidden > 0:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, reg_hidden))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.h2_weight = torch.nn.Parameter(torch.Tensor(reg_hidden, 1))
            torch.nn.init.normal_(self.h2_weight, mean=0, std=w_scale)
            self.last_w = self.h2_weight
        else:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.last_w = self.h1_weight

        # S2V scatter message passing
        self.scatter_aggr = (scatter_mean if self.avg else scatter_add)
        
# 首先对输入的节点和边特征进行线性变换和激活函数处理，然后进行了T次的图嵌入迭代。
# 在每次迭代中，计算节点到边的信息传递，然后计算边到节点的信息传递，最后更新节点的嵌入。
# !!!在这里还应该加入从之前数据到现在数据的转换，需要写一个新的函数
    def forward(self, data):
        '''
           xv: observation, nodes selected are 1, not selected yet are 0
           adj: adjacency matrix of the whole graph

           node_feat, edge_feat, adj in pytorch_geometric batch for varying 
           graph size

           node_input/node_feat: (batch_size x num_node) x node_feat
           edge_input/edge_feat: (batch_size x num_edge) x edge_feat
           adj: (batch_size x num_node) x num_node, sparse might be better
           action_select: batch_size x 1
           data.y: action_select, processed so that it can be directly used 
                   in a batch
           rep_global: in a batch, graph embedding for each node

        '''
        # (batch_size x num_node) x embed_size
        # num_node can vary for different graphs
        # 它接收一个包含图数据的data对象，该对象包含节点特征、边特征、邻接矩阵等信息。
        # 在前向传播过程中，首先将节点和边的特征通过相应的权重矩阵映射到嵌入空间，并应用ReLU激活函数。
        data.x = data.x
        data.x = torch.matmul(data.x, self.w_n2l)
        data.x = F.relu(data.x)
        # (batch_size x num_edge) x embed_size
        # num_edge can vary for different graphs
        data.edge_attr = torch.matmul(data.edge_attr, self.w_e2l)
        # 然后，通过多次迭代（由T决定）的消息传递和聚合操作来更新节点的嵌入表示。
        for _ in range(self.T):
            # (batch_size x num_node) x embed_size
            msg_linear = torch.matmul(data.x, self.p_node_conv)
            # n2esum_param sparse matrix to aggregate node embed to edge embed
            # (batch_size x num_edge) x embed_size
            n2e_linear = msg_linear[data.edge_index[0]]

            # (batch_size x num_edge) x embed_size
            edge_rep = torch.add(n2e_linear, data.edge_attr)
            edge_rep = F.relu(edge_rep)

            # e2nsum_param sparse matrix to aggregate edge embed to node embed
            # (batch_size x num_node) x embed_size
            e2n = self.scatter_aggr(edge_rep, data.edge_index[1], dim=0, dim_size=data.x.size(0))

            # (batch_size x num_node) x embed_size
            data.x = torch.add(torch.matmul(e2n, self.trans_node_1), 
                               torch.matmul(data.x, self.trans_node_2))
            data.x = F.relu(data.x)


        # 如果输入数据中包含了动作选择，那么将动作对应节点的嵌入和图的全局嵌入进行拼接，然后通过一个或两个线性层（取决于reg_hidden参数的值）得到预测的Q值。
        # 如果输入数据中没有动作选择，那么对所有节点进行同样的处理，得到所有动作的Q值。
        # subgsum_param sparse matrix to aggregate node embed to graph embed
        # batch_size x embed_size
        # torch_scatter can do broadcasting
        y_potential = self.scatter_aggr(data.x, data.batch, dim=0)
        # can concatenate budget to global representation
        # 最后，根据是否提供了动作选择（data.y），模型会计算并返回所有可能动作的Q值或给定动作的Q值。
        if data.y is not None: # Q func given a
            # batch_size x embed_size
            action_embed = data.x[data.y]

            # batch_size x (2 x embed_size)
            embed_s_a = torch.cat((action_embed, y_potential), dim=-1) # ConcatCols

            last_output = embed_s_a
            if self.reg_hidden > 0:
                # batch_size x reg_hidden
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = F.relu(hidden)
            # batch_size x 1
            q_pred = torch.matmul(last_output, self.last_w)

            return q_pred

        else: # Q func on all a
            rep_y = y_potential[data.batch]
            embed_s_a_all = torch.cat((data.x, rep_y), dim=-1) # ConcatCols

            last_output = embed_s_a_all
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                last_output = torch.relu(hidden)

            q_on_all = torch.matmul(last_output, self.last_w)

            return q_on_all

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, heads=1, dropout=0.0):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, heads=1, dropout=0.0):
        super(GATDecoder, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0):
        super(GATAutoencoder, self).__init__()
        self.encoder = GATEncoder(in_channels, hidden_channels, hidden_channels, heads, dropout)
        self.decoder = GATDecoder(hidden_channels, out_channels, hidden_channels, heads, dropout)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z, edge_index)
        return x_hat, z

def node_embed(graph, device, node_dim, model_path):
    num_epoch = 100
    # 从graph对象获取节点特征和边索引，并转换成张量
    x = torch.from_numpy(graph.get_node_features()).float().to(device)
    edge_index = torch.from_numpy(graph.get_edge_index()).long().to(device)
    # 定义模型
    in_channels = x.shape[1]
    hidden_channels = node_dim
    out_channels = in_channels  # 自编码器的输出维度应与输入维度相同
    model = GATAutoencoder(in_channels, hidden_channels, out_channels).to(device)

    # 检查模型文件是否存在
    if os.path.exists(model_path):
        # 加载模型
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # 创建一个包含x和edge_index的简单数据对象
        data = Data(x=x, edge_index=edge_index)
        # 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        # 训练模型
        for _ in range(num_epoch):
            model = train(model, optimizer, data)
        # 保存模型
        torch.save(model.state_dict(), model_path)
    
    data = Data(x=x, edge_index=edge_index)
    _, embedding = model(data.x, data.edge_index)
    return embedding.detach().clone()

# 定义无监督损失函数
def unsupervised_loss(model, data):
    x_hat, z = model(data.x, data.edge_index)
    loss = F.mse_loss(x_hat, data.x)  # 使用均方误差作为重构损失
    return loss

# 训练循环
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    loss = unsupervised_loss(model, data)
    loss.backward()
    optimizer.step()
    return model

# 示例图数据类
# class Graph:
#     def __init__(self, node_features, edge_index):
#         self.node_features = node_features
#         self.edge_index = edge_index

#     def get_node_features(self):
#         return self.node_features

#     def get_edge_index(self):
#         return self.edge_index

# # 示例使用
# if __name__ == "__main__":
#     # 示例图数据
#     node_features = np.random.rand(100, 16)  # 100个节点，每个节点有16维特征
#     edge_index = np.random.randint(0, 100, (2, 200))  # 200条边

#     graph = Graph(node_features, edge_index)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     node_dim = 32
#     model_path = "gat_autoencoder.pth"

#     embedding = node_embed(graph, device, node_dim, model_path)
#     print(embedding)