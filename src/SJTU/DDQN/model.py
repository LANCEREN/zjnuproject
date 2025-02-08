import argparse
import os
import random
import warnings

# import utils.graph_utils as graph_utils
from collections import deque

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv

# from federatedscope.gfl.model import SAGE_Net
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_softmax
from tqdm import tqdm

from src.ZJNU.Analog_Propagation.Finetuning_GAT.DataProcess import DataProcess
from src.ZJNU.Analog_Propagation.Finetuning_GAT.model import GAT  # 导入模型

warnings.filterwarnings("ignore")


class GNN_DDQN(nn.Module):
    """to check and verify with the design in the paper"""

    def __init__(self, reg_hidden, embed_dim, node_dim, edge_dim, T, w_scale, avg):
        """w_scale=0.01, node_dim=2, edge_dim=4"""
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
        self.w_n2l = torch.nn.Parameter(torch.Tensor(node_dim + 1, embed_dim))
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
        self.scatter_aggr = scatter_mean if self.avg else scatter_add

    # 首先对输入的节点和边特征进行线性变换和激活函数处理，然后进行了T次的图嵌入迭代。
    # 在每次迭代中，计算节点到边的信息传递，然后计算边到节点的信息传递，最后更新节点的嵌入。
    # !!!在这里还应该加入从之前数据到现在数据的转换，需要写一个新的函数
    def forward(self, data):
        """
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

        """
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
            e2n = self.scatter_aggr(
                edge_rep, data.edge_index[1], dim=0, dim_size=data.x.size(0)
            )

            # (batch_size x num_node) x embed_size
            data.x = torch.add(
                torch.matmul(e2n, self.trans_node_1),
                torch.matmul(data.x, self.trans_node_2),
            )
            data.x = F.relu(data.x)

        # 如果输入数据中包含了动作选择，那么将动作对应节点的嵌入和图的全局嵌入进行拼接，然后通过一个或两个线性层（取决于reg_hidden参数的值）得到预测的Q值。
        # 如果输入数据中没有动作选择，那么对所有节点进行同样的处理，得到所有动作的Q值。
        # subgsum_param sparse matrix to aggregate node embed to graph embed
        # batch_size x embed_size
        # torch_scatter can do broadcasting
        y_potential = self.scatter_aggr(data.x, data.batch, dim=0)
        # can concatenate budget to global representation
        # 最后，根据是否提供了动作选择（data.y），模型会计算并返回所有可能动作的Q值或给定动作的Q值。
        if data.y is not None:  # Q func given a
            # batch_size x embed_size
            action_embed = data.x[data.y]

            # batch_size x (2 x embed_size)
            embed_s_a = torch.cat((action_embed, y_potential), dim=-1)  # ConcatCols

            last_output = embed_s_a
            if self.reg_hidden > 0:
                # batch_size x reg_hidden
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = F.relu(hidden)
            # batch_size x 1
            q_pred = torch.matmul(last_output, self.last_w)

            return q_pred

        else:  # Q func on all a
            rep_y = y_potential[data.batch]
            embed_s_a_all = torch.cat((data.x, rep_y), dim=-1)  # ConcatCols

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
        self.conv2 = GATConv(
            hidden * heads, out_channels, heads=1, concat=False, dropout=dropout
        )

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
        self.conv2 = GATConv(
            hidden * heads, out_channels, heads=1, concat=False, dropout=dropout
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATAutoencoder(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0
    ):
        super(GATAutoencoder, self).__init__()
        self.encoder = GATEncoder(
            in_channels, hidden_channels, hidden_channels, heads, dropout
        )
        self.decoder = GATDecoder(
            hidden_channels, out_channels, hidden_channels, heads, dropout
        )

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


class FinetuningGAT(torch.nn.Module):
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        precision = precision_score(y_true, (y_pred > 0.5).astype(int))
        recall = recall_score(y_true, (y_pred > 0.5).astype(int))
        f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
        return accuracy, precision, recall, f1

    def split_masks(self, data):
        # 划分训练集和测试集
        if data.num_nodes < 20:
            train_mask = np.ones(data.num_nodes, dtype=bool)  # 所有节点都用于训练
            val_mask = np.ones(data.num_nodes, dtype=bool)  # 所有节点都用于验证
            test_mask = np.ones(data.num_nodes, dtype=bool)
        else:
            train_mask = np.random.rand(data.num_nodes) < 0.8
            val_mask = (np.random.rand(data.num_nodes) >= 0.8) & (
                np.random.rand(data.num_nodes) < 0.9
            )  # 10% 验证集
            test_mask = np.random.rand(data.num_nodes) >= 0.9

        return train_mask, val_mask, test_mask

    # 训练和测试过程封装成函数
    def train_and_test(self, model, data, train_mask, test_mask, optimizer, loss_fn):
        optimizer.zero_grad()
        output = model(data)[train_mask]  # 只计算训练集的输出
        loss = loss_fn(output, data.y[train_mask].view(-1, 1))
        loss.backward()
        optimizer.step()

        # 测试集上的指标和保存转发概率
        model.eval()
        with torch.no_grad():
            test_output = model(data)[test_mask]
            test_prob = torch.sigmoid(test_output)
            test_pred = (test_prob > 0.5).float()
            # 确保 y_true 和 y_pred 是正确的二进制类型
            y_true = data.y[test_mask].int()  # 强制转换为整数类型
            y_pred = test_pred.int()  # 强制转换为整数类型
            # test_acc, test_prec, test_rec, test_f1 = self.calculate_metrics(y_true, y_pred)  # 计算测试集指标

            test_acc, test_prec, test_rec, test_f1 = self.calculate_metrics(
                data.y[test_mask], test_pred
            )

        # 输出转发概率并保存
        output_prob = torch.sigmoid(output)
        retweet_probabilities = output_prob.squeeze()  # 得到一维向量
        return retweet_probabilities

    def FinetunningGAT(self, account_info_list, post_info_list):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="Disables CUDA training.",
        )
        parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
        # parser.add_argument('--node_features_file', type=str, default='../zjnuproject-master1/Influence_Prediction/out/test.content',
        #                     help='Path to the node features file')
        # parser.add_argument('--edges_file', type=str, default='../zjnuproject-master1/Feature_Extract/out/微博用户关注关系.txt',
        #                     help='Path to the edges file')
        parser.add_argument("--in_channels", type=int, help="Number of input features")
        parser.add_argument(
            "--out_channels",
            type=int,
            help="Number of output channels (number of classes)",
        )
        parser.add_argument(
            "--heads", type=int, default=4, help="Number of attention heads"
        )
        parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
        parser.add_argument(
            "--weight_decay", type=float, default=5e-4, help="Weight decay coefficient"
        )
        parser.add_argument(
            "--num_epochs", type=int, default=100, help="Number of training epochs"
        )
        parser.add_argument(
            "--test_size",
            type=float,
            default=0.1,
            help="Proportion of the dataset to include in the test split",
        )
        parser.add_argument(
            "--random_state",
            type=int,
            default=42,
            help="Random state for reproducibility",
        )
        # parser.add_argument('--output_file', type=str, default='../zjnuproject-master1/Finetuning_GAT/output/predicted_retweet_probabilities_3.txt',
        #                     help='Path to the output file for predicted retweet probabilities')

        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        # 1.加载数据：用户特征+影响力特征+网络结构
        data_process = DataProcess()
        # # 数据集路径
        # node_features_file = args.node_features_file
        # edges_file = args.edges_file
        # influence_vector_file = '../Finetuning_GAT/data/test/influence.npy'
        # influence_vector_file = '../zjnuproject-master1/Influence_Prediction/out/influence.npy'

        # 加载数据
        # x, y = data_process.read_node_features(node_features_file)
        # edge_index = data_process.read_edges(edges_file)
        # 1.构建关注关系并获取边索引列表
        attention_graph = data_process.construct_attnetwork(account_info_list)
        node_mapping = {node: i for i, node in enumerate(attention_graph.nodes())}
        node_features = data_process.construct_node_features(
            account_info_list, node_mapping
        )
        node_features = torch.tensor(
            node_features, dtype=torch.float32
        )  # 确保节点特征为float32
        edges = [(node_mapping[u], node_mapping[v]) for u, v in attention_graph.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # 2.设置节点的转发标签
        positive_posts = [post for post in post_info_list if post.sentiment == 1]
        negative_posts = [post for post in post_info_list if post.sentiment == -1]
        # node_labels = data_process.label_based_retweet(post_info_list,attention_graph)
        pos_node_labels = data_process.label_based_retweet(
            positive_posts, attention_graph
        )
        neg_node_labels = data_process.label_based_retweet(
            negative_posts, attention_graph
        )
        new_posnode_labels = {}
        new_negnode_labels = {}
        for origin_node_id, label in pos_node_labels.items():
            new_node_id = node_mapping[origin_node_id]
            new_posnode_labels[new_node_id] = label
        sorted_posnode_labels = dict(sorted(new_posnode_labels.items()))
        y_pos = torch.tensor(list(sorted_posnode_labels.values()), dtype=torch.float32)
        # print(y_pos)
        for origin_node_id, label in neg_node_labels.items():
            new_node_id = node_mapping[origin_node_id]
            new_negnode_labels[new_node_id] = label
        sorted_negnode_labels = dict(sorted(new_negnode_labels.items()))
        y_neg = torch.tensor(list(sorted_negnode_labels.values()), dtype=torch.float32)
        # print(y_neg)
        # influence_vector = data_process.read_influence_features(influence_vector_file)
        # node_features = data_process.concatenate_features_and_influence(x,influence_vector)
        # print(node_features.shape)
        # print(node_features)

        # 构建 PyG 图数据对象
        data1 = Data(x=node_features, edge_index=edge_index, y=y_pos)  # 将标签 y 也传入数据对象
        data2 = Data(x=node_features, edge_index=edge_index, y=y_neg)
        data1 = data1.to(device)
        data2 = data2.to(device)
        # 模型超参数
        in_channels = data1.num_node_features  # 输入特征维度
        out_channels = 1  # 输出维度
        heads = 4  # 多头注意力的头数
        # 保存转发概率的列表
        retweet_prob_lists = []
        # 创建模型

        model = GAT(in_channels=in_channels, out_channels=out_channels, heads=heads).to(
            device
        )
        # model = GAT(in_channels=in_channels, out_channels=out_channels, heads=heads)

        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()  # 二分类问题使用BCEWithLogitsLoss

        data_sets = [data1, data2]

        # 模型训练和测试过程
        for data in data_sets:
            train_mask, val_mask, test_mask = self.split_masks(data)

            # 模型训练
            model.train()
            for epoch in range(args.num_epochs):
                retweet_probabilities = self.train_and_test(
                    model, data, train_mask, test_mask, optimizer, loss_fn
                )

                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        train_output = model(data)[train_mask]
                        test_output = model(data)[test_mask]

                        train_prob = torch.sigmoid(train_output)
                        test_prob = torch.sigmoid(test_output)

                        train_pred = (train_prob > 0.5).float()  # 假设阈值为0.5
                        test_pred = (test_prob > 0.5).float()

                        # 计算训练集和测试集指标
                        (
                            train_acc,
                            train_prec,
                            train_rec,
                            train_f1,
                        ) = self.calculate_metrics(data.y[train_mask], train_pred)
                        test_acc, test_prec, test_rec, test_f1 = self.calculate_metrics(
                            data.y[test_mask], test_pred
                        )
                        # 打印训练集和测试集的指标
                        print(f"Epoch [{epoch + 1}/{args.num_epochs}]")
                        print(
                            f"Train - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}"
                        )
                        print(
                            f"Test - Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}"
                        )

                    model.train()

            # 保存结果
            # output_file = args.output_file
            # retweet_prob_list = []
            # # with open(output_file, 'w') as f:
            # for idx, prob in enumerate(retweet_probabilities):
            #     formatted_prob = round(prob.item(), 6)  # 保留6位小数
            #     # f.write(f"{idx} {prob.item():.6f}\n")  # 保存节点索引和对应的转发概率到文件中
            #     retweet_prob_list.append((idx, formatted_prob))

            # 假设account_info_list已经定义
            # retweet_prob_list = [0] * len(account_info_list)  # 初始化为与account_info_list相同长度的空值列表
            retweet_prob_list = [(i, 0.0) for i in range(len(account_info_list))]
            for idx, prob in enumerate(retweet_probabilities):
                # 格式化概率值，保留6位小数
                formatted_prob = round(prob.item(), 6)
                # 更新retweet_prob_list中的对应位置
                if idx < len(retweet_prob_list):
                    retweet_prob_list[idx] = (idx, formatted_prob)

            # 返回所有数据集的转发概率列表
            retweet_prob_lists.append(retweet_prob_list)

        # 返回所有数据集的转发概率列表
        return retweet_prob_lists[0], retweet_prob_lists[1]

    def update_retweet_prob(self, account_info_list, post_info_list):
        retweet_pos_probability, retweet_neg_probability = self.FinetunningGAT(
            account_info_list, post_info_list
        )
        for i, account in enumerate(account_info_list):
            account.retweet_pos_probability = round(retweet_pos_probability[i][1], 6)
            account.retweet_neg_probability = round(retweet_neg_probability[i][1], 6)
        return account_info_list
