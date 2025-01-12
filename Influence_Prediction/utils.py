import random

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def static_network_partition(static_network, retweet_network, target_user, top_k):

    # node_list = bfs_collect_nodes(static_network, target_user, 1)
    # node_list = list(static_network.neighbors(target_user))
    node_list = list(nx.single_source_shortest_path_length(static_network, target_user, cutoff=2).keys())
    second_order_graph = static_network.subgraph(node_list)
    perception_groups =rwr_with_decay(second_order_graph, target_user, top_k=top_k)
    # 计算目标用户的平均感知度
    average_perception = sum(perception_groups.values()) / len(perception_groups)
    # 计算目标用户的交互概率
    retweet_group = [(target_user, item) for item in perception_groups]
    f = 0
    for target_user, item in retweet_group:
        if retweet_edge_exist(target_user, item, retweet_network):
            f = f + 1
    f = f / len(perception_groups)
    return average_perception, f

def dynamic_network_partition(dynamic_network, data, target_user, top_k):
    # 获取目标用户的一阶邻居节点
    # node_list = bfs_collect_nodes(dynamic_network, target_user, 1)
    node_list = list(dynamic_network.neighbors(target_user))
    node_list.append(target_user)
    first_order_graph = dynamic_network.subgraph(node_list)
    perception_groups = rwr_with_decay(first_order_graph, target_user, top_k=top_k)
    c = 0
    retweet_group = [(target_user, item) for item in perception_groups]
    for target_user, item in retweet_group:
        # 计算源节点的转发次数
        for row in data:
            if row[0] == target_user and row[1] == item:
                c = c+1
    return c
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    # 有向图归一化
    mx = r_mat_inv_sqrt.dot(mx)
    # 无向图归一化
    # return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return mx

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def get_adj(adj):
    # 从 COO 格式获取行、列索引和数据
    row = torch.from_numpy(adj.row).long()
    col = torch.from_numpy(adj.col).long()
    data = torch.from_numpy(adj.data).float()
    # 创建边索引和边权重
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = data
    return row, col, edge_attr, edge_index

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def get_output(output):
    # 假设你有一个名为output的PyTorch张量，位于GPU上
    output = output.detach().cpu().numpy()
    # 将数组的索引添加到第一列，并转换为整数类型
    index_column = np.arange(output.shape[0], dtype=int)[:, np.newaxis]
    # 将第一列和数据列组合
    output_with_index = np.hstack((index_column, output))
    return output_with_index

def get_mapping(idx_map):
    # 将字典转换为包含键值对的列表
    data_to_save = np.array(list(idx_map.items()))
    # 保存到txt文件
    np.savetxt('node_mapping.txt', data_to_save, fmt='%d', delimiter=' ', comments='')

def create_networkx_graph(path, is_directed=True):

    if is_directed:
        # 创建一个空的有向图
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    # 读取文件
    edges_data = np.loadtxt(path, dtype=str)
    # 使用逗号作为分隔符读取 CSV 文件
    # edges_data = np.loadtxt(path, delimiter=',',dtype=str)
    # 添加边到图中
    for edge in edges_data:
         source, target = edge
         G.add_edge(source, target)
    idx_map = {j: i for i, j in enumerate(sorted(G.nodes))}
    idx = sorted(G.nodes)
    network = nx.relabel_nodes(G, idx_map)
    return network, idx_map, idx



def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()

def instance_normalization(X):
    """
    对输入向量 X 进行实例归一化。

    参数:
    X -- 输入向量。

    返回:
    X_norm -- 归一化后的向量。
    """
    # 计算向量的均值和标准差
    mean = np.mean(X)
    std = np.std(X)

    # 进行归一化
    X_norm = (X - mean) / (std + 1e-5)  # 1e-5 为数值稳定性添加的小常数

    return X_norm
def bfs_collect_nodes(graph, start_node, max_depth):
    visited = set()
    queue = [(start_node, 0)]
    while queue:
        current_node, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if current_node not in visited:
            visited.add(current_node)
            # 有向图
            neighbors = list(graph.successors(current_node))
            # 无向图
            # neighbors = list(graph.neighbors(current_node))  # 改为 neighbors
            for neighbor in neighbors:
                queue.append((neighbor, depth + 1))
    return visited

def retweet_edge_exist(source_node, target_node, retweet_network):
    return retweet_network.has_edge(source_node, target_node)

def rwr_with_decay(graph, target_user, decay_factor=0.8,restart_probability=0.8, max_steps=20, top_k=20):
    # 初始化周围节点的采样次数
    sampling_counts = {node: 0 for node in graph.nodes()}
    # 开始随机游走
    current_node = target_user
    for _ in range(max_steps):
        # 如果按照重启概率重新开始游走
        if random.random() < restart_probability:
            current_node = target_user
        else:
            # 随机选择一个邻居节点进行移动
            next_node_candidates = list(graph.neighbors(current_node))
            # try:
            #     next_node_candidates = list(graph.neighbors(current_node))
            # except Exception as e:
            #     breakpoint()
            #     raise e
            if next_node_candidates:
                next_node = random.choice(next_node_candidates)
                # 计算距离目标用户的距离
                distance = nx.shortest_path_length(graph, target_user, next_node)
                # 更新节点的采样次数，乘以衰减函数
                sampling_counts[next_node] += 1 * (decay_factor ** distance)
                # 更新当前节点为下一个节点
                current_node = next_node
    # 计算感知度pi
    perceptions = {node: count for node, count in sampling_counts.items()}
    # 对感知度进行排序
    sorted_perceptions = sorted(perceptions.items(), key=lambda x: x[1], reverse=True)
    perception_groups = {node: value for node, value in sorted_perceptions[:top_k]}
    return perception_groups


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)

    # 获取正样本和负样本的索引
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]

    # 打乱正负样本的顺序
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # 将正负样本的索引转换为列表
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    print(f"neg_idx: {len(neg_idx)}")
    print(f"pos_idx: {len(pos_idx)}")
    print(f"all_idx: {len(all_idx)}")
    # 确定验证集和测试集的样本数
    nb_val = round(val_prop * nb_nodes)
    nb_test = round(test_prop * nb_nodes)
    nb_train = nb_nodes - nb_val - nb_test

    # # 划分验证集、测试集和训练集
    # idx_val = all_idx[:nb_val]
    # idx_test = all_idx[nb_val:nb_val + nb_test]
    # idx_train = all_idx[nb_val + nb_test:]
    #
    # return idx_val, idx_test, idx_train
    # 确定验证集和测试集的样本数
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)

    # 划分验证集和测试集
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[nb_val + nb_test:]

    # 确保训练集正负样本均衡
    nb_train_pos = len(idx_train_pos)
    nb_train_neg = len(idx_train_neg)
    if nb_train_pos > nb_train_neg:
        idx_train_pos = idx_train_pos[:nb_train_neg]  # 截取正样本使其与负样本数量相等
    else:
        idx_train_neg = idx_train_neg[:nb_train_pos]  # 截取负样本使其与正样本数量相等

    # 返回验证集、测试集和训练集的索引
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg




def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1