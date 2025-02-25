import random

import networkx
import torch
import numpy as np
import networkx as nx
from src.ZJNU.data_structure import PostInfo, AccountInfo


class DataProcess:
    def __init__(self):
        pass


    def read_influence_features(self,file_path):
        data = np.load(file_path)
        # 获取第二列（假设数据是二维的）
        second_column = data[:, 1]
        # print(second_column)
        return torch.tensor(second_column, dtype=torch.float)

    def label_based_retweet(self,post_info_list, graph):
        # 创建一个字典存储每个用户的标签，默认都设置为 0
        user_labels = {node: 0 for node in graph.nodes()}

        # 遍历所有的转发帖子信息
        for post in post_info_list:
            if not post.is_original:  # 只考虑转发的帖子
                # 获取转发者和原创者
                user_A = post.userid  # 转发者
                user_B = post.relevant_user_id  # 原创者

                # 判断用户A是否关注了用户B
                if graph.has_edge(user_A, user_B):  # 如果 A -> B 的边存在，表示 A 关注了 B
                    user_labels[user_A] = 1  # 设置标签为 1，表示 A 转发了 B 且 A 关注了 B
                else:
                    user_labels[user_A] = 0  # 如果 A 没有关注 B，标签设置为 0

        return user_labels

    # 构建关注关系网络
    def construct_attnetwork(self,account_info_list):
        # G = nx.DiGraph()  # 使用有向图来表示关注关系
        #
        # # 遍历每个用户的账号信息
        # for account_info in account_info_list:
        #     # 将每个用户作为一个节点加入图中
        #     G.add_node(account_info.account_id)
        #
        #     # 遍历该用户的粉丝列表，建立用户与粉丝之间的有向边（表示关注关系）
        #     for follower in account_info.followers:
        #         # 添加边：粉丝关注用户
        #         G.add_edge(follower.account_id, account_info.account_id)  # follower -> user
        #
        # return G
        """加载用户关注关系网络"""
        follower_network = nx.DiGraph()
        for account in account_info_list:
            account_id = account.account_id
            for friend in account.friends:
                follower_network.add_edge(account_id, friend.account_id)
            for follower in account.followers:
                follower_network.add_edge(follower.account_id, account_id)
        return follower_network

    def construct_node_features(self,account_info_list,node_mapping):
        node_features = [0]*len(node_mapping)
        for account in account_info_list:
            # 获取每个账号的映射索引
            account_index = node_mapping[account.account_id]
            # print(account_index, account.account_id)
            # 提取user_feature中的特征
            user_feature = account.user_feature
            # 拼接user_feature中的特征（例如：gender, verified, contents_count等）和影响力
            feature = torch.tensor([
                user_feature.gender,  # 性别
                account.influence,  # 影响力
                user_feature.verified,  # 认证状态
                user_feature.ip,  # IP地址
                user_feature.contents_count,  # 发布内容数
                user_feature.friends_count,  # 好友数
                user_feature.followers_count,  # 粉丝数
                user_feature.platform,  # 平台标识符

            ], dtype=torch.float)
            from sklearn.preprocessing import MinMaxScaler

            # 获取数值型特征（从第 2 个索引开始）
            numeric_features = feature[2:]

            # 创建一个 MinMaxScaler 实例
            scaler = MinMaxScaler(feature_range=(0, 1))

            # 将数值型特征转换为二维数组，以便进行 MinMaxScaler 处理
            numeric_features_reshaped = numeric_features.view(-1, 1).numpy()  # 转换为二维数组

            # 使用 MinMaxScaler 进行归一化
            normalized_numeric_features = scaler.fit_transform(numeric_features_reshaped)

            # 将归一化后的特征转换回 Tensor，并展平成一维张量
            normalized_feature = torch.tensor(normalized_numeric_features.flatten(), dtype=torch.float)

            # 显式将切片部分转换为张量副本，避免视图问题
            feature[2:] = normalized_feature  # 将归一化后的数值型特征赋回原 tensor

            # 输出最终的 feature 张量
            # print(feature)
            # 拼接用户嵌入向量
            feature = torch.cat((feature, account.user_embeddings), dim=0)

            # 将该特征放入相应索引位置
            node_features[account_index] = feature

        # 转换为Tensor
        return torch.stack(node_features)



    def construct_retnetwork(self, post_info_list):
        retweet_network = nx.DiGraph()
        for post in post_info_list:
            if not post.is_original:
                retweet_network.add_edge(post.userid, post.relevant_user_id)
        return retweet_network

    def label_network(self,follow_network: networkx, retweet_network: networkx) -> np.ndarray:
        """为静态网络数据生成标签"""
        follow_edges = set(follow_network.edges())
        nodes = list(sorted(follow_network.nodes))
        labels = []
        # 读取转发文件中的数据并构建源节点集合
        source_nodes = set()
        retweet_edges = set(retweet_network.edges())
        for line in retweet_edges:
            if line in follow_edges:  # 只有关注关系中也存在的边才计入源节点集合
                source_nodes.add(line[0])

        for node in nodes:
            if node in source_nodes:
                labels.append(1)
            else:
                labels.append(0)
        # 随机挑选100个数值，将值改为1
        indices = random.sample(range(len(nodes)), 100)
        for node in nodes:
            if node in source_nodes:
                labels.append(1)
            else:
                labels.append(0)

        # 将labels中的100个随机值更改为1
        for i in indices:
            labels[i] = 1
        node_label = np.array([(node, label) for node, label in
                              zip(nodes, labels)])
        return node_label

    def split_data(self,labels, val_prop, test_prop, seed):
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
        idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                       nb_val + nb_test:]
        idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                       nb_val + nb_test:]

        # 确保训练集正负样本均衡
        nb_train_pos = len(idx_train_pos)
        nb_train_neg = len(idx_train_neg)
        if nb_train_pos > nb_train_neg:
            idx_train_pos = idx_train_pos[:nb_train_neg]  # 截取正样本使其与负样本数量相等
        else:
            idx_train_neg = idx_train_neg[:nb_train_pos]  # 截取负样本使其与正样本数量相等

        # 返回验证集、测试集和训练集的索引
        return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg