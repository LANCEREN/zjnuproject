import datetime
import random

import networkx
import networkx as nx
import numpy as np
from src.ZJNU.Influence_Prediction.LINE.line import LINE
from src.ZJNU.Influence_Prediction.utils import (
    create_networkx_graph,
    dynamic_network_partition,
    instance_normalization,
    static_network_partition,
)
from tqdm import tqdm


class LabelGenerator:
    def label_network(
        self, follow_network: networkx, retweet_network: networkx
    ) -> np.ndarray:
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
        node_label = np.array([(node, label) for node, label in zip(nodes, labels)])
        return node_label


class TimeWindowSplitter:
    # def split_by_time(self,static_network : str, retweet_network: str, idx_map, date_start_1, date_end_1, date_start_2, date_end_2) -> networkx:
    #     """按时间窗口划分转发关系网络"""
    #     # 定义日期范围
    #     # date_start_1 = datetime.date(2012, 7, 1)
    #     # date_end_1 = datetime.date(2012, 7, 4)
    #     # date_start_2 = datetime.date(2012, 7, 5)
    #     # date_end_2 = datetime.date(2012, 7, 7)
    #     static_data = np.loadtxt(static_network, dtype=str)
    #     # static_data = np.loadtxt(static_network, delimiter=',',dtype=str)
    #     # 打开输入文件和输出文件
    #     dynamic1 = nx.DiGraph()
    #     dynamic2 = nx.DiGraph()
    #     with open(retweet_network, "r") as infile:
    #         for line in infile:
    #             parts = line.strip().split()
    #             source_node = parts[0]
    #             target_node = parts[1]
    #             date_str = parts[2]  # 日期字符串 "YYYY-MM-DD"
    #
    #             # 转换字符串日期为 datetime.date 类型
    #             date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    #
    #             # 根据日期范围将数据写入相应的文件中
    #             if date_start_1 <= date <= date_end_1:
    #                 dynamic1.add_edge(source_node,target_node)
    #             elif date_start_2 <= date <= date_end_2:
    #                 dynamic2.add_edge(source_node,target_node)
    #     for row in static_data:
    #         dynamic1.add_node(row[0])
    #         dynamic1.add_node(row[1])
    #         dynamic2.add_node(row[0])
    #         dynamic2.add_node(row[1])
    #     dynamic1 = nx.relabel_nodes(dynamic1, idx_map)
    #     dynamic2 = nx.relabel_nodes(dynamic2, idx_map)
    #     return dynamic1, dynamic2

    def split_by_time(self, post_info_list, follower_network):
        """按时间窗口划分转发关系网络"""
        # 定义日期范围
        min_time = min(
            [
                datetime.datetime.strptime(post.publish_time, "%Y%m%d%H%M%S").date()
                for post in post_info_list
            ]
        )
        max_time = max(
            [
                datetime.datetime.strptime(post.publish_time, "%Y%m%d%H%M%S").date()
                for post in post_info_list
            ]
        )
        mid_time = min_time + (max_time - min_time) / 2
        dynamic1 = nx.DiGraph()
        dynamic2 = nx.DiGraph()
        # 根据日期范围将数据写入相应的文件中
        for post in post_info_list:
            if not post.is_original:
                date = datetime.datetime.strptime(
                    post.publish_time, "%Y%m%d%H%M%S"
                ).date()
                if date <= mid_time:
                    dynamic1.add_edge(post.userid, post.relevant_user_id)
                else:
                    dynamic2.add_edge(post.userid, post.relevant_user_id)
        for node in follower_network.nodes:
            dynamic1.add_node(node)
            dynamic2.add_node(node)
        return dynamic1, dynamic2


class DataProcessor:
    def load_networks_np(self, follow_network: np.ndarray, retweet_network: np.ndarray):
        """加载用户关注关系网络和转发关系网络"""
        pass

    def load_networks_txt(self, network: str) -> networkx:
        """加载用户关注关系网络和转发关系网络"""
        network, idx_map, idx = create_networkx_graph(network)
        return network, idx_map, idx

    def adj_matrix(self, network_path: networkx) -> np.ndarray:
        network = create_networkx_graph(network_path)
        adj_matrix = nx.adjacency_matrix(network).todense()
        return adj_matrix

    def load_networks_with_time_txt(
        self, static_network: str, dynamic_network: str, idx_map
    ) -> networkx:
        edges = np.loadtxt(dynamic_network, dtype=str)
        static_data = np.loadtxt(static_network, dtype=str)
        # static_data = np.loadtxt(static_network, delimiter=',',dtype=str)
        G = nx.DiGraph()
        for edge in edges:
            source, target = edge[0], edge[1]
            G.add_edge(source, target)
        for row in static_data:
            G.add_node(row[0])
            G.add_node(row[1])
        network = nx.relabel_nodes(G, idx_map)
        return network

    def preprocess_networks(self):
        """对网络数据进行预处理，如去重和归一化"""
        pass

    def load_static_network(self, account_info_list):
        """加载用户关注关系网络"""
        follower_network = nx.DiGraph()
        for account in account_info_list:
            account_id = account.account_id
            for friend in account.friends:
                follower_network.add_edge(account_id, friend.account_id)
            for follower in account.followers:
                follower_network.add_edge(follower.account_id, account_id)
        return follower_network

    def load_dynamic_network(self, post_info_list):
        retweet_network = nx.DiGraph()
        for post in post_info_list:
            if not post.is_original:
                retweet_network.add_edge(post.userid, post.relevant_user_id)
        return retweet_network


class NodeEmbedding:
    def generate_embeddings(self, G: networkx, embedding_dim: int) -> np.ndarray:
        """生成节点嵌入"""
        # 实例化 LINE 模型 设置模型参数
        model = LINE(
            dimension=embedding_dim,
            walk_length=20,
            walk_num=20,
            negative=10,
            batch_size=64,
            alpha=0.01,
            order=2,
        )
        embeddings = model.train(G)
        # embedding进行归一化
        embeddings = instance_normalization(embeddings)
        # node_embedding = {node: embeddings[i] for i, node in enumerate(sorted(G.nodes()))}
        # 创建一个 NumPy 数组来保存节点编号和嵌入
        node_ids = np.array(sorted(G.nodes()))
        # embedding_matrix = np.array([embeddings[i] for i in range(len(G.nodes()))])
        # 将节点编号和对应的嵌入存储为一个 NumPy 字典格式
        # embedding_data = {'node_ids': node_ids, 'embeddings': embedding_matrix}
        # 将节点编号和对应的嵌入存储为一个二维 NumPy 数组
        embedding_data = np.column_stack((node_ids, embeddings))
        return embedding_data


class FeatureExtractor:
    def extract_group_static_features(
        self, static_network: networkx, dynamic_network: networkx, k: int
    ) -> np.ndarray:
        """提取节点的群体特征"""
        average_perceptions = []
        fs = []
        nodes = list(sorted(static_network.nodes))
        for i in tqdm(nodes, desc="Processing nodes"):
            average_perception, f = static_network_partition(
                static_network, dynamic_network, target_user=i, top_k=k
            )
            average_perceptions.append(average_perception)
            fs.append(f)
        # 将节点编号、平均感知度和 f 值存储为 NumPy 数组
        node_data = np.array(
            [
                (node, perception, f_value)
                for node, perception, f_value in zip(nodes, average_perceptions, fs)
            ]
        )
        return node_data

    def extract_group_dynamic_features(
        self, dynamic_network: networkx, k: int
    ) -> np.ndarray:
        """提取节点的群体特征"""
        finals = []
        nodes = list(sorted(dynamic_network.nodes))
        for i in tqdm(nodes, desc="Processing nodes"):
            final = dynamic_network_partition(
                dynamic_network, np.array(dynamic_network.edges()), i, k
            )
            finals.append(final)
        # 将节点编号、 f 值存储为 NumPy 数组
        node_data = np.array([(node, f_value) for node, f_value in zip(nodes, finals)])
        return node_data
