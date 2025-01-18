import copy
import time
import random
import math
import statistics
import networkx as nx
import community
import os
import json
import csv
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any 
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from DDQN_Interface import DDQNInterface
from pathlib import Path 
from IMPIM_SJTU.DDQN.config import dataPath


class Graph:
    def __init__(self, nodes, edges, children, parents,node_feature_path): 
        self.nodes = nodes # set()
        self.edges = edges # dict{(src,dst): weight, }
        self.children = children # dict{node: set(), }
        self.parents = parents # dict{node: set(), }
        self.G = nx.DiGraph()
        self._build_nx_graph()
        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])
        
        if node_feature_path:
            # 读取外部输入的特征
            with open(node_feature_path, 'r') as f:
                external_features = json.load(f)
        else:
            # 调用 get_node_features 函数获取节点特征
            self.get_node_features()
            return

        # 创建一个字典来存储外部特征，键为 account_id
        external_features_dict = {feature_dict['account_id']: feature_dict for feature_dict in external_features}

        # 获取 personal_desc 的最大长度
        max_personal_desc_length = max(len(feature_dict.get('personal_desc', [])) for feature_dict in external_features)

        pagerank = nx.pagerank(self.G)
        for node in self.G.nodes():
            importance = pagerank[node]
            
            # 初始化特征列表
            features = [importance]
            
            # 查找对应的外部特征
            if node in external_features_dict:
                feature_dict = external_features_dict[node]
                followers_count = feature_dict.get('followers_count', 0)
                friends_count = feature_dict.get('friends_count', 0)
                personal_desc = feature_dict.get('personal_desc', [])
                
                # 添加特征到特征列表
                features.append(followers_count)
                features.append(friends_count)
                features.extend(personal_desc)
                
                # 如果 personal_desc 长度不足，则补零
                if len(personal_desc) < max_personal_desc_length:
                    features.extend([0] * (max_personal_desc_length - len(personal_desc)))
            else:
                # 如果没有外部特征，则生成同维度的全零特征
                features.extend([0, 0])
                features.extend([0] * max_personal_desc_length)

            # 将所有特征合并
            self.G.nodes[node]['features'] = features
    
    def get_node_features(self):
        pagerank = nx.pagerank(self.G)
        for node in self.G.nodes():
            out_degree = self.G.out_degree(node)
            in_degree = self.G.in_degree(node)
            importance = pagerank[node]
            self.G.nodes[node]['features'] = [out_degree, in_degree, importance]

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])
    
    def _build_nx_graph(self):
        for (src, dst), weight in self.edges.items():
            self.G.add_edge(src, dst, weight=weight)

    def get_edge_index(self):
        '''Return edge index similar to PyG data.edge_index'''
        # 从self.edges中提取边的源节点和目标节点
        edge_index = np.array([[src, dst] for src, dst in self.edges.keys()]).T
        return edge_index
    
    def get_edge_weight(self):
        '''Return edge weight'''
        edge_weight = np.array([self.edges[edge] for edge in self.edges])
        return edge_weight

    def compute_node_features(self):
        pagerank = nx.pagerank(self.G)
        for node in self.G.nodes():
            out_degree = self.G.out_degree(node)
            in_degree = self.G.in_degree(node)
            importance = pagerank[node]
            self.G.nodes[node]['features'] = [out_degree, in_degree, importance]

    def get_node_features(self):
        self.compute_node_features()
        node_features = np.array([self.G.nodes[node]['features'] for node in self.G.nodes()])
        return node_features

def read_graph(path, node_feature_path, ind=0, directed=False):
    ''' method to load edge as node pair graph '''
    parents = {}
    children = {}
    edges = {}
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # if not len(line) or line.startswith('#') or line.startswith('%'):
            #     continue
            row = line.split()
            src = int(row[0]) - ind
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst)
            parents.setdefault(dst, set()).add(src)
            edges[(src, dst)] = 0.0
            # if not(directed):
            #     # regard as undirectional
            #     children.setdefault(dst, set()).add(src)
            #     parents.setdefault(src, set()).add(dst)
            #     edges[(dst, src)] = 0.0

    # change the probability to 1/indegree
    # Calculate weights based on intersection and union of neighbors
    for src, dst in edges:
        src_neighbors = children.get(src, set()) | parents.get(src, set())
        dst_neighbors = children.get(dst, set()) | parents.get(dst, set())
        intersection = src_neighbors & dst_neighbors
        union = src_neighbors | dst_neighbors
        if len(intersection) > 0:
            weight = len(intersection) / len(union)
        else:
            weight = 0.1
        edges[(src, dst)] = weight
    # for src, dst in edges:
    #     edges[(src, dst)] = 1.0 / len(parents[dst])
            
    return Graph(nodes, edges, children, parents,node_feature_path)

# 完整的数据处理代码
def preprocess_graph(file_path=None, ori_feature_path=None, output_dir=None):
    # 先处理图
    graph_path, _, node_feature_path = process_graph(file_path, ori_feature_path, output_dir)
    # 再按照平台进行划分
    # platform_split(output_dir)
    # # 再处理划分后的图
    # platform_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('platform_')]
    # # 这里保存的t和subgraphs_dir只是针对一个平台后面补充多平台保存
    # for platform_dir in platform_dirs:
    #     platform_path = os.path.join(output_dir, platform_dir)
    #     platform_graph_path = os.path.join(platform_path, 'graph.txt')
    #     t = 0
    #     subgraphs_dir = split_graph(platform_graph_path,platform_dir)
    #     for root, dirs, files in os.walk(subgraphs_dir):
    #         for file in files:
    #             if file.endswith('.txt') and file.startswith('subgraph'):
    #                 subgraph_path = os.path.join(root, file)
    #                 process_split_graph(subgraph_path, root, node_feature_path, t)
    #                 t += 1
    #     return subgraphs_dir,t
    t = 0
    subgraphs_dir = split_graph(graph_path, output_dir)
    for root, dirs, files in os.walk(subgraphs_dir):
        for file in files:
            if file.endswith('.txt') and file.startswith('subgraph'):
                subgraph_path = os.path.join(root, file)
                process_split_graph(subgraph_path, root, node_feature_path, t)
                t += 1
    if t == 0:
        return output_dir, 1
    return subgraphs_dir, t

def process_graph(file_path =None, node_feature_path=None, output_dir=None):
    # 读取图数据
    edge_list = []
    node_set = set()
    # try:
    #     with open(file_path, 'r') as file:
    #         print(f"Successfully opened file: {file_path}")
    #         for line in file:
    #             print(line)
    # except FileNotFoundError:
    #     print(f"File not found: {file_path}")
    # except Exception as e:
    #     print(f"An error occurred while reading the file: {e}")
    with open(file_path, 'r') as file:
        for line in file:
            if ',' in line:
                parts = line.strip().split(',')
            else:
                parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, timestamp = int(parts[0]), int(parts[1]), parts[2]
                edge_list.append((node1, node2, timestamp))
            elif len(parts) == 2:
                node1, node2 = int(parts[0]), int(parts[1])
                edge_list.append((node1, node2))
            node_set.update([node1, node2])

    # 为每个节点分配一个从零开始的新编号
    node_list = sorted(node_set)
    # print(len(node_list))
    node_mapping = {node: idx for idx, node in enumerate(node_list)}

    with open(node_feature_path, 'r') as f:
        external_features = json.load(f)
    # 获取 external_features 列表的长度并打印出来
    # external_features_length = len(external_features)
    # print(f"Length of external_features: {external_features_length}")
    # 更新外部特征中的节点编号
    for feature_dict in external_features:
        old_account_id = feature_dict.get('account_id')
        if isinstance(old_account_id, str) and old_account_id.strip():  # 检查 old_account_id 是否为非空字符串
            try:
                old_account_id = int(old_account_id)
            except ValueError as e:
                # print(f"Skipping invalid account_id: {old_account_id} - {e}")
                continue  # 跳过无效的 account_id
        if old_account_id in node_mapping:
            feature_dict['account_id'] = node_mapping[old_account_id]

    
    # 保存更新后的 external_features 为新的 JSON 文件
    new_feature_path = output_dir / Path('node_features.json')
    with open(new_feature_path, 'w') as f:
        json.dump(external_features, f, indent=4) 
    
    # 重构图数据，使用新的节点编号
    if len(parts) == 3:
        new_edge_list = [(node_mapping[node1], node_mapping[node2], timestamp) for node1, node2, timestamp in edge_list]
    elif len(parts) == 2:
        new_edge_list = [(node_mapping[node1], node_mapping[node2]) for node1, node2 in edge_list]
    # 保存新的图数据
    new_graph_file = output_dir / Path('graph.txt')
    with open(new_graph_file, 'w') as file:
        if len(parts) == 3:
        ### 这里应该在存储时应该保留时间戳
            for node1, node2, timestamp in new_edge_list:
                file.write(f"{node1} {node2}\n")
        elif len(parts) == 2:
            for node1, node2 in new_edge_list:
                file.write(f"{node1} {node2}\n")

    # 保存节点编号映射文件
    mapping_file = output_dir / Path('node_mapping.txt')
    with open(mapping_file, 'w') as file:
        for node, new_id in node_mapping.items():
            file.write(f"{node} {new_id}\n")
    return new_graph_file,mapping_file,new_feature_path

# 未来如果有跨平台数据可以用这个函数
def platform_split(path):
    graph_path = path / Path('graph.txt')
    node_feature_path = path / Path('node_features.json')
    with open(node_feature_path, 'r') as f:
        external_features = json.load(f)
    # 创建一个字典来存储每个平台的节点
    platform_nodes = {}
    for feature_dict in external_features:
        node_id = feature_dict.get('account_id')
        node_platform_id = feature_dict.get('platform')
        if node_platform_id not in platform_nodes:
            platform_nodes[node_platform_id] = set()
        platform_nodes[node_platform_id].add(node_id)
    
    # 读取图数据
    edge_list = []
    with open(graph_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, timestamp = int(parts[0]), int(parts[1]), parts[2]
                edge_list.append((node1, node2, timestamp))
            elif len(parts) == 2:
                node1, node2 = int(parts[0]), int(parts[1])
                edge_list.append((node1, node2))

    # 按照节点所在平台进行分割
    platform_edges = {}
    for edge in edge_list:
        node1, node2 = edge[:2]
        platforms = set()
        for platform, nodes in platform_nodes.items():
            if node1 in nodes:
                platforms.add(platform)
            if node2 in nodes:
                platforms.add(platform)
        
        if len(platforms) == 1:
            # 两个节点在同一个平台
            platform = platforms.pop()
            if platform not in platform_edges:
                platform_edges[platform] = []
            platform_edges[platform].append(edge)
        else:
            # 两个节点不在同一个平台
            for platform in platforms:
                if platform not in platform_edges:
                    platform_edges[platform] = []
                platform_edges[platform].append(edge)
                # 添加不存在的节点
                if node1 not in platform_nodes[platform]:
                    platform_nodes[platform].add(node1)
                if node2 not in platform_nodes[platform]:
                    platform_nodes[platform].add(node2)
    
    # 保存分割后的图数据
    for platform, edges in platform_edges.items():
        platform_dir = path / Path(f'platform_{platform}')
        os.makedirs(platform_dir, exist_ok=True)
        platform_graph_path = platform_dir / Path('graph.txt')
        with open(platform_graph_path, 'w') as file:
            for edge in edges:
                if len(edge) == 3:
                    node1, node2, timestamp = edge
                    file.write(f"{node1} {node2} {timestamp}\n")
                elif len(edge) == 2:
                    node1, node2 = edge
                    file.write(f"{node1} {node2}\n")

def split_graph(graph_path, dir):
    # 创建一个示例图
    G = nx.DiGraph()
    G = G.to_undirected()
    with open(graph_path, 'r') as f:
        for line in f:
            node1, node2 = line.strip().split()
            G.add_edge(node1, node2)

    # 使用 Louvain 算法进行社区检测
    partition = community.best_partition(G, resolution=0.1, randomize=False)

    # 将节点按照社区分组
    community_groups = {}
    for node, community_id in partition.items():
        if community_id not in community_groups:
            community_groups[community_id] = []
        community_groups[community_id].append(node)

    # 创建每个社区的子图并保存为文件
    output_dir = dir / Path('subgraphs')
    os.makedirs(output_dir, exist_ok=True)

    for community_id, nodes in community_groups.items():
        subgraph = G.subgraph(nodes).copy()  # 提取子图
        num_nodes = subgraph.number_of_nodes()
        # 如果子图的节点数目少于 500，则跳过
        if num_nodes < 500:
            continue
        # print('nodes:', community_id, num_nodes)
        
        # 保存子图
        output_file_path = output_dir / Path(f"subgraph_{community_id}.txt")
        with open(output_file_path, 'w') as output_file:
            for edge in subgraph.edges():
                output_file.write(f"{edge[0]} {edge[1]}\n")
    # print(f"{platform_dir} Graph split completed.")
    return output_dir
    
def process_split_graph(file_path =None, output_dir=None, node_feature_path=None,t=0):
    # 读取图数据
    edge_list = []
    node_set = set()
    with open(file_path, 'r') as file:
        for line in file:
            if ',' in line:
                parts = line.strip().split(',')
            else:
                parts = line.strip().split()
            if len(parts) == 3:
                node1, node2, timestamp = int(parts[0]), int(parts[1]), parts[2]
                edge_list.append((node1, node2, timestamp))
            elif len(parts) == 2:
                node1, node2 = int(parts[0]), int(parts[1])
                edge_list.append((node1, node2))
            node_set.update([node1, node2])

    # 为每个节点分配一个从零开始的新编号
    node_list = sorted(node_set)
    node_mapping = {node: idx for idx, node in enumerate(node_list)}

    with open(node_feature_path, 'r') as f:
        external_features = json.load(f)
    
    # 更新外部特征中的节点编号
    for feature_dict in external_features:
        old_account_id = feature_dict.get('account_id')
        if isinstance(old_account_id, str):
            old_account_id = int(old_account_id)
        if old_account_id in node_mapping:
            feature_dict['account_id'] = node_mapping[old_account_id]

    output_dir = output_dir / Path(f'subgraph_{t}')
    os.makedirs(output_dir,exist_ok=True)
    # 保存更新后的 external_features 为新的 JSON 文件
    new_feature_path = output_dir / Path('node_features.json')
    with open(new_feature_path, 'w') as f:
        json.dump(external_features, f, indent=4) 
    
    # 重构图数据，使用新的节点编号
    if len(parts) == 3:
        new_edge_list = [(node_mapping[node1], node_mapping[node2], timestamp) for node1, node2, timestamp in edge_list]
    elif len(parts) == 2:
        new_edge_list = [(node_mapping[node1], node_mapping[node2]) for node1, node2 in edge_list]
    # 保存新的图数据
    new_graph_file = output_dir / Path('graph.txt')
    with open(new_graph_file, 'w') as file:
        if len(parts) == 3:
        ### 这里应该在存储时应该保留时间戳
            for node1, node2, timestamp in new_edge_list:
                file.write(f"{node1} {node2}\n")
        elif len(parts) == 2:
            for node1, node2 in new_edge_list:
                file.write(f"{node1} {node2}\n")

    # 保存节点编号映射文件
    mapping_file = output_dir / Path('node_mapping.txt')
    with open(mapping_file, 'w') as file:
        for node, new_id in node_mapping.items():
            file.write(f"{node} {new_id}\n")
    return new_graph_file,mapping_file,new_feature_path

def user_map(seeds_list, node_map_dir):
    node_map_path = node_map_dir / Path('node_mapping.txt')
    original_ids = []

    # 读取映射表
    with open(node_map_path, 'r') as file:
        mapping = {}
        for line in file:
            original_id, new_id = line.strip().split()
            mapping[int(new_id)] = int(original_id)

    # 根据映射表找到原先的节点 ID
    for seed in seeds_list:
        if seed in mapping:
            original_ids.append(mapping[seed])
        # else:
        #     print(f"Warning: Seed {seed} not found in mapping")
    # print(original_ids)
    return original_ids   

# 将接口输入转换成算法输入的函数
def process_graph_api(ddqn_interface:DDQNInterface,dataset_name='sansuo'):     
    # 使用 DDQNInterface 类中的 budget
    budget = ddqn_interface.budget 

    # 将用户特征保存为 JSON 文件
    user_features = []
    post_features = []
    for user in ddqn_interface.user_feature:
        user_dict = {
            'account_id': user.account_id,
            'personal_desc_tensor': user.personal_desc_tensor.tolist(),  # 将数组转换为列表
            'followers_count': user.followers_count,
            'friends_count': user.friends_count,
            'platform': user.platform
        }
        user_features.append(user_dict)
    for post in ddqn_interface.post_feature:
        post_dict = {
            'userid': post.userid,
            'relevant_user_id': post.relevant_user_id,
            'publish_time': post.publish_time
        }
        post_features.append(post_dict)

    if dataset_name == 'sansuo':
        graph_path = dataPath.data_zjnu_directory / Path('temp_graph.csv')
        node_feature_path = dataPath.data_zjnu_directory / Path('temp_properties.json')

        with open(node_feature_path, 'w') as json_file:
            json.dump(user_features, json_file, indent=4)

        # 将关联关系构成的图保存为 csv 文件
        with open(graph_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for post in post_features:
                if post['userid'] and post['relevant_user_id']:  # 检查 userid 和 relevant_user_id 是否为空
                    try:
                        csv_writer.writerow([int(post['userid']), int(post['relevant_user_id'])])
                    except ValueError as e:
                        print(f"Skipping invalid data: {post['userid']}, {post['relevant_user_id']} - {e}")
    
    elif dataset_name =='meiya':
        graph_path = dataPath.data_zjnu_directory / Path('graph.csv')
        node_feature_path = dataPath.data_zjnu_directory / Path('user_properties.json')
    return budget, graph_path, node_feature_path