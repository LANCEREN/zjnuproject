import networkx as nx
import numpy as np
import torch

from src.ZJNU.data_structure import AccountInfo, PostInfo


class DataProcess:
    def __init__(self):
        pass

    def read_influence_features(self, file_path):
        data = np.load(file_path)
        # 获取第二列（假设数据是二维的）
        second_column = data[:, 1]
        # print(second_column)
        return torch.tensor(second_column, dtype=torch.float)

    def label_based_retweet(self, post_info_list, graph):
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
    def construct_attnetwork(self, account_info_list):
        G = nx.DiGraph()  # 使用有向图来表示关注关系

        # 遍历每个用户的账号信息
        for account_info in account_info_list:
            # 将每个用户作为一个节点加入图中
            G.add_node(account_info.account_id)

            # 遍历该用户的粉丝列表，建立用户与粉丝之间的有向边（表示关注关系）
            for follower in account_info.followers:
                # 添加边：粉丝关注用户
                G.add_edge(
                    follower.account_id, account_info.account_id
                )  # follower -> user

        return G

    def construct_node_features(self, account_info_list, node_mapping):
        node_features = [0] * len(node_mapping)
        for account in account_info_list:
            # 获取每个账号的映射索引
            account_index = node_mapping[account.account_id]
            # print(account_index, account.account_id)
            # 提取user_feature中的特征
            user_feature = account.user_feature
            # 拼接user_feature中的特征（例如：gender, verified, contents_count等）和影响力
            feature = torch.tensor(
                [
                    user_feature.gender,  # 性别
                    user_feature.verified,  # 认证状态
                    user_feature.ip,  # IP地址
                    user_feature.contents_count,  # 发布内容数
                    user_feature.friends_count,  # 好友数
                    user_feature.followers_count,  # 粉丝数
                    user_feature.platform,  # 平台标识符
                    account.influence,  # 影响力
                ],
                dtype=torch.float,
            )

            # 拼接用户嵌入向量
            feature = torch.cat((feature, account.user_embeddings), dim=0)

            # 将该特征放入相应索引位置
            node_features[account_index] = feature

        # 转换为Tensor
        return torch.stack(node_features)
