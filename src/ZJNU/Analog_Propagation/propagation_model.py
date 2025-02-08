import argparse
import random
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

#


class ICResult:
    def __init__(
        self,
        P_S=None,
        P_I1=None,
        P_I2=None,
        P_R=None,
        activation_paths_info1=None,
        activation_paths_info2=None,
        step_activations_info1=None,
        step_activations_info2=None,
    ):
        """
        数据结构初始化，存储传播过程中的各项数据

        :param P_S: list[int]，易感状态节点的数量
        :param P_I1: list[int]，I1状态节点的累计数量
        :param P_I2: list[int]，I2状态节点的累计数量
        :param P_R: list[int]，免疫状态节点的累计数量
        :param activation_paths_info1: list[str]，记录I1传播的激活路径
        :param activation_paths_info2: list[str]，记录I2传播的激活路径
        :param step_activations_info1: list[str]，记录每步I1状态节点的激活信息
        :param step_activations_info2: list[str]，记录每步I2状态节点的激活信息
        """
        self.P_S = P_S
        self.P_I1 = P_I1
        self.P_I2 = P_I2
        self.P_R = P_R
        self.activation_paths_info1 = activation_paths_info1
        self.activation_paths_info2 = activation_paths_info2
        self.step_activations_info1 = step_activations_info1
        self.step_activations_info2 = step_activations_info2

    @staticmethod
    # 构建关注关系网络
    def build_user_network1(account_info_list):
        G = nx.DiGraph()  # 使用有向图来表示关注关系

        # 遍历每个用户的账号信息
        for account_info in account_info_list:
            # 将每个用户作为一个节点加入图中
            G.add_node(account_info.account_id)

            # 遍历该用户的粉丝列表，建立用户与粉丝之间的有向边（表示关注关系）
            for follower in account_info.followers:
                # 添加边：粉丝关注用户
                G.add_edge(
                    account_info.account_id, follower.account_id
                )  # follower <- user

            G.nodes[account_info.account_id][
                "w"
            ] = account_info.retweet_neg_probability  # 根据账号的 retweet_probability 设置权重

        return G

    @staticmethod
    # 构建关注关系网络
    def build_user_network2(account_info_list):
        G = nx.DiGraph()  # 使用有向图来表示关注关系

        # 遍历每个用户的账号信息
        for account_info in account_info_list:
            # 将每个用户作为一个节点加入图中
            G.add_node(account_info.account_id)

            # 遍历该用户的粉丝列表，建立用户与粉丝之间的有向边（表示关注关系）
            for follower in account_info.followers:
                # 添加边：粉丝关注用户
                G.add_edge(
                    account_info.account_id, follower.account_id
                )  # follower <- user

            G.nodes[account_info.account_id][
                "w"
            ] = account_info.retweet_pos_probability  # 根据账号的 retweet_probability 设置权重

        return G

    @classmethod
    def IC1(cls, selected_id_nodes, account_info_list, max_iter_num):
        """
        运行IC模型的传播模拟
        :param selected_id_nodes: 种子节点文件
        :param account_info_list: 用户信息文件
        :param max_iter_num: 最大传播轮数
        :return: ICResult 对象
        """
        activation_paths = []
        activation_steps = []
        G = cls.build_user_network1(account_info_list)

        for node in G:
            G.nodes[node]["state"] = 0

        seed_nodes = selected_id_nodes
        for seed in seed_nodes:
            G.nodes[seed]["state"] = 1

        all_active_nodes = seed_nodes[:]
        start_influence_nodes = seed_nodes[:]
        count_activated = len(seed_nodes)
        state0_count = [len(G.nodes)]
        state2count = 0
        state2_count = [0]
        activated_count = [count_activated]
        simulation_activation_steps = []

        for i in range(max_iter_num):
            new_active = []
            print(f"Time step {i}: {len(all_active_nodes)} active nodes")

            for v in start_influence_nodes:
                for nbr in G.neighbors(v):
                    if G.nodes[nbr]["state"] == 0:
                        if random.uniform(0, 1) < G.nodes[nbr]["w"]:
                            G.nodes[nbr]["state"] = 1
                            new_active.append(nbr)
                            count_activated += 1
                            activation_paths.append(f"{v} {nbr}")
                        else:
                            G.nodes[nbr]["state"] = 2
                            state2count += 1

            simulation_activation_steps.append(new_active)
            activated_count.append(count_activated)
            start_influence_nodes = new_active[:]
            all_active_nodes.extend(new_active)
            state_0_count = sum(1 for node in G.nodes if G.nodes[node]["state"] == 0)
            state0_count.append(state_0_count)
            state2_count.append(state2count)

            activation_steps.append(" ".join(map(str, new_active)))

        # activation_paths.append("\n")
        # activation_steps.extend(simulation_activation_steps)

        print(f"易感态数：{state0_count}")
        print(f"激活态数：{activated_count}")
        print(f"免疫态数：{state2_count}")
        print(f"激活路径：{activation_paths}")
        print(f"激活步数：{activation_steps}")

        return cls(
            P_S=state0_count,
            P_I1=activated_count,
            P_R=state2_count,
            activation_paths_info1=activation_paths,
            step_activations_info1=activation_steps,
        )

    @classmethod
    def IC2(cls, selected_id_nodes, account_info_list, max_iter_num):
        """
        运行IC模型的传播模拟
        :param edges_file: 边文件
        :param weights_file: 权重文件
        :param max_iter_num: 最大传播轮数
        :return: ICResult 对象
        """
        activation_paths = []
        activation_steps = []
        G = cls.build_user_network2(account_info_list)

        for node in G:
            G.nodes[node]["state"] = 0

        seed_nodes = selected_id_nodes
        for seed in seed_nodes:
            G.nodes[seed]["state"] = 1

        all_active_nodes = seed_nodes[:]
        start_influence_nodes = seed_nodes[:]
        count_activated = len(seed_nodes)
        state0_count = [len(G.nodes)]
        state2count = 0
        state2_count = [0]
        activated_count = [count_activated]
        simulation_activation_steps = []

        for i in range(max_iter_num):
            new_active = []
            print(f"Time step {i}: {len(all_active_nodes)} active nodes")

            for v in start_influence_nodes:
                for nbr in G.neighbors(v):
                    if G.nodes[nbr]["state"] == 0:
                        if random.uniform(0, 1) < G.nodes[nbr]["w"]:
                            G.nodes[nbr]["state"] = 1
                            new_active.append(nbr)
                            count_activated += 1
                            activation_paths.append(f"{v} {nbr}")
                        else:
                            G.nodes[nbr]["state"] = 2
                            state2count += 1

            simulation_activation_steps.append(new_active)
            activated_count.append(count_activated)
            start_influence_nodes = new_active[:]
            all_active_nodes.extend(new_active)
            state_0_count = sum(1 for node in G.nodes if G.nodes[node]["state"] == 0)
            state0_count.append(state_0_count)
            state2_count.append(state2count)

            activation_steps.append(" ".join(map(str, new_active)))

        # activation_paths.append("\n")
        # activation_steps.extend(simulation_activation_steps)

        print(f"易感态数：{state0_count}")
        print(f"激活态数：{activated_count}")
        print(f"免疫态数：{state2_count}")
        print(f"激活路径：{activation_paths}")
        print(f"激活步数：{activation_steps}")

        return cls(
            P_S=state0_count,
            P_I2=activated_count,
            P_R=state2_count,
            activation_paths_info2=activation_paths,
            step_activations_info2=activation_steps,
        )

    @staticmethod
    def build_user_network3(account_info_list):
        G = nx.DiGraph()  # 使用有向图来表示关注关系

        # 遍历每个用户的账号信息
        for account_info in account_info_list:
            # 将每个用户作为一个节点加入图中
            G.add_node(account_info.account_id)

            # 遍历该用户的粉丝列表，建立用户与粉丝之间的有向边（表示关注关系）
            for follower in account_info.followers:
                # 添加边：粉丝关注用户
                G.add_edge(
                    account_info.account_id, follower.account_id
                )  # follower <- user

            # 3. 为每个节点添加权重（retweet_probability）
            G.nodes[account_info.account_id][
                "w1"
            ] = account_info.retweet_neg_probability  # 根据账号的 retweet_probability 设置权重
            G.nodes[account_info.account_id][
                "w2"
            ] = account_info.retweet_pos_probability  # 根据账号的 retweet_probability 设置权重

        return G

    @classmethod
    def IC_vs(cls, selected_id_nodes, account_info_list, max_iter_num):
        # 加载网络
        G = cls.build_user_network3(account_info_list)
        n_nodes = G.number_of_nodes()

        a = 5  # I1初始节点数
        b = 3  # I2初始节点数

        # 初始化状态
        for node in G.nodes:
            G.nodes[node]["status"] = "S"

        I1, I2, I1_ = [], [], []
        S1 = []

        # 随机选5个节点设为I2
        initial_I2_nodes = selected_id_nodes
        for node in initial_I2_nodes:
            G.nodes[node]["status"] = "I2"
            I2.append(node)

        # 随机选3个节点设为I2
        remaining_S_nodes = [node for node in G.nodes if G.nodes[node]["status"] == "S"]
        initial_I1_nodes = random.sample(remaining_S_nodes, b)
        for node in initial_I1_nodes:
            G.nodes[node]["status"] = "I1"
            I1.append(node)
            I1_.append(node)

        count_S = n_nodes - a - b
        count_I1 = a
        count_I2 = b
        count_R = 0

        P_S = [count_S]
        P_I1 = [count_I1]
        P_I2 = [count_I2]
        P_R = [count_R]

        activation_paths_info1 = []
        activation_paths_info2 = []
        step_activations_info1 = []
        step_activations_info2 = []

        # 单次传播过程
        for _ in range(max_iter_num):
            new_I1, new_I2 = [], []
            current_step_info1 = []
            current_step_info2 = []

            # 遍历S1中的每个节点
            for node in S1[:]:
                neighbors = list(G.neighbors(node))
                I2_neighbors = [n for n in neighbors if n in I2]
                if I2_neighbors:
                    S1.remove(node)
                    count_S -= 1
                    if G.nodes[node]["w2"] > random.random():
                        G.nodes[node]["status"] = "I2"
                        count_I2 += 1
                        new_I2.append(node)
                        current_step_info2.append(node)
                        for src in I2_neighbors:
                            activation_paths_info2.append(f"{src} {node}")
                    else:
                        G.nodes[node]["status"] = "R"
                        count_R += 1

            # 判断状态为S且不在S1和S2中的节点
            for node in G.nodes:
                if G.nodes[node]["status"] == "S" and node not in S1:
                    neighbors = list(G.neighbors(node))
                    I1_neighbors = [n for n in neighbors if n in I1]
                    I2_neighbors = [n for n in neighbors if n in I2]

                    if I1_neighbors and I2_neighbors:  # 同时存在I1和I2邻居
                        count_S -= 1
                        if G.nodes[node]["w2"] > random.random():
                            G.nodes[node]["status"] = "I2"
                            count_I2 += 1
                            new_I2.append(node)
                            current_step_info2.append(node)
                            for src in I2_neighbors:
                                activation_paths_info2.append(f"{src} {node}")
                        else:
                            G.nodes[node]["status"] = "R"
                            count_R += 1

                    elif I1_neighbors:  # 只存在I1邻居
                        if G.nodes[node]["w1"] > random.random():
                            G.nodes[node]["status"] = "I1"
                            count_S -= 1
                            count_I1 += 1
                            new_I1.append(node)
                            I1_.append(node)
                            current_step_info1.append(node)
                            for src in I1_neighbors:
                                activation_paths_info1.append(f"{src} {node}")
                        else:
                            S1.append(node)

                    elif I2_neighbors:  # 只存在I2邻居
                        count_S -= 1
                        if G.nodes[node]["w2"] > random.random():
                            G.nodes[node]["status"] = "I2"
                            count_I2 += 1
                            new_I2.append(node)
                            current_step_info2.append(node)
                            for src in I2_neighbors:
                                activation_paths_info2.append(f"{src} {node}")
                        else:
                            G.nodes[node]["status"] = "R"
                            count_R += 1

            for node in I1_:
                if node not in new_I1:
                    neighbors = list(G.neighbors(node))
                    I2_neighbors = [n for n in neighbors if n in I2]

                    if I2_neighbors:
                        I1_.remove(node)
                        if G.nodes[node]["w2"] > random.random():
                            G.nodes[node]["status"] = "I2"
                            count_I2 += 1
                            count_I1 -= 1
                            new_I2.append(node)
                            current_step_info2.append(node)
                            for src in I2_neighbors:
                                activation_paths_info2.append(f"{src} {node}")

            I1 = new_I1
            I2 = new_I2
            P_S.append(count_S)
            P_I1.append(count_I1)
            P_I2.append(count_I2)
            P_R.append(count_R)

            # 记录当前步骤的激活节点
            step_activations_info1.append(" ".join(map(str, current_step_info1)))
            step_activations_info2.append(" ".join(map(str, current_step_info2)))

        # 打包数据到ICResult对象
        # result = ICResult(P_S, P_I1, P_I2, P_R, activation_paths_info1, activation_paths_info2, step_activations_info1,step_activations_info2)

        print(f"易感数：{P_S}")
        print(f"负感染数：{P_I1}")
        print(f"正感染数：{P_I2}")
        print(f"免疫数：{P_R}")
        print(f"负激活路径：{activation_paths_info1}")
        print(f"正激活路径：{activation_paths_info2}")
        print(f"负激活步数：{step_activations_info1}")
        print(f"正激活步数：{step_activations_info2}")

        return cls(
            P_S,
            P_I1,
            P_I2,
            P_R,
            activation_paths_info1,
            activation_paths_info2,
            step_activations_info1,
            step_activations_info2,
        )

    def simulation_all(self, selected_id_nodes, account_info_list, flag):
        if flag == 1:
            result = ICResult.IC2(selected_id_nodes, account_info_list, max_iter_num=20)
        elif flag == 2:
            result = ICResult.IC1(selected_id_nodes, account_info_list, max_iter_num=20)
        elif flag == 3:
            result = ICResult.IC_vs(
                selected_id_nodes, account_info_list, max_iter_num=20
            )
        else:
            raise ValueError("Invalid flag value. Choose either 1 or 2 or 3.")

        return result


if __name__ == "__main__":
    from Analog_Propagation.test_data import account_info_list

    ic_result = ICResult()
    selected_id_nodes = ["user1"]

    result = ic_result.simulation_all(selected_id_nodes, account_info_list, flag=1)
    print("P_S:", result.P_S)
    print("P_I1:", result.P_I1)
    print("P_I2:", result.P_I2)
    print("P_R:", result.P_R)
    print("Activation Paths Info 1:", result.activation_paths_info1)
    print("Step Activations Info 1:", result.step_activations_info1)
    print("Activation Paths Info 2:", result.activation_paths_info2)
    print("Step Activations Info 2:", result.step_activations_info2)
