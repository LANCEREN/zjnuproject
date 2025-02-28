import json
import os
import random
import statistics
import time
import warnings
from collections import deque, namedtuple
from itertools import count
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max
from tqdm import tqdm

from src.SJTU.DDQN import data_process_utils, graph_utils, reward_fun
from src.SJTU.DDQN.config import dataPath
from src.SJTU.DDQN.model import GNN_DDQN, node_embed


class MyTrainer:
    def __init__(self, device, training, T, lr, epoch, budget, data_dir, output_dir):
        self.device = device
        self.memory_size = 50000
        self.reg_hidden = 32
        self.embed_dim = 64
        self.lr = lr
        self.bs = 16
        self.n_step = 1
        self.model_file = "GNN_DQN.ckpt"

        # 以下是重要参数
        self.training = training
        self.T = T
        self.epoch = epoch
        self.budget = budget

        # 数据加载和各种创建应该在训练器开始时就定下来
        save_path = output_dir

        #    划分训练和测试数据
        graph_path = os.path.join(data_dir, "graph.txt")
        node_feature_path = os.path.join(data_dir, "node_features.json")
        graph_train = graph_utils.read_graph(
            graph_path, node_feature_path, ind=0, directed=True
        )
        graph_test = graph_train
        # graph_train , graph_test = process_graph(data_path ,path_dir, 0.8)

        self.model_file = os.path.join(save_path, self.model_file)
        self.reward_file = os.path.join(save_path, "reward.txt")
        self.embed_file = os.path.join(save_path, "embed_model.ckpt")

        self.train_env = Environment(
            graph_train,
            self.budget,
            method="RR",
            use_cache=True,
            training=self.training,
        )
        self.test_env = Environment(
            graph_test,
            self.budget,
            method="MC",
            use_cache=True,
            training=not (self.training),
        )
        self.agent = Agent(
            self.device,
            self.reg_hidden,
            self.embed_dim,
            self.training,
            self.T,
            self.n_step,
            self.lr,
            self.bs,
            memory_size=self.memory_size,
        )

    # 下面讲基于Runner和Environment重构train，这也是核心，
    # 使得之后调用此train方法可以直接返回样本数目、奖励与种子节点和模型参数

    def My_train(self):
        my_runner = Runner(self.train_env, self.test_env, self.agent, self.training)
        if self.training:
            sample_size, clf_para, last_rewards = my_runner.train(
                self.epoch,
                self.model_file,
                self.reward_file,
                embed_file=self.embed_file,
            )
        else:
            sample_size, clf_para = my_runner.test(
                num_trials=10, embed_file=self.embed_file
            )
        return sample_size, clf_para, last_rewards


class Agent:
    def __init__(
        self,
        device,
        reg_hidden,
        embed_dim,
        training,
        T,
        n_step,
        lr,
        batch_size=64,
        memory_size=1000,
    ):
        self.gamma = 0.99
        self.node_dim = 7
        self.edge_dim = 3
        self.batch_size = batch_size  # batch size for experience replay
        self.n_step = n_step  # num of steps to accumulate rewards
        self.training = training
        self.T = T
        self.memory = ReplayMemory(memory_size)
        self.device = device
        self.reg_hidden = reg_hidden
        self.embed_dim = embed_dim
        # store node embeddings of each graph, avoid multiprocess copy
        # 原来这个节点映射只在touplegdd里面用了，在外面根本没有用到，需要参考重新写input以及神经网络循环
        self.graph_node_embed = {}
        self.model = GNN_DDQN(
            reg_hidden=self.reg_hidden,
            embed_dim=self.embed_dim,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            T=self.T,
            w_scale=0.01,
            avg=False,
        ).to(self.device)
        if self.training:
            self.target = GNN_DDQN(
                reg_hidden=self.reg_hidden,
                embed_dim=self.embed_dim,
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                T=self.T,
                w_scale=0.01,
                avg=False,
            ).to(self.device)
            self.target.load_state_dict(self.model.state_dict())
            self.target.eval()
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if not self.training:
            # load pretrained model for testing
            cwd = os.getcwd()
            self.model.load_state_dict(torch.load(os.path.join(cwd, self.model_file)))
            self.model.eval()

    def reset(self):
        """restart"""
        pass

    @torch.no_grad()
    def setup_graph_input(self, graph, states, actions=None, path=None):
        """create a batch data loader from a batch of
            states, # pred all
            states, actions, # pred
            node features (states), edge features
        return a batch from the data loader
        """
        data = []
        i = 0
        if id(graph) not in self.graph_node_embed:
            self.graph_node_embed[id(graph)] = node_embed(
                graph, self.device, self.node_dim, path
            )  # epochs for initial embedding
        with torch.no_grad():
            # 节点特征
            x = self.graph_node_embed[id(graph)].detach().clone()
            # 为每个节点增添新的特征即当前节点的状态
            x = torch.cat((x, states[i].detach().clone().unsqueeze(dim=1)), dim=-1)
            # 通过graphs[i].from_to_edges()获取边的索引，并将其转换为PyTorch张量edge_index
            edge_index = torch.from_numpy(graph.get_edge_index()).contiguous().long()

            # 初始化边特征矩阵，并根据边的权重和节点状态调整其值
            edge_attr = torch.ones(graph.num_edges(), self.edge_dim)
            # 将权重列表转换为tensor张量
            edge_attr[:, 1] = torch.tensor(
                list(graph.edges.values()), dtype=torch.float32
            )
            # 状态加入转换向量
            edge_attr[:, 0] = states[i][edge_index[0]]
            edge_attr[:, 2] = torch.abs(
                states[i][edge_index[0]] - states[i][edge_index[1]]
            )
            y = actions[i].clone() if actions is not None else None
            data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        with torch.no_grad():
            # 使用DataLoader创建一个数据加载器，将data列表中的数据组织成批处理形式，以便于模型训练或推理
            loader = DataLoader(data, batch_size=1, shuffle=False)
            for batch in loader:
                # adjust y if applicable
                if actions is not None:
                    total_num = 0
                    # 计算图中节点的总数
                    total_num += batch.num_nodes
                    batch.y += total_num
                return batch.to(self.device)

    @torch.no_grad()
    def setup_graph_batch_input(self, graph, states, actions=None, path=None):
        """create a batch data loader from a batch of
            states, # pred all
            states, actions, # pred
            node features (states), edge features
        return a batch from the data loader
        """
        data = []
        sample_size = len(graph)
        for i in range(sample_size):
            if id(graph[i]) not in self.graph_node_embed:
                self.graph_node_embed[id(graph[i])] = node_embed(
                    graph[i], self.device, self.node_dim, path
                )  # epochs for initial embedding
            with torch.no_grad():
                # 节点特征
                x = self.graph_node_embed[id(graph[i])].detach().clone()
                # 为每个节点增添新的特征即当前节点的状态
                state_i = states[i].detach().clone().unsqueeze(dim=1).to(self.device)
                x = torch.cat((x, state_i), dim=-1)
                # 通过graphs[i].from_to_edges()获取边的索引，并将其转换为PyTorch张量edge_index
                edge_index = (
                    torch.from_numpy(graph[i].get_edge_index()).contiguous().long()
                )

                # 初始化边特征矩阵，并根据边的权重和节点状态调整其值
                edge_attr = torch.ones(graph[i].num_edges(), self.edge_dim)
                # 将权重列表转换为tensor张量
                edge_attr[:, 1] = torch.tensor(
                    list(graph[i].edges.values()), dtype=torch.float32
                )
                # 状态加入转换向量
                edge_attr[:, 0] = states[i][edge_index[0]]
                edge_attr[:, 2] = torch.abs(
                    states[i][edge_index[0]] - states[i][edge_index[1]]
                )
                y = actions[i].clone() if actions is not None else None
                data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        # 使用DataLoader创建一个数据加载器，将data列表中的数据组织成批处理形式，以便于模型训练或推理
        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            # adjust y if applicable
            if actions is not None:
                total_num = 0
                # 计算图中节点的总数
                for i in range(1, sample_size):
                    total_num += batch[i - 1].num_nodes
                    batch[i].y += total_num
            return batch.to(self.device)

    # select_action 的函数，它基于当前状态选择一个动作。这个函数适用于强化学习中的代理，特别是在图形结构的数据上进行操作。
    # 函数的参数包括图形(graph)、当前状态(state)、探索率(epsilon)、是否处于训练模式(training)以及可选的预算限制(budget)
    # 这个函数体现了强化学习中的两个关键策略：探索（通过随机选择动作）和利用（通过选择Q值最高的动作）。
    # 探索允许代理发现新的、可能更好的策略，而利用则使代理能够利用已知的最佳策略。通过调整epsilon值，可以控制探索与利用之间的平衡
    def select_action(
        self, graph, state, epsilon, training=True, budget=None, path=None
    ):
        """act upon state"""
        # 函数首先将图形和状态转换为模型可以处理的输入格式。然后，使用模型预测每个动作的Q值，但是将当前状态已选择的动作的Q值设为一个非常低的值(-1e5)，以避免重复选择。
        # 如果没有提供预算(budget)，函数将返回具有最高Q值的动作。如果提供了预算，函数将返回Q值最高的budget个动作
        if not (training):
            graph_input = self.setup_graph_input(
                graph, state.unsqueeze(dim=0), path=path
            )
            with torch.no_grad():
                q_a = self.model(graph_input)
            q_a[state.nonzero()] = -1e5

            if budget is None:
                return torch.argmax(q_a).detach().clone()
            else:  # return all seed nodes within budget at one time
                return torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone()

        # training
        available = (state == 0).nonzero()
        if epsilon > random.random():
            return random.choice(available)
        # 在训练模式下，函数首先找出所有可用的动作（即状态值为0的动作）。然后，根据探索率(epsilon)随机选择一个可用的动作或者选择具有最高Q值的动作。为了找到最高Q值的动作，函数再次生成模型的输入，预测Q值，然后从可用动作中选择Q值最高的动作。如果有多个动作具有相同的最高Q值，它将随机选择一个。
        # 为了实现这一点，函数使用了numpy.intersect1d方法来找出既在可用动作中又在具有最高Q值的动作中的动作，然后随机选择其中一个。
        else:
            # print("state:",state.size())
            graph_input = self.setup_graph_input(
                graph, state.unsqueeze(dim=0), path=path
            )
            with torch.no_grad():
                q_a = self.model(graph_input)
            max_position = (q_a == q_a[available].max().item()).nonzero()
            return torch.tensor(
                [
                    random.choice(
                        np.intersect1d(
                            available.cpu().contiguous().view(-1).numpy(),
                            max_position.cpu().contiguous().view(-1).numpy(),
                        )
                    )
                ],
                dtype=torch.long,
            )

    # 具体来说，它实现了一个名为 memorize 的方法，
    # 用于将环境（env）中的状态、动作和奖励信息添加到所谓的 n步回放记忆（n-step replay memory）中。这种记忆机制是为了增加学习的稳定性。
    # 总的来说，这段代码通过计算累积奖励并将状态、动作和奖励的信息以特定的格式存储，为使用深度学习方法进行强化学习提供了数据准备的基础。
    # 这种n步回放记忆机制有助于提高学习过程的稳定性和效率。
    def memorize(self, env):
        """n step for stability"""
        # access state list, reward list and action list from env
        # to add to n step replay memory
        sum_rewards = [0.0]
        for reward in reversed(env.rewards):
            # normalize reward by number of nodes
            reward /= env.graph.num_nodes()
            sum_rewards.append(reward + self.gamma * sum_rewards[-1])
        sum_rewards = sum_rewards[::-1]

        for i in range(len(env.states)):
            if i + self.n_step < len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long),
                    torch.tensor([env.actions[i]], dtype=torch.long),
                    torch.tensor(env.states[i + self.n_step], dtype=torch.long),
                    torch.tensor(
                        [
                            sum_rewards[i]
                            - (self.gamma**self.n_step) * sum_rewards[i + self.n_step]
                        ],
                        dtype=torch.float,
                    ),
                    env.graph,
                )
            elif i + self.n_step == len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long),
                    torch.tensor([env.actions[i]], dtype=torch.long),
                    None,
                    torch.tensor([sum_rewards[i]], dtype=torch.float),
                    env.graph,
                )

    # 小回放
    def replay(self, path):
        """fit on a batch sampled from replay memory"""
        # optimize model
        sample_size = (
            self.batch_size if len(self.memory) >= self.batch_size else len(self.memory)
        )
        # need to fix dimension and restrict action space
        transitions = self.memory.sample(sample_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool,
            device=self.device,
        )

        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states_graphs = [
            batch.graph[i] for i, s in enumerate(batch.next_state) if s is not None
        ]

        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # graph batch for setting up training batch
        graph_batch = batch.graph

        state_action_values = self.model(
            self.setup_graph_batch_input(
                graph_batch, state_batch, action_batch, path=path
            )
        ).squeeze(dim=1)
        next_state_values = torch.zeros(sample_size, device=self.device)

        if len(non_final_next_states) > 0:
            batch_non_final = self.setup_graph_batch_input(
                non_final_next_states_graphs, non_final_next_states, path=path
            )
            next_state_values[non_final_mask] = (
                scatter_max(
                    self.target(batch_non_final)
                    .squeeze(dim=1)
                    .add_(torch.cat(non_final_next_states).to(self.device) * (-1e5)),
                    batch_non_final.batch,
                )[0]
                .clamp_(min=0)
                .detach()
            )
        # 根据贝尔曼方程计算了期望的状态-动作值，并使用这个期望值和实际的状态-动作值来计算损失。然后，通过反向传播更新模型的权重。
        expected_state_action_values = (
            next_state_values * self.gamma**self.n_step + reward_batch.to(self.device)
        )

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # if double dqn, update target network if needed
        self.target.load_state_dict(self.model.state_dict())
        return True

    def update_target_net(self):
        # if double dqn, update target network if needed
        self.target.load_state_dict(self.model.state_dict())
        return True

    # 函数用于保存模型的参数。它将模型的参数保存到当前工作目录下的一个文件中。
    def save_model(self, file_name):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), os.path.join(cwd, file_name))


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "graph")
)


class ReplayMemory(object):
    """random replay memory"""

    def __init__(self, capacity):
        # temporily save 1-step snapshot
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Runner:
    """run an agent in an environment"""

    def __init__(self, train_env, test_env, agent, training):
        # 训练环境
        self.train_env = train_env
        # 测试环境
        self.test_env = test_env  # environment for testing
        self.agent = agent
        self.training = training

    # play_game方法是类的核心，它允许智能体在环境中进行一定次数的迭代。这个方法可以用于训练和测试，支持时间统计和一次性生成种子集（seed set）。
    # 在训练模式下，智能体会根据环境状态选择动作，并根据动作的结果更新其知识。在测试模式下，可以选择一次性生成动作序列，以评估智能体的性能。
    def play_game(
        self,
        num_iterations,
        epsilon,
        training=True,
        time_usage=False,
        one_time=False,
        path=None,
    ):
        """play the game num_iterations times
        Arguments:
            time_usage: off if False; True: print average time usage for seed set generation
            one_time: generate the seed set at once without regenerating embeddings
        """
        c_rewards = []
        im_seeds = []

        if time_usage:
            total_time = 0.0  # total time for all iterations on all testing graphs

        for iteration in range(num_iterations):
            # handle multiple graphs for evaluation during training
            train_rewards = 0.0
            if training:
                self.env.reset()
                for i in count():
                    # 张量转换 self.env.state的长度等于图中节点的数量
                    state = torch.tensor(self.env.state, dtype=torch.long).to(
                        self.agent.device
                    )
                    action = self.agent.select_action(
                        self.env.graph,
                        state,
                        epsilon,
                        training=self.training,
                        path=path,
                    ).item()
                    reward, done = self.env.step(action)
                    train_rewards += reward
                    # this game is over
                    if done:
                        # memorize the trajectory
                        self.agent.memorize(self.env)
                        c_rewards.append(reward)
                        im_seeds.append(self.env.actions)
                        break
            else:
                if id(self.env.graph not in self.agent.graph_node_embed):
                    self.agent.graph_node_embed[id(self.env.graph)] = node_embed(
                        self.env.graph, self.agent.device, 7, path
                    )  # 20 = epochs for initial embedding , 7 = node_dim

                if time_usage:
                    start_time = time.time()
                    time_reward = [
                        0.0
                    ]  # time of calculating reward, needs to be subtracted
                else:
                    time_reward = None

                self.env.reset(training=training)
                if one_time:
                    state = torch.tensor(self.env.state, dtype=torch.long).to(
                        self.agent.device
                    )
                    actions = self.agent.select_action(
                        self.env.graph,
                        state,
                        epsilon,
                        training=training,
                        budget=self.env.budget,
                        path=path,
                    ).tolist()

                    # no sort of actions selected
                    im_seeds.append(actions)

                    if time_usage:
                        total_time += time.time() - start_time

                    final_reward = self.env.compute_reward(actions)
                    c_rewards.append(final_reward)

                else:
                    for i in count():
                        state = torch.tensor(self.env.state, dtype=torch.long).to(
                            self.agent.device
                        )
                        action = self.agent.select_action(
                            self.env.graph, state, epsilon, training=training, path=path
                        ).item()
                        final_reward, done = self.env.step(action, time_reward)
                        # this game is over
                        if done:
                            # no sort of action selected
                            im_seeds.append(self.env.actions)
                            c_rewards.append(final_reward)
                            break
                    if time_usage:
                        total_time += time.time() - start_time - time_reward[0]
        if time_usage:
            print(
                f"Seed set generation per iteration time usage is: {total_time/num_iterations:.2f} seconds"
            )
        if training:
            return train_rewards, im_seeds
        else:
            return c_rewards, im_seeds

    # train方法用于训练智能体。它首先进行预训练，然后在多个周期内调整智能体的行为策略，逐渐减少探索率并定期保存模型和更新目标网络。
    def train(self, num_epoch, model_file, reward_file, embed_file):
        """let agent act and learn from the environment"""
        # pretrain 预训练效果不够好
        # tqdm.write('Pretraining:')
        # self.play_game(1000, 1.0)
        if self.training:
            self.env = self.train_env
        else:
            self.env = self.test_env
        eps_start = 1.0
        eps_end = 0.05
        eps_step = num_epoch
        train_rewards = []
        # train
        tqdm.write("Start training:")
        progress_fitting = tqdm(total=num_epoch)
        for epoch in range(num_epoch):
            eps = eps_end + max(
                0.0, (eps_start - eps_end) * (eps_step - epoch) / eps_step
            )
            # eps = 0.1
            reward, _ = self.play_game(1, eps, path=embed_file)
            train_rewards.append(reward)
            # if epoch % 10 == 0:
            #     reward , _ = self.play_game(10, eps,path=embed_file)
            #     train_rewards.append(np.mean(reward))
            if epoch % 10 == 0:
                # test,牛逼，想要在此测试，测试集应该远大于训练集
                rewards, seeds = self.play_game(1, 0.0, training=False, path=embed_file)
                # tqdm.write(f'{epoch}/{num_epoch}: ({str(seeds[0])[1:-1]}) | {rewards[0]}')

            # if epoch % 10 == 0:
            #     # save model
            #     self.agent.save_model(model_file + str(epoch))

            if epoch % 10 == 0:
                self.agent.update_target_net()
            # train the model
            self.agent.replay(embed_file)
            progress_fitting.update(1)

        # 打开一个文件用于写入，如果文件不存在则创建
        with open(reward_file, "w") as file:
            # 遍历列表中的每个元素
            for reward in train_rewards:
                # 将每个元素转换为字符串，添加换行符，然后写入文件
                file.write(f"{reward}\n")

        # show test results after training
        rewards, seeds = self.play_game(1, 0.0, training=False, path=embed_file)
        # tqdm.write(f'{num_epoch}/{num_epoch}: ({str(seeds[0])[1:-1]}) | {rewards[0]}')
        self.agent.save_model(model_file)
        sample_size = (self.env.graph.num_nodes(), self.env.graph.num_edges())
        clf_para = self.agent.model.state_dict()
        last_reward = train_rewards[-1]
        return sample_size, clf_para, seeds[0]

    def test(self, num_trials, embed_file):
        """let agent act in the environment
        num_trials: may need multiple trials to get average
        """
        print("Generate seeds at one time:", flush=True)
        all_rewards, all_seeds = self.play_game(
            num_trials, 0.0, False, time_usage=True, one_time=True, path=embed_file
        )
        print(f"Number of trials: {num_trials}")
        print(f'Graph path: {", ".join(g.path_graph for g in self.env.graph)}')
        cnt = 0
        for a_r, a_s in zip(all_rewards, all_seeds):
            print(f"Seeds: {a_s} | Reward: {a_r}")
            if len(self.env.graph) > 1:
                cnt += 1
                if cnt == len(self.env.graph):
                    print("")
                    cnt = 0

        print("Generate seed one by one:", flush=True)
        all_rewards, all_seeds = self.play_game(
            num_trials, 0.0, False, time_usage=True, one_time=False, path=embed_file
        )
        print(f"Number of trials: {num_trials}")
        print(f'Graph path: {", ".join(g.path_graph for g in self.env.graph)}')
        cnt = 0
        for a_r, a_s in zip(all_rewards, all_seeds):
            print(f"Seeds: {a_s} | Reward: {a_r}")
            if len(self.env.graph) > 1:
                cnt += 1
                if cnt == len(self.env.graph):
                    print("")
                    cnt = 0


class Environment:
    """environment that the agents run in"""

    def __init__(self, graph, budget, method="RR", use_cache=False, training=True):
        """
        method: 'RR' or 'MC'
        use_cache: use cache to speed up
        """
        # sampled set of graphs
        self.graph = graph
        # IM
        self.budget = budget
        self.method = method
        # useful only if run on the same graph multiple times
        self.use_cache = use_cache
        if self.use_cache:
            if self.method == "MC":
                # this may be not needed by cached RR
                # not used for RR
                self.influences = {}  # cache source set to influence value mapping
            elif self.method == "RR":
                self.RRs_dict = {}
        self.training = training  # whether in training mode or testing mode

    def reset_graphs(self, num_graphs=10):
        # generate new graph
        raise NotImplementedError()

    # reset方法用于重置环境。如果没有提供索引，它会随机选择一个图作为当前图。然后，它会初始化状态为全零的列表，并设置prev_inf（前一个影响力得分）为0。
    # 如果使用缓存并且方法为'RR'，它还会从RRs_dict中获取或创建一个空列表来存储RR集。最后，它会初始化states、actions和rewards列表，并设置训练模式。

    def reset(self, training=True):
        """restart"""
        self.state = [0 for _ in range(self.graph.num_nodes())]
        # IM
        self.prev_inf = 0  # previous influence score
        # store RR sets in case there are more than one graph
        if self.use_cache and self.method == "RR":
            self.RRs = self.RRs_dict.setdefault(id(self.graph), [])
        self.states = []
        self.actions = []
        self.rewards = []
        self.training = training

    # '''
    # compute_reward用于计算给定行动集S的奖励。它首先检查是否需要计算影响力值。如果使用缓存并且方法为'MC'，它会检查influences字典中是否已经有了对应S的影响力值。
    # 它使用MC或RR方法计算预期影响,并返回奖励。计算得到的影响力值会被存储在influences或RRs中，以便后续使用。最后，奖励被计算为当前影响力值与前一个影响力值的差，并被添加到rewards列表中。
    # '''
    def compute_reward(self, S):
        num_process = 5  # number of parallel processes
        num_trial = 10000  # number of trials
        # fetch influence value
        need_compute = True
        if self.use_cache and self.method == "MC":
            S_str = f"{id(self.graph)}.{','.join(map(str, sorted(S)))}"
            need_compute = S_str not in self.influences

        if need_compute:
            if self.method == "MC":
                with Pool(num_process) as p:
                    es_inf = statistics.mean(
                        p.map(
                            reward_fun.workerMC,
                            [
                                [self.graph, S, int(num_trial / num_process)]
                                for _ in range(num_process)
                            ],
                        )
                    )
            elif self.method == "RR":
                if self.use_cache:
                    # cached without incremental
                    es_inf = reward_fun.computeRR(
                        self.graph, S, num_trial, cache=self.RRs
                    )
                else:
                    es_inf = reward_fun.computeRR(self.graph, S, num_trial)
            else:
                raise NotImplementedError(f"{self.method}")

            if self.use_cache and self.method == "MC":
                self.influences[S_str] = es_inf
        else:
            es_inf = self.influences[S_str]

        reward = es_inf - self.prev_inf
        self.prev_inf = es_inf
        # store reward
        self.rewards.append(reward)
        return reward

    # step方法用于执行一个行动并获取奖励。它首先检查给定的节点是否已经被选择过。如果没有，它会更新状态，将该节点对应的状态设置为1，并计算奖励。如果是训练模式，它会计算奖励；如果是测试模式，它会计算奖励并记录时间。最后，它返回奖励和是否结束的标志。
    def step(self, node, time_reward=None):
        """change state and get reward"""
        # node has already been selected
        if self.state[node] == 1:
            return
        # store state and action
        self.states.append(self.state.copy())
        self.actions.append(node)
        # update state
        self.state[node] = 1
        # calculate reward
        S = self.actions
        # whether game is over, budget is reached
        done = len(S) >= self.budget

        if self.training:
            reward = self.compute_reward(S)
        else:
            if done:
                if time_reward is not None:
                    start_time = time.time()
                reward = self.compute_reward(S)
                if time_reward is not None:
                    time_reward[0] = time.time() - start_time
            else:
                reward = None
        return (reward, done)


# 忽略特定的警告
warnings.filterwarnings(
    "ignore", category=UserWarning, message="This overload of nonzero is deprecated:"
)


def train_IMP():
    # 选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 训练模型
    training = True
    # 训练周期
    T = 50
    # 学习率
    lr = 0.001
    # 训练次数
    epoch = 100

    # 预算，种子节点数目
    budget = 100
    seed_set = set()

    # 数据结果输出路径
    output = dataPath.current_directory /  Path("data") / Path("SJTU") / Path("DDQN") / Path("output")

    # 下面是从外部api接口接收数据并且调用函数返回
    ddqn_interface = data_process_utils.json_preprocess_zjnu(
        budget=budget,
        account_info_list_file_name="account_info_list.json",
        post_info_list_file_name="post_info_list.json",
    )
    budget, graph_path, node_feature_path = graph_utils.process_graph_api(
        ddqn_interface
    )

    # 数据预处理
    graph_dir, sub_num = graph_utils.preprocess_graph(
        graph_path, node_feature_path, dataPath.data_sjtu_directory
    )

    if sub_num == 1:
        # 创建训练器
        save_path = output / Path("weights")
        os.makedirs(save_path, exist_ok=True)
        trainer = MyTrainer(
            device, training, T, lr, epoch, budget, graph_dir, save_path
        )
        # 开始训练，并返回结果
        sample_size, _, seeds = trainer.My_train()
        user_list = graph_utils.user_map(seeds, graph_dir)
        seed_set.update(user_list)
    else:
        # 分配预算
        budget_per_sub = int(budget / sub_num)
        remaining_budget = budget - budget_per_sub * (sub_num - 1)
        for i in range(sub_num):
            if i == sub_num - 1:
                current_budget = remaining_budget
            else:
                current_budget = budget_per_sub
            # 创建训练器
            sub_graph_dir = graph_dir / Path(f"subgraph_{i}")
            save_path = output / Path("weights") / Path(f"subgraph_{i}")
            os.makedirs(save_path, exist_ok=True)
            trainer = MyTrainer(
                device, training, T, lr, epoch, current_budget, sub_graph_dir, save_path
            )
            # 开始训练，并返回结果
            sample_size, _, seeds = trainer.My_train()
            user_list = graph_utils.user_map(seeds, sub_graph_dir)
            seed_set.update(user_list)

    seed_user_id = graph_utils.user_map(seed_set, dataPath.data_sjtu_directory)
    seed_user_id_str = [
        ddqn_interface.user_dict_reverse[str(user_id)] for user_id in seed_user_id
    ]
    seed_user_id_path = output / Path("node_features.json")
    with open(seed_user_id_path, "w") as file:
        json.dump(seed_user_id_str, file)
    print(f"Seed user IDs saved to: {seed_user_id_path}")


if __name__ == "__main__":
    train_IMP()
