import random
import numpy as np
import torch
from torch_geometric.data import Data

from src.ZJNU.Analog_Propagation.Finetuning_GAT.DataProcess import DataProcess
from src.ZJNU.Analog_Propagation.Finetuning_GAT.model import GAT  # 导入模型
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

#from Analog_Propagation11.test_data import account_info_list, post_info_list

warnings.filterwarnings("ignore")


class FinetuningGAT(torch.nn.Module):
    def __init__(self):
        pass

    from sklearn.metrics import f1_score


    def calculate_metrics(self, y_true, y_pred, threshold=0.3):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        pred_bin = (y_pred > threshold).astype(int)
        accuracy = accuracy_score(y_true, pred_bin)
        precision = precision_score(y_true, pred_bin)
        recall = recall_score(y_true, pred_bin)
        f1 = f1_score(y_true, pred_bin)
        auc = roc_auc_score(y_true, y_pred)  # AUC分数
        return accuracy, precision, recall, f1, auc

    def split_masks(self, data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        np.random.seed(seed)
        if data.num_nodes < 20:
            train_mask = np.ones(data.num_nodes, dtype=bool)
            val_mask = np.ones(data.num_nodes, dtype=bool)
            test_mask = np.ones(data.num_nodes, dtype=bool)
        else:
            indices = np.random.permutation(data.num_nodes)
            train_size = int(data.num_nodes * train_ratio)
            val_size = int(data.num_nodes * val_ratio)
            test_size = data.num_nodes - train_size - val_size

            train_mask = np.zeros(data.num_nodes, dtype=bool)
            val_mask = np.zeros(data.num_nodes, dtype=bool)
            test_mask = np.zeros(data.num_nodes, dtype=bool)

            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True

        return train_mask, val_mask, test_mask

    # 训练和测试过程封装成函数
    def train_and_test(self,model,data, train_mask, test_mask,val_mask, optimizer, loss_fn, threshold=0.3):
        optimizer.zero_grad()
        output = model(data)[train_mask]  # 只计算训练集的输出
        loss = loss_fn(output, data.y[train_mask].view(-1, 1))
        loss.backward()
        optimizer.step()
        # 计算训练集指标
        train_pred = (torch.sigmoid(output) > threshold).float()

        train_acc, train_prec, train_rec, train_f1, train_auc = self.calculate_metrics(data.y[train_mask], train_pred,threshold)

        # 测试集上的指标和保存转发概率
        model.eval()
        with torch.no_grad():
            val_output = model(data)[val_mask]
            val_pred = (torch.sigmoid(val_output) > threshold).float()
            val_acc, val_prec, val_rec, val_f1, val_auc = self.calculate_metrics(data.y[val_mask], val_pred, threshold)

            # 测试集评估
            test_output = model(data)[test_mask]
            test_pred = (torch.sigmoid(test_output) > threshold).float()
            test_acc, test_prec, test_rec, test_f1, test_auc = self.calculate_metrics(data.y[test_mask], test_pred,threshold)

            # 输出训练、验证、测试集的指标
        print(
            f"Train Metrics: Acc={train_acc:.4f}, Precision={train_prec:.4f}, Recall={train_rec:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}")
        print(
            f"Val Metrics: Acc={val_acc:.4f}, Precision={val_prec:.4f}, Recall={val_rec:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
        print(
            f"Test Metrics: Acc={test_acc:.4f}, Precision={test_prec:.4f}, Recall={test_rec:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")

        # 输出转发概率并保存
        output_prob = torch.sigmoid(output)
        retweet_probabilities = output_prob.squeeze()  # 得到一维向量
        return retweet_probabilities

    def FinetunningGAT(self,account_info_list,post_info_list):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
        # parser.add_argument('--node_features_file', type=str, default='../zjnuproject-master1/Influence_Prediction/out/test.content',
        #                     help='Path to the node features file')
        # parser.add_argument('--edges_file', type=str, default='../zjnuproject-master1/Feature_Extract/out/微博用户关注关系.txt',
        #                     help='Path to the edges file')
        parser.add_argument('--in_channels', type=int, help='Number of input features')
        parser.add_argument('--out_channels', type=int, help='Number of output channels (number of classes)')
        parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay coefficient')
        parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
        parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of the dataset to include in the test split')
        parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
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
        node_features = data_process.construct_node_features(account_info_list,node_mapping)
        node_features = torch.tensor(node_features, dtype=torch.float32)  # 确保节点特征为float32
        edges = [(node_mapping[u], node_mapping[v]) for u, v in attention_graph.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # 2.设置节点的转发标签
        positive_posts = [post for post in post_info_list if post.sentiment == 1]
        negative_posts = [post for post in post_info_list if post.sentiment == -1]
        # node_labels = data_process.label_based_retweet(post_info_list,attention_graph)
        pos_node_labels = data_process.label_based_retweet(positive_posts, attention_graph)
        neg_node_labels = data_process.label_based_retweet(negative_posts, attention_graph)
        node_labels = data_process.label_based_retweet(post_info_list, attention_graph)
        new_node_labels = {}
        new_posnode_labels = {}
        new_negnode_labels = {}
        for origin_node_id,label in node_labels.items():
            new_node_id = node_mapping[origin_node_id]
            new_node_labels[new_node_id] = label
        sorted_node_labels = dict(sorted(new_node_labels.items()))
        y=torch.tensor(list(sorted_node_labels.values()), dtype=torch.float32)
        # retweet_network = data_process.construct_retnetwork(post_info_list)
        # static_labels = data_process.label_network(attention_graph, retweet_network)
        # y=static_labels
        for origin_node_id,label in pos_node_labels.items():
            new_node_id = node_mapping[origin_node_id]
            new_posnode_labels[new_node_id] = label
        sorted_posnode_labels = dict(sorted(new_posnode_labels.items()))
        y_pos=torch.tensor(list(sorted_posnode_labels.values()), dtype=torch.float32)
        # print(y_pos)
        for origin_node_id,label in neg_node_labels.items():
            new_node_id = node_mapping[origin_node_id]
            new_negnode_labels[new_node_id] = label
        sorted_negnode_labels = dict(sorted(new_negnode_labels.items()))
        y_neg=torch.tensor(list(sorted_negnode_labels.values()), dtype=torch.float32)
        # print(y_neg)
        # influence_vector = data_process.read_influence_features(influence_vector_file)
        # node_features = data_process.concatenate_features_and_influence(x,influence_vector)
        # print(node_features.shape)
        # print(node_features)

        # 构建 PyG 图数据对象
        data1 = Data(x=node_features, edge_index=edge_index, y=y_pos)  # 将标签 y 也传入数据对象
        data2 = Data(x=node_features, edge_index=edge_index, y=y_neg)
        data3 = Data(x=node_features, edge_index=edge_index, y=y)
        data1 = data1.to(device)
        data2 = data2.to(device)
        data3 = data3.to(device)
        # 模型超参数
        in_channels = data1.num_node_features  # 输入特征维度
        out_channels = 1  # 输出维度
        heads = 4  # 多头注意力的头数
        # 保存转发概率的列表
        retweet_prob_lists = []
        # 创建模型

        model = GAT(in_channels=in_channels, out_channels=out_channels, heads=heads).to(device)
        # model = GAT(in_channels=in_channels, out_channels=out_channels, heads=heads)

        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # loss_fn = torch.nn.BCEWithLogitsLoss()  # 二分类问题使用BCEWithLogitsLoss
        # 假设 0 标签占 80%，1 标签占 20%
        pos_weight = torch.tensor([data3.num_nodes / data3.y.sum()]).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        data_sets = [data1, data2,data3]

        # 模型训练和测试过程
        for data in data_sets:
            train_mask, val_mask, test_mask = self.split_masks(data)

            # 模型训练
            model.train()
            for epoch in range(args.num_epochs):
                print(f'Epoch [{epoch + 1}/{args.num_epochs}]')
                retweet_probabilities = self.train_and_test(model,data, train_mask, test_mask, val_mask,optimizer, loss_fn)
                #
                # if epoch % 10 == 0:
                #     model.eval()
                #     with torch.no_grad():
                #         train_output = model(data)[train_mask]
                #         test_output = model(data)[test_mask]
                #
                #         train_prob = torch.sigmoid(train_output)
                #         test_prob = torch.sigmoid(test_output)
                #
                #         train_pred = (train_prob > 0.5).float()  # 假设阈值为0.5
                #         test_pred = (test_prob > 0.5).float()
                #
                #         # 计算训练集和测试集指标
                #         train_acc, train_prec, train_rec, train_f1 = self.calculate_metrics(data.y[train_mask],
                #                                                                             train_pred)
                #         test_acc, test_prec, test_rec, test_f1 = self.calculate_metrics(data.y[test_mask], test_pred)
                #         # 打印训练集和测试集的指标
                #         print(f'Epoch [{epoch + 1}/{args.num_epochs}]')
                #         print(
                #             f'Train - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}')
                #         print(
                #             f'Test - Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}')

                    # model.train()

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
        return retweet_prob_lists[0],retweet_prob_lists[1],retweet_prob_lists[2]

    def update_retweet_prob(self,account_info_list, post_info_list):
        retweet_pos_probability, retweet_neg_probability,retweet_probability = self.FinetunningGAT(account_info_list, post_info_list)
        for i, account in enumerate(account_info_list):
            account.retweet_pos_probability = round(retweet_pos_probability[i][1], 6)
            print(f"Positive Retweet Probability: {account.retweet_pos_probability}")
            account.retweet_neg_probability = round(retweet_neg_probability[i][1], 6)
            print(f"Negative Retweet Probability: {account.retweet_neg_probability}")
            account.retweet_probability = round(retweet_probability[i][1], 6)
            print(f"Retweet Probability: {account.retweet_probability}")
        return account_info_list


if __name__ == '__main__':
    from src.ZJNU.Analog_Propagation.test_data import account_info_list, post_info_list
    model = FinetuningGAT()
    model.update_retweet_prob(account_info_list, post_info_list)
    # # 验证更新后的值
    # for account in account_info_list:
    #     print(f"Account: {account.account_id}")
    #     print(f"Positive Retweet Probability: {account.retweet_pos_probability}")
    #     print(f"Negative Retweet Probability: {account.retweet_neg_probability}")
    #     print("-" * 30)


