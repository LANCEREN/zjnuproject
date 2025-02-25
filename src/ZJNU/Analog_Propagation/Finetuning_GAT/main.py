import random
import numpy as np
import torch
from torch_geometric.data import Data

from src.ZJNU.Analog_Propagation.Finetuning_GAT.DataProcess import DataProcess
# from Finetuning_GAT.data_concat import influence_vector, influence_vector_file
# from dataloader import read_node_features, read_edges  # 导入数据读取函数
from src.ZJNU.Analog_Propagation.Finetuning_GAT.model import GAT  # 导入模型
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")


class FinetuningGAT(torch.nn.Module):
    def __init__(self):
        pass
    def calculate_metrics(self,y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        precision = precision_score(y_true, (y_pred > 0.5).astype(int))
        recall = recall_score(y_true, (y_pred > 0.5).astype(int))
        f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
        return accuracy, precision, recall, f1
    def FinetunningGAT(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
        parser.add_argument('--node_features_file', type=str, default='../zjnuproject-master1/Influence_Prediction/out/test.content',
                            help='Path to the node features file')
        parser.add_argument('--edges_file', type=str, default='../zjnuproject-master1/Feature_Extract/out/微博用户关注关系.txt',
                            help='Path to the edges file')
        parser.add_argument('--in_channels', type=int, help='Number of input features')
        parser.add_argument('--out_channels', type=int, help='Number of output channels (number of classes)')
        parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
        parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay coefficient')
        parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
        parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of the dataset to include in the test split')
        parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
        parser.add_argument('--output_file', type=str, default='../zjnuproject-master1/Finetuning_GAT/output/predicted_retweet_probabilities_3.txt',
                            help='Path to the output file for predicted retweet probabilities')

        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        # 1.加载数据：用户特征+影响力特征+网络结构
        data_process = DataProcess()
        # 数据集路径
        node_features_file = args.node_features_file
        edges_file = args.edges_file
        # influence_vector_file = '../Finetuning_GAT/data/test/influence.npy'
        influence_vector_file = '../zjnuproject-master1/Influence_Prediction/out/influence.npy'

        # 加载数据
        x, y = data_process.read_node_features(node_features_file)
        edge_index = data_process.read_edges(edges_file)
        # print(edge_index.min(), edge_index.max())
        influence_vector = data_process.read_influence_features(influence_vector_file)
        node_features = data_process.concatenate_features_and_influence(x,influence_vector)
        # print(node_features.shape)
        # print(node_features)
        # 构建 PyG 图数据对象
        data = Data(x=node_features, edge_index=edge_index, y=y)  # 将标签 y 也传入数据对象

        # 模型超参数
        in_channels = data.num_node_features  # 输入特征维度
        out_channels = 1  # 输出维度
        heads = 4  # 多头注意力的头数

        # 创建模型
        model = GAT(in_channels=in_channels, out_channels=out_channels, heads=heads)

        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()  # 二分类问题使用BCEWithLogitsLoss

        # 划分训练集和测试集
        train_mask = np.random.rand(data.num_nodes) < 0.8
        val_mask = (np.random.rand(data.num_nodes) >= 0.8) & (np.random.rand(data.num_nodes) < 0.9)  # 10% 验证集
        test_mask = np.random.rand(data.num_nodes) >= 0.9

        # 模型训练
        model.train()
        for epoch in range(args.num_epochs):
            optimizer.zero_grad()
            output = model(data)[train_mask]  # 只计算训练集的输出
            loss = loss_fn(output, data.y[train_mask].view(-1, 1))
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    train_output = model(data)[train_mask]
                    test_output = model(data)[test_mask]

                    train_prob = torch.sigmoid(train_output)
                    test_prob = torch.sigmoid(test_output)

                    train_pred = (train_prob > 0.5).float()  # 假设阈值为0.5
                    test_pred = (test_prob > 0.5).float()

                    train_acc, train_prec, train_rec, train_f1 = self.calculate_metrics(data.y[train_mask], train_pred)
                    test_acc, test_prec, test_rec, test_f1 = self.calculate_metrics(data.y[test_mask], test_pred)
                    # print(f"Epoch {epoch} | Loss: {loss.item()} | Train Acc: {train_acc} | Test Acc: {test_acc}")
                model.train()

        # 测试集上的指标
        model.eval()
        with torch.no_grad():
            test_output = model(data)[test_mask]
            test_prob = torch.sigmoid(test_output)
            test_pred = (test_prob > 0.5).float()
            test_acc, test_prec, test_rec, test_f1 = self.calculate_metrics(data.y[test_mask], test_pred)
            # print(f"Test Acc: {test_acc} | Test Prec: {test_prec} | Test Rec: {test_rec} | Test F1: {test_f1}")

        # 输出转发概率
        output = model(data)
        output_prob = torch.sigmoid(output)
        # print(output_prob.shape)
        retweet_probabilities = output_prob.squeeze()  # 得到一维向量

        # print("预测所得转发概率:", retweet_probabilities)

        # 将预测的转发概率保存到文件，包含节点索引
        output_file = args.output_file
        retweet_prob_list = []
        with open(output_file, 'w') as f:
            for idx, prob in enumerate(retweet_probabilities):
                formatted_prob = round(prob.item(), 6)  # 保留6位小数
                f.write(f"{idx} {prob.item():.6f}\n")  # 保存节点索引和对应的转发概率到文件中
                retweet_prob_list.append((idx, formatted_prob))

        # print(f"Predicted retweet probabilities with node indices have been saved to {output_file}")
        return retweet_prob_list



if __name__ == '__main__':
    model = FinetuningGAT()
    result=model.FinetunningGAT()
    print(f"预测网络大小：{len(result)}")
    print(f"预测转发概率：{result}")



