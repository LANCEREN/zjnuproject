import argparse
import datetime



import numpy as np
from torch import dtype

from Influence_Prediction.Attention import AttnTrainer
from Influence_Prediction.Data_process import DataProcessor, TimeWindowSplitter, NodeEmbedding, LabelGenerator, FeatureExtractor
from Influence_Prediction.GAT_Training import GATTrainer_Static, GATTrainer_Dynamic1, GATTrainer_Dynamic2
from Influence_Prediction.MLP_Training import MLPTrainer
import scipy.sparse as sp
import torch
import networkx as nx


class InfluencePredictor:
    def __init__(self, static_path: str, dynamic_path: str, datatime1: datetime.date,
                 datatime2: datetime.date, datatime3: datetime.date, datatime4: datetime.date):

        self.static_path = static_path
        self.dynamic_path = dynamic_path
        self.datatime1 = datatime1
        self.datatime2 = datatime2
        self.datatime3 = datatime3
        self.datatime4 = datatime4


    def predict_influence(self) -> np.ndarray[float]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
        parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
        parser.add_argument('--seed', type=int, default=72, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
        parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
        parser.add_argument('--patience', type=int, default=100, help='Patience')
        parser.add_argument('--feat_dim', type=int, default=64, help='Feature dimension')
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # 创建类
        data_processor = DataProcessor()
        time_Windows = TimeWindowSplitter()
        node_embedding = NodeEmbedding()
        label_generator = LabelGenerator()
        feature_extractor = FeatureExtractor()

        # 加载网络数据
        # static_network = data_processor.load_networks_txt(r'./Influence_Prediction/data/higgs_social_data4.txt')
        # dynamic_network = data_processor.load_networks_txt(r'./Influence_Prediction/data/higgs_retweet_data4.txt')
        # dynamic1, dynamic2 = time_Windows.split_by_time(r'./Influence_Prediction/data/higgs_social_data4.txt',
        #                                                 r'./Influence_Prediction/data/higgs_retweet_data_with_date.txt',
        #                                                 datetime.date(2012, 7, 1),
        #                                                 datetime.date(2012, 7, 4),
        #                                                 datetime.date(2012, 7, 5),
        #                                                 datetime.date(2012, 7, 7))
        # 加载网络数据
        static_network,idx_map,idx = data_processor.load_networks_txt(self.static_path)
        dynamic_network = data_processor.load_networks_with_time_txt(self.static_path,
                                                                     self.dynamic_path,idx_map)
        dynamic1, dynamic2 = time_Windows.split_by_time(self.static_path,
                                                        self.dynamic_path,idx_map,
                                                        self.datatime1,
                                                        self.datatime2,
                                                        self.datatime3,
                                                        self.datatime4)
        print(len(static_network.nodes()))
        print(len(dynamic_network.nodes()))
        print(len(dynamic1.nodes()))
        print(len(dynamic2.nodes()))
        # 生成节点嵌入
        print("生成静态网络节点嵌入...")
        static_embedding_data = node_embedding.generate_embeddings(static_network, args.feat_dim)
        # static_embedding_data = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 64))))
        # static_embedding_data = np.genfromtxt(r'./Influence_Prediction/data/static_embeddings.txt',dtype=float)
        print("生成第一个时间窗口节点嵌入...")
        dynamic1_embedding_data = node_embedding.generate_embeddings(dynamic1, args.feat_dim)
        # dynamic1_embedding_data = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 64))))
        # dynamic1_embedding_data = np.genfromtxt(r'./Influence_Prediction/data/dynamic_embeddings1.txt',dtype=float)
        print("生成第二个时间窗口节点嵌入...")
        dynamic2_embedding_data = node_embedding.generate_embeddings(dynamic2, args.feat_dim)
        # dynamic2_embedding_data = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 64))))
        # dynamic2_embedding_data = np.genfromtxt(r'./Influence_Prediction/data/dynamic_embeddings2.txt',dtype=float)
        # 生成群体特征
        print("提取静态网络群体特征...")
        # static_features = feature_extractor.extract_group_static_features(static_network, dynamic_network, 20)
        static_features = np.genfromtxt(r'./Influence_Prediction/data/node_data.txt',dtype=float)
        print("提取第一个时间窗口群体特征...")
        # dynamic1_features = feature_extractor.extract_group_dynamic_features(dynamic1, 20)
        dynamic1_features = np.genfromtxt(r'./Influence_Prediction/data/node_data1.txt',dtype=float)
        # dynamic1_features = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 1))))

        print("提取第二个时间窗口群体特征...")
        # dynamic2_features = feature_extractor.extract_group_dynamic_features(dynamic2, 20)
        dynamic2_features = np.genfromtxt(r'./Influence_Prediction/data/node_data2.txt',dtype=float)
        # dynamic2_features = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 1))))
        # 生成标签
        print("生成网络标签...")
        static_labels = label_generator.label_network(static_network, dynamic_network)
        dynamic1_labels = label_generator.label_network(static_network, dynamic1)
        dynamic2_labels = label_generator.label_network(static_network, dynamic2)
        # 训练网络
        print("训练静态网络...")
        static_trainer = GATTrainer_Static(args, static_embedding_data, static_features, static_labels, static_network)
        # static_trainer = GATTrainer_Static(args, static_embedding_data, np.ones((static_embedding_data['embeddings'].shape[0],2)), static_labels, static_network)
        emb_static_with_idx = static_trainer.train()
        print("训练第一个时间窗口网络...")
        dynamic1_trainer = GATTrainer_Dynamic1(args, dynamic1_embedding_data, dynamic1_features, dynamic1_labels, dynamic1)
        emb_dynamic1_with_idx = dynamic1_trainer.train()
        print("训练第二个时间窗口网络...")
        dynamic2_trainer = GATTrainer_Dynamic2(args, dynamic2_embedding_data, dynamic2_features, dynamic2_labels, dynamic2, emb_dynamic1_with_idx)
        emb_dynamic2_with_idx = dynamic2_trainer.train()
        # 合并两个时间窗口嵌入
        emb_dynamic = np.concatenate((emb_dynamic1_with_idx[:,1:], emb_dynamic2_with_idx[:,1:]), axis=1)
        # 训练注意力机制
        attn_trainer = AttnTrainer(args,emb_dynamic,static_labels)
        emb_dynamic_idx = attn_trainer.train()
        # 合并静态网络和动态网络的向量
        mlp_input = np.concatenate((emb_static_with_idx[:,1:], emb_dynamic_idx[:,1:]), axis=1)
        # embeddings = np.column_stack((emb_static_with_idx[:,0].astype(int), mlp_input, static_labels[:,1]))
        embeddings = np.column_stack((np.array(idx), mlp_input, static_labels[:,1]))
        with open('./Influence_Prediction/out/test.content', 'w') as f:
            for row in embeddings:
                f.write('\t'.join(map(lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x), row)) + '\n')
        # np.savetxt('./Influence_Prediction/out/test.content', embeddings, delimiter=',')
        # 训练MLP
        mlp_trainer = MLPTrainer(args, mlp_input, static_labels)
        influence = mlp_trainer.train()
        return influence

class Pridict_User_Influence:

    def __init__(self, account_info_list: list, post_info_list: list):

        self.account_info_list = account_info_list
        self.post_info_list = post_info_list

    def predict_influence(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
        parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
        parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
        parser.add_argument('--seed', type=int, default=72, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
        parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
        parser.add_argument('--patience', type=int, default=100, help='Patience')
        parser.add_argument('--feat_dim', type=int, default=64, help='Feature dimension')
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # 创建类
        data_processor = DataProcessor()
        time_Windows = TimeWindowSplitter()
        node_embedding = NodeEmbedding()
        label_generator = LabelGenerator()
        feature_extractor = FeatureExtractor()
        # 加载网络数据
        static_network = data_processor.load_static_network(self.account_info_list)

        dynamic_network = data_processor.load_dynamic_network(self.post_info_list)

        dynamic1, dynamic2 = time_Windows.split_by_time(self.post_info_list,static_network)
        print(len(static_network.nodes()))
        print(len(dynamic_network.nodes()))
        print(len(dynamic1.nodes()))
        print(len(dynamic2.nodes()))
        # 对网络节点进行映射
        idx_map = {j: i for i, j in enumerate(sorted(static_network.nodes))}
        idx = sorted(static_network.nodes)
        static_network = nx.relabel_nodes(static_network, idx_map)
        dynamic_network = nx.relabel_nodes(dynamic_network, idx_map)
        dynamic1 = nx.relabel_nodes(dynamic1, idx_map)
        dynamic2 = nx.relabel_nodes(dynamic2, idx_map)
        # 生成节点嵌入
        print("生成静态网络节点嵌入...")
        static_embedding_data = node_embedding.generate_embeddings(static_network, args.feat_dim)
        print("生成第一个时间窗口节点嵌入...")
        dynamic1_embedding_data = node_embedding.generate_embeddings(dynamic1, args.feat_dim)
        print("生成第二个时间窗口节点嵌入...")
        dynamic2_embedding_data = node_embedding.generate_embeddings(dynamic2, args.feat_dim)
        # # 生成节点嵌入
        # print("生成静态网络节点嵌入...")
        # static_embedding_data = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 64))))
        # print("生成第一个时间窗口节点嵌入...")
        # dynamic1_embedding_data = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 64))))
        # print("生成第二个时间窗口节点嵌入...")
        # dynamic2_embedding_data = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 64))))
        # 生成群体特征
        print("提取静态网络群体特征...")
        static_features = feature_extractor.extract_group_static_features(static_network, dynamic_network, 20)
        # static_features = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 2))))
        print("提取第一个时间窗口群体特征...")
        dynamic1_features = feature_extractor.extract_group_dynamic_features(dynamic1, 20)
        # dynamic1_features = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 1))))
        print("提取第二个时间窗口群体特征...")
        dynamic2_features = feature_extractor.extract_group_dynamic_features(dynamic2, 20)
        # dynamic2_features = np.column_stack((sorted(static_network.nodes()), np.ones((len(static_network.nodes()), 1))))
        # 生成标签
        print("生成网络标签...")
        static_labels = label_generator.label_network(static_network, dynamic_network)
        dynamic1_labels = label_generator.label_network(static_network, dynamic1)
        dynamic2_labels = label_generator.label_network(static_network, dynamic2)
        # 训练网络
        print("训练静态网络...")
        static_trainer = GATTrainer_Static(args, static_embedding_data, static_features, static_labels, static_network)
        emb_static_with_idx = static_trainer.train()
        print("训练第一个时间窗口网络...")
        dynamic1_trainer = GATTrainer_Dynamic1(args, dynamic1_embedding_data, dynamic1_features, dynamic1_labels,
                                               dynamic1)
        emb_dynamic1_with_idx = dynamic1_trainer.train()
        print("训练第二个时间窗口网络...")
        dynamic2_trainer = GATTrainer_Dynamic2(args, dynamic2_embedding_data, dynamic2_features, dynamic2_labels,
                                               dynamic2, emb_dynamic1_with_idx)
        emb_dynamic2_with_idx = dynamic2_trainer.train()
        # 合并两个时间窗口嵌入
        emb_dynamic = np.concatenate((emb_dynamic1_with_idx[:, 1:], emb_dynamic2_with_idx[:, 1:]), axis=1)
        # 训练注意力机制
        attn_trainer = AttnTrainer(args, emb_dynamic, static_labels)
        emb_dynamic_idx = attn_trainer.train()
        # 合并静态网络和动态网络的向量
        mlp_input = np.concatenate((emb_static_with_idx[:, 1:], emb_dynamic_idx[:, 1:]), axis=1)
        # embeddings = np.column_stack((np.array(idx), mlp_input)
        embeddings = {i: mlp_input[i] for i in range(len(idx))}
        # 训练MLP
        mlp_trainer = MLPTrainer(args, mlp_input, static_labels)
        influence = mlp_trainer.train()
        for account in self.account_info_list:
            account_id = account.account_id
            idx_value = idx_map.get(account_id)
            account.user_embeddings = torch.tensor(embeddings[idx_value])
            account.influence = influence[idx_value]
        return self.account_info_list
# if __name__ == '__main__':
#     predictor = InfluencePredictor()
#     result = predictor.predict_influence()
#     # 保存为 .npy 文件
#     np.save('influence.npy', result)
#     print(f"文件已保存")







