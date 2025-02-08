import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from Influence_Prediction.GAT.models import GAT
from Influence_Prediction.utils import acc_f1, get_output, split_data
from sklearn.metrics import precision_score, recall_score, roc_auc_score


class GATTrainer:
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo()
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.Tensor(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def normalize(self, mx):
        """Row-normalize sparse matrix."""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class GATTrainer_Static(GATTrainer):
    def __init__(self, args, embeddings, node_features, labels, network):
        self.args = args
        self.model = None
        self.optimizer = None
        self.features = None
        self.adj = None
        self.labels = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self._prepare_data(embeddings, node_features, labels, network)
        self._prepare_model()

    def _prepare_data(self, embeddings, node_features, labels, network):
        node_embeddings = embeddings[:, 1:]
        node_features = node_features[:, 1:]
        labels = labels[:, 1]
        features = np.concatenate((node_embeddings, node_features[:, 1:]), axis=1)
        idx_val, idx_test, idx_train = split_data(labels, 0.1, 0.1, 1234)
        # 划分训练集和测试集
        idx = embeddings[:, 0]
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.array(network.edges())
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
        ).reshape(edges_unordered.shape)
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        adj.tocsr()
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        labels = torch.LongTensor(labels)
        features = torch.FloatTensor(features)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        self.features = features
        self.adj = adj
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def _prepare_model(self):
        # Initialize the model and optimizer
        self.model = GAT(
            nfeat=self.features.shape[1],
            nhid=self.args.hidden,
            nclass=int(self.labels.max()) + 1,
            dropout=self.args.dropout,
            nheads=self.args.nb_heads,
            alpha=self.args.alpha,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        if self.args.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

    def train(self):
        f1_values = []
        bad_counter = 0
        best_f1 = -1
        best_epoch = 0
        best_emb = None
        best_test_metrics = None
        for epoch in range(self.args.epochs):
            f1_value, _ = self._train_epoch(epoch)
            f1_values.append(f1_value)
            if f1_values[-1] > best_f1:
                best_f1 = f1_values[-1]
                best_epoch = epoch
                best_emb, _, best_test_metrics = self.compute_test()
                best_emb = get_output(best_emb)
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == self.args.patience:
                print("Early stopping")
                break
        # Print the final test metrics
        print("Test set results:")
        print(
            "loss_test: {:.4f}".format(best_test_metrics["loss_test"]),
            "acc_test: {:.4f}".format(best_test_metrics["acc_test"]),
            "auc_test: {:.4f}".format(best_test_metrics["auc_test"]),
            "precision_test: {:.4f}".format(best_test_metrics["precision_test"]),
            "recall_test: {:.4f}".format(best_test_metrics["recall_test"]),
            "f1_test: {:.4f}".format(best_test_metrics["f1_test"]),
        )
        print("Optimization Finished!")
        return best_emb

    def _train_epoch(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output, embeddings = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train, f1_train = acc_f1(
            output[self.idx_train], self.labels[self.idx_train], average="binary"
        )
        probabilities = torch.exp(output[self.idx_train])
        # auc_train = roc_auc_score(self.labels[self.idx_train].cpu().numpy(), probabilities[:, 1].cpu().detach().numpy())
        try:
            auc_train = roc_auc_score(
                self.labels[self.idx_train].cpu().numpy(),
                probabilities[:, 1].cpu().detach().numpy(),
            )
        except Exception as e:
            print("无正样本，auc等指标无法计算")
            raise e  # 重新抛出异常
        predicted_labels = torch.argmax(output[self.idx_train], dim=1)
        precision_train = precision_score(
            self.labels[self.idx_train].cpu().numpy(),
            predicted_labels.cpu().numpy(),
            average="binary",
            zero_division=0,
        )
        recall_train = recall_score(
            self.labels[self.idx_train].cpu().numpy(),
            predicted_labels.cpu().numpy(),
            average="binary",
        )

        if (epoch + 1) % 5 == 0:
            print(
                "Epoch: {:04d}".format(epoch + 1),
                "loss_train: {:.4f}".format(loss_train.data.item()),
                "acc_train: {:.4f}".format(acc_train),
                "auc_train: {:.4f}".format(auc_train),
                "precision_train: {:.4f}".format(precision_train),
                "recall_train: {:.4f}".format(recall_train),
                "f1_train: {:.4f}".format(f1_train),
                "time: {:.4f}s".format(time.time() - t),
            )
        loss_train.backward()
        self.optimizer.step()
        t = time.time()
        if not self.args.fastmode:
            self.model.eval()
            output, embeddings = self.model(self.features, self.adj)
        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val, f1_val = acc_f1(
            output[self.idx_val], self.labels[self.idx_val], average="binary"
        )
        probabilities = torch.exp(output[self.idx_val])
        auc_val = roc_auc_score(
            self.labels[self.idx_val].cpu().numpy(),
            probabilities[:, 1].cpu().detach().numpy(),
        )
        predicted_labels = torch.argmax(output[self.idx_val], dim=1)
        precision_val = precision_score(
            self.labels[self.idx_val].cpu().numpy(),
            predicted_labels.cpu().numpy(),
            average="binary",
            zero_division=0,
        )
        recall_val = recall_score(
            self.labels[self.idx_val].cpu().numpy(),
            predicted_labels.cpu().numpy(),
            average="binary",
        )

        if (epoch + 1) % 5 == 0:
            print(
                "Epoch: {:04d}".format(epoch + 1),
                "loss_val: {:.4f}".format(loss_val.data.item()),
                "acc_val: {:.4f}".format(acc_val),
                "auc_val: {:.4f}".format(auc_val),
                "precision_val: {:.4f}".format(precision_val),
                "recall_val: {:.4f}".format(recall_val),
                "f1_val: {:.4f}".format(f1_val),
                "time: {:.4f}s".format(time.time() - t),
            )

        return f1_val, output

    def compute_test(self):
        self.model.eval()
        output, embeddings = self.model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test, f1_test = acc_f1(
            output[self.idx_test], self.labels[self.idx_test], average="binary"
        )
        probabilities = torch.exp(output[self.idx_test])
        auc_test = roc_auc_score(
            self.labels[self.idx_test].cpu().numpy(),
            probabilities[:, 1].cpu().detach().numpy(),
        )
        predicted_labels = torch.argmax(output[self.idx_test], dim=1)
        precision_test = precision_score(
            self.labels[self.idx_test].cpu().numpy(),
            predicted_labels.cpu().numpy(),
            average="binary",
            zero_division=0,
        )
        recall_test = recall_score(
            self.labels[self.idx_test].cpu().numpy(),
            predicted_labels.cpu().numpy(),
            average="binary",
        )
        test_metrics = {
            "loss_test": loss_test,
            "acc_test": acc_test,
            "auc_test": auc_test,
            "precision_test": precision_test,
            "recall_test": recall_test,
            "f1_test": f1_test,
        }
        return embeddings, output, test_metrics


class GATTrainer_Dynamic1(GATTrainer_Static):
    def _prepare_data(self, embeddings, node_features, labels, network):
        node_embeddings = embeddings[:, 1:]
        node_features = node_features[:, 1:]
        # 在每行末尾添加64个零元素
        node_features = np.concatenate(
            (node_features, np.zeros((node_features.shape[0], 64))), axis=1
        )
        labels = labels[:, 1]
        features = np.concatenate((node_embeddings, node_features[:, 1:]), axis=1)
        idx_val, idx_test, idx_train = split_data(labels, 0.1, 0.1, 1234)
        idx = embeddings[:, 0]
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.array(network.edges())
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
        ).reshape(edges_unordered.shape)
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        adj.tocsr()
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        labels = torch.LongTensor(labels)
        features = torch.FloatTensor(features)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        self.features = features
        self.adj = adj
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def train(self):
        f1_values = []
        bad_counter = 0
        best_f1 = -1
        best_epoch = 0
        best_emb = None
        best_test_metrics = None
        for epoch in range(self.args.epochs):
            f1_value, _ = self._train_epoch(epoch)
            f1_values.append(f1_value)
            if f1_values[-1] > best_f1:
                best_f1 = f1_values[-1]
                best_epoch = epoch
                best_emb, _, best_test_metrics = self.compute_test()
                best_emb = get_output(best_emb)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(r"./Influence_Prediction/", "model.pth"),
                )
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == self.args.patience:
                print("Early stopping")
                break
        # Print the final test metrics
        print("Test set results:")
        print(
            "loss_test: {:.4f}".format(best_test_metrics["loss_test"]),
            "acc_test: {:.4f}".format(best_test_metrics["acc_test"]),
            "auc_test: {:.4f}".format(best_test_metrics["auc_test"]),
            "precision_test: {:.4f}".format(best_test_metrics["precision_test"]),
            "recall_test: {:.4f}".format(best_test_metrics["recall_test"]),
            "f1_test: {:.4f}".format(best_test_metrics["f1_test"]),
        )
        print("Optimization Finished!")
        return best_emb


class GATTrainer_Dynamic2(GATTrainer_Static):
    def __init__(self, args, embeddings, node_features, labels, network, embeddings1):
        self.embeddings1 = embeddings1
        super(GATTrainer_Dynamic2, self).__init__(
            args, embeddings, node_features, labels, network
        )

    def _prepare_data(self, embeddings, node_features, labels, network):
        node_embeddings = embeddings[:, 1:]
        node_features = node_features[:, 1:]
        # 在每行末尾添加dynamic1中的64个元素
        node_features = np.hstack((node_features, self.embeddings1[:, 1:]))
        labels = labels[:, 1]
        features = np.concatenate((node_embeddings, node_features[:, 1:]), axis=1)
        idx_val, idx_test, idx_train = split_data(labels, 0.1, 0.1, 1234)
        idx = embeddings[:, 0]
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.array(network.edges())
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
        ).reshape(edges_unordered.shape)
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        adj.tocsr()
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        labels = torch.LongTensor(labels)
        features = torch.FloatTensor(features)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        self.features = features
        self.adj = adj
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def _prepare_model(self):
        # Initialize the model and optimizer
        self.model = GAT(
            nfeat=self.features.shape[1],
            nhid=self.args.hidden,
            nclass=int(self.labels.max()) + 1,
            dropout=self.args.dropout,
            nheads=self.args.nb_heads,
            alpha=self.args.alpha,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        # 加载训练好的模型参数
        self.model.load_state_dict(torch.load(r"./Influence_Prediction/model.pth"))
        if self.args.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
