import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from Influence_Prediction.utils import acc_f1
import logging
import os
import time
import numpy as np
import torch
import torch.optim as optim
import argparse


from Influence_Prediction.utils import get_output, split_data


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class MLPTrainer:
    def __init__(self, args, embeddings, labels):
        self.args = args
        self.model = None
        self.optimizer = None
        self.features = None
        self.adj = None
        self.labels = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None
        self._prepare_data(embeddings, labels)
        self._prepare_model()

    def _prepare_data(self,embeddings, labels):
        labels = labels[:, 1]
        features = embeddings
        idx_val, idx_test, idx_train = split_data(labels, 0.1, 0.1, 1234)
        labels = torch.LongTensor(labels)
        features = torch.FloatTensor(features)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

    def _prepare_model(self):
        # Initialize the model and optimizer
        self.model = SimpleMLP(self.args.feat_dim*2, int(self.labels.max()) + 1)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        if self.args.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

    def train(self):
        f1_values = []
        bad_counter = 0
        best_f1 = -1
        best_epoch = 0
        probs = None
        best_test_metrics = None
        for epoch in range(self.args.epochs):
            f1_value, _ = self._train_epoch(epoch)
            f1_values.append(f1_value)
            if f1_values[-1] > best_f1:
                best_f1 = f1_values[-1]
                best_epoch = epoch
                output, best_test_metrics = self.compute_test()
                probs = torch.exp(output)
                probs = get_output(probs)
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == self.args.patience:
                print("Early stopping")
                break
        # Print the final test metrics
        print("Test set results:")
        print('loss_test: {:.4f}'.format(best_test_metrics['loss_test']),
              'acc_test: {:.4f}'.format(best_test_metrics['acc_test']),
              'auc_test: {:.4f}'.format(best_test_metrics['auc_test']),
              'precision_test: {:.4f}'.format(best_test_metrics['precision_test']),
              'recall_test: {:.4f}'.format(best_test_metrics['recall_test']),
              'f1_test: {:.4f}'.format(best_test_metrics['f1_test']))
        print("Optimization Finished!")
        # 使用索引选择第 1 列和第 3 列
        # probs = probs[:, [0, 2]]
        probs = probs[:, -1]
        return probs

    def _train_epoch(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train, f1_train = acc_f1(output[self.idx_train], self.labels[self.idx_train], average='binary')
        probabilities = torch.exp(output[self.idx_train])
        auc_train = roc_auc_score(self.labels[self.idx_train].cpu().numpy(), probabilities[:, 1].cpu().detach().numpy())
        predicted_labels = torch.argmax(output[self.idx_train], dim=1)
        precision_train = precision_score(self.labels[self.idx_train].cpu().numpy(), predicted_labels.cpu().numpy(),
                                          average='binary', zero_division=0)
        recall_train = recall_score(self.labels[self.idx_train].cpu().numpy(), predicted_labels.cpu().numpy(), average='binary')

        if (epoch + 1) % 5 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'auc_train: {:.4f}'.format(auc_train),
                  'precision_train: {:.4f}'.format(precision_train),
                  'recall_train: {:.4f}'.format(recall_train),
                  'f1_train: {:.4f}'.format(f1_train),
                  'time: {:.4f}s'.format(time.time() - t))
        loss_train.backward()
        self.optimizer.step()
        t = time.time()
        if not self.args.fastmode:
            self.model.eval()
            output = self.model(self.features)
        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val, f1_val = acc_f1(output[self.idx_val], self.labels[self.idx_val], average='binary')
        probabilities = torch.exp(output[self.idx_val])
        auc_val = roc_auc_score(self.labels[self.idx_val].cpu().numpy(), probabilities[:, 1].cpu().detach().numpy())
        predicted_labels = torch.argmax(output[self.idx_val], dim=1)
        precision_val = precision_score(self.labels[self.idx_val].cpu().numpy(), predicted_labels.cpu().numpy(),
                                        average='binary', zero_division=0)
        recall_val = recall_score(self.labels[self.idx_val].cpu().numpy(), predicted_labels.cpu().numpy(), average='binary')

        if (epoch + 1) % 5 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'auc_val: {:.4f}'.format(auc_val),
                  'precision_val: {:.4f}'.format(precision_val),
                  'recall_val: {:.4f}'.format(recall_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'time: {:.4f}s'.format(time.time() - t))

        return f1_val, output

    def compute_test(self):
        self.model.eval()
        output = self.model(self.features)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test, f1_test = acc_f1(output[self.idx_test], self.labels[self.idx_test], average='binary')
        probabilities = torch.exp(output[self.idx_test])
        auc_test = roc_auc_score(self.labels[self.idx_test].cpu().numpy(), probabilities[:, 1].cpu().detach().numpy())
        predicted_labels = torch.argmax(output[self.idx_test], dim=1)
        precision_test = precision_score(self.labels[self.idx_test].cpu().numpy(), predicted_labels.cpu().numpy(),
                                          average='binary', zero_division=0)
        recall_test = recall_score(self.labels[self.idx_test].cpu().numpy(), predicted_labels.cpu().numpy(), average='binary')
        test_metrics = {'loss_test': loss_test, 'acc_test': acc_test, 'auc_test': auc_test, 'precision_test': precision_test, 'recall_test': recall_test, 'f1_test': f1_test}
        return output, test_metrics