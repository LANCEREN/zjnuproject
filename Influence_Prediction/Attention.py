import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from Influence_Prediction.utils import acc_f1

import time

import torch
import torch.optim as optim

from Influence_Prediction.utils import get_output, split_data


class attn_model(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # self.attn = attn_mechanism(in_feats, out_feats)
        self.attn1 = nn.Linear(in_feats, 1, bias=False)
        self.attn2 = nn.Linear(in_feats, 1, bias=False)
        self.linear = nn.Linear(in_feats, out_feats)
    def forward(self, x):
        # 拆分为两个 [22933, 64] 的张量
        x1, x2 = torch.split(x, int(x.shape[1]/2), dim=1)
        attn1 = self.attn1(x1)
        attn2 = self.attn2(x2)
        # 将attn1和attn2拼接在一起（形状：[22933, 2]）
        attn = torch.cat([attn1, attn2], dim=1)
        # 对每一行的两个注意力值进行softmax操作
        attn = F.softmax(attn, dim=1)  # dim=1 表示沿着列（每行的注意力值）
        # 分别获得 softmax 后的 attn1 和 attn2
        attn1, attn2 = attn[:, 0].view(-1, 1), attn[:, 1].view(-1, 1)
        x1 = attn1 * x1
        x2 = attn2 * x2
        embeddings = x1 + x2
        x = self.linear(embeddings)
        return F.log_softmax(x, dim=1), embeddings


class AttnTrainer:
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
        self.model = attn_model(self.args.feat_dim, int(self.labels.max()) + 1)

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
                # torch.save(self.model.state_dict(),
                #            os.path.join(r'C:\Social_Influence\autoprocess', 'model.pth'))
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
        return best_emb

    def _train_epoch(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output, embeddings = self.model(self.features)
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
            output, embeddings = self.model(self.features)
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
        output, embeddings = self.model(self.features)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test, f1_test = acc_f1(output[self.idx_test], self.labels[self.idx_test], average='binary')
        probabilities = torch.exp(output[self.idx_test])
        auc_test = roc_auc_score(self.labels[self.idx_test].cpu().numpy(), probabilities[:, 1].cpu().detach().numpy())
        predicted_labels = torch.argmax(output[self.idx_test], dim=1)
        precision_test = precision_score(self.labels[self.idx_test].cpu().numpy(), predicted_labels.cpu().numpy(),
                                          average='binary', zero_division=0)
        recall_test = recall_score(self.labels[self.idx_test].cpu().numpy(), predicted_labels.cpu().numpy(), average='binary')
        test_metrics = {'loss_test': loss_test, 'acc_test': acc_test, 'auc_test': auc_test, 'precision_test': precision_test, 'recall_test': recall_test, 'f1_test': f1_test}
        return embeddings, output, test_metrics