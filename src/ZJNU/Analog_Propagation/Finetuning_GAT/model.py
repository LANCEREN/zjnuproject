import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# 构建 GAT 模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, hidden_channels=64):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, dropout=0.6
        )  # 第二层GAT

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))  # 第一层GAT，带ELU激活
        x = self.conv2(x, edge_index)  # 第二层GAT
        # x = torch.sigmoid(x)  # 将输出映射到[0,1]之间
        return x


class GATModel(nn.Module):
    def __init__(
        self, user_feature_dim, post_feature_dim, hidden_dim, output_dim, num_heads=4
    ):
        super(GATModel, self).__init__()

        # GAT卷积层
        self.gat1 = GATConv(user_feature_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)

        # 拼接用户特征和帖子特征
        self.fc1 = nn.Linear(hidden_dim + post_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, user_features, post_features, edge_index):
        # 第一层GAT卷积
        x_user = F.relu(self.gat1(user_features, edge_index))

        # 第二层GAT卷积
        x_user = F.relu(self.gat2(x_user, edge_index))

        # 拼接用户表示和帖子特征
        x = torch.cat([x_user, post_features], dim=1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 输出转发概率

        return x
