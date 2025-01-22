import warnings
from typing import List

warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

from data_structure import AccountInfo


class FeatureClusterSelector:
    def __init__(self,
                 accounts_info: List[AccountInfo],
                 sample_size=2000,
                 unwanted_cols=None,
                 vis_threshold=0.2,
                 color_threshold=0.7,
                 correlation_threshold=0.7):
        """
        初始化 FeatureClusterSelector 类。

        参数:
        - accounts_info (List[AccountInfo]): 包含账号信息的列表。
        - sample_size (int): 用于聚类和特征选择的样本大小。
        - unwanted_cols (list): 不用于相关性分析的列名列表。
        - vis_threshold (float): 相关性网络图中显示边的最小相关性阈值。
        - color_threshold (float): 决定相关性边颜色的阈值。
        - correlation_threshold (float): 谱聚类中应用的相关性阈值。
        """
        self.accounts_info = accounts_info
        self.sample_size = sample_size
        self.unwanted_cols = unwanted_cols if unwanted_cols is not None else []
        self.vis_threshold = vis_threshold
        self.color_threshold = color_threshold
        self.correlation_threshold = correlation_threshold

        # 初始化属性
        self.season_stats_full = None
        self.season_stats_sample = None
        self.season_stats_rest = None
        self.numeric_stats_sample = None
        self.constant_numeric_cols = None
        self.variable_numeric_stats_sample = None
        self.norm_variable_numeric_stats_sample = None
        self.corr_matrix_sample = None
        self.n_clusters = None
        self.clusters = None
        self.most_important_features = {}
        self.features_to_keep = []
        self.reduced_numeric_stats_full = None
        self.non_numeric_cols = None

    def read_data(self):
        """从 self.accounts_info 中加载数据并处理列名"""
        # 将 accounts_info 转换为 DataFrame
        self.season_stats_full = pd.DataFrame(self.accounts_info)

        # # 设置显示所有列
        # pd.set_option('display.max_columns', None)

        # 如果列名为数字，尝试修复为字段名
        if self.season_stats_full.columns.dtype == 'int64':  # 列名是数字
            print("Column names are numerical indices, trying to infer field names...")
            first_row = self.season_stats_full.iloc[0]  # 获取第一行内容
            if isinstance(first_row[0], tuple):  # 判断值是否是 tuple
                # 提取所有字段名 (第一行每个元组的第一个元素)
                inferred_column_names = [col[0] if isinstance(col, tuple) else col for col in first_row]
                self.season_stats_full.columns = inferred_column_names  # 设置列名
                self.season_stats_full = self.season_stats_full[1:].reset_index(drop=True)  # 删除第一行数据（已作为列名）
            else:
                raise ValueError("Failed to infer column names. Data format is unexpected.")

        # # 如果列中的值是 tuple，提取 tuple 的第二个元素
        # self.season_stats_full = self.season_stats_full.applymap(lambda x: x[1] if isinstance(x, tuple) else x)
        self.season_stats_full = self.season_stats_full.apply(
            lambda col: col.apply(lambda x: x[1] if isinstance(x, tuple) else x))

        # # 打印前几行用于检查
        # print(self.season_stats_full.head())
        # print(f"Total number of rows in the dataset: {self.season_stats_full.shape[0]}")
        # print(f"Total number of columns in the dataset: {self.season_stats_full.shape[1]}")

        # 确保 sample_size 不超过总行数
        self.sample_size = min(self.season_stats_full.shape[0], self.sample_size)

        # 提取样本数据和剩余数据
        self.season_stats_sample = self.season_stats_full.head(self.sample_size)
        self.season_stats_rest = self.season_stats_full.iloc[self.sample_size:].copy()

        # print(f"Sample size (used for clustering and feature selection): {self.season_stats_sample.shape[0]}")
        # print(f"Remaining data size: {self.season_stats_rest.shape[0]}")

    def preprocess_data(self):
        """选择数值型的列、排除不需要的列、分离零方差列、标准化数据以及清理数据。"""
        # 1. 排除不需要的列
        if self.unwanted_cols:
            print(f"Excluding unwanted columns: {self.unwanted_cols}")
        selected_cols = [col for col in self.season_stats_sample.columns if col not in self.unwanted_cols]

        # 2. 选择数值型的列
        self.numeric_stats_sample = self.season_stats_sample[selected_cols].select_dtypes(include=[np.number])
        print(f"Number of numeric columns selected for analysis: {self.numeric_stats_sample.shape[1]}")

        # 3. 检查是否有数值型列
        if self.numeric_stats_sample.empty:
            raise ValueError("No numeric columns found after excluding unwanted columns.")

        # 4. 识别零方差的数值型列和具有变化的数值型列
        self.constant_numeric_cols = self.numeric_stats_sample.columns[
            self.numeric_stats_sample.nunique(dropna=False) <= 1].tolist()
        self.variable_numeric_stats_sample = self.numeric_stats_sample.drop(columns=self.constant_numeric_cols)
        print(f"Number of constant numeric columns to retain: {len(self.constant_numeric_cols)}")
        if self.constant_numeric_cols:
            print(f"Constant numeric columns: {self.constant_numeric_cols}")

        print(f"Number of variable numeric columns for analysis: {self.variable_numeric_stats_sample.shape[1]}")

        # 5. 识别非数值型的列（包括未被排除的）
        self.non_numeric_cols = [col for col in self.season_stats_full.columns
                                 if col not in self.numeric_stats_sample.columns and col not in self.unwanted_cols]
        print(f"Number of non-numeric columns to retain: {len(self.non_numeric_cols)}")

        # 6. 检查是否有重复列名
        if self.season_stats_full.columns.duplicated().any():
            duplicated_cols = self.season_stats_full.columns[self.season_stats_full.columns.duplicated()].unique()
            raise ValueError(
                f"Duplicated column names found: {duplicated_cols}. Please ensure all column names are unique.")

        # 7. 检查数值型列是否包含非数值数据
        for col in self.variable_numeric_stats_sample.columns:
            if not pd.api.types.is_numeric_dtype(self.season_stats_sample[col]):
                print(f"警告：列 '{col}' 被标记为数值型，但包含非数值数据。将其转换为数值型，非数值数据将被置为 NaN。")
                self.variable_numeric_stats_sample[col] = pd.to_numeric(self.variable_numeric_stats_sample[col],
                                                                        errors='coerce')

        # 8. 删除仍包含 NaN 的行（由于转换可能引入的 NaN）
        initial_shape = self.variable_numeric_stats_sample.shape
        self.variable_numeric_stats_sample.dropna(inplace=True)
        final_shape = self.variable_numeric_stats_sample.shape
        # if initial_shape != final_shape:
        # print(f"Deleted {initial_shape[0] - final_shape[0]} rows due to NaN values after conversion.")

        # 9. 数据标准化
        self.norm_variable_numeric_stats_sample = (
                                                          self.variable_numeric_stats_sample - self.variable_numeric_stats_sample.mean()) / self.variable_numeric_stats_sample.std()
        # print("Data has been standardized (mean=0, std=1).")

        # 10. 清理数据（虽然已经删除了 NaN 和零方差列，但这里保留以防）
        self.norm_variable_numeric_stats_sample = self.clean_dataset(self.norm_variable_numeric_stats_sample)
        # print(f"Sample data size after cleaning: {self.norm_variable_numeric_stats_sample.shape}")

    @staticmethod
    def clean_dataset(df):
        """
        清理数据集，去除包含NaN或无穷大值的行。

        参数:
        - df (pd.DataFrame): 要清理的数据框。

        返回:
        - pd.DataFrame: 清理后的数据框。
        """
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        initial_shape = df.shape
        df.dropna(inplace=True)
        final_shape = df.shape
        # print(f"Cleaned data: dropped {initial_shape[0] - final_shape[0]} rows containing NaN or infinite values.")
        return df.astype(np.float64)

    def compute_correlation_matrix(self):
        """计算样本数据的相关矩阵。"""
        self.corr_matrix_sample = self.norm_variable_numeric_stats_sample.corr()
        # print("相关矩阵（样本数据）：")
        # print(self.corr_matrix_sample)

        # 检查相关矩阵是否包含 NaN
        # if self.corr_matrix_sample.isnull().values.any():
        #     print("警告：相关矩阵包含 NaN 值。请检查数据预处理步骤。")
        # else:
        #     print("相关矩阵已成功计算，不包含 NaN 值。")

    @staticmethod
    def get_color(val, color_threshold):
        """根据相关性值决定边的颜色。"""
        return "r" if val > color_threshold else "b"

    def draw_correlation_graph(self):
        """
        绘制相关矩阵的网络图。

        边的颜色由相关性值决定，颜色阈值由 color_threshold 参数控制。
        """
        columns = self.corr_matrix_sample.columns.to_list()
        G = nx.Graph()

        for i in range(len(columns)):
            col1 = columns[i]
            for j in range(i):
                col2 = columns[j]
                val = self.corr_matrix_sample.values[i, j]
                if abs(val) > self.vis_threshold:
                    G.add_edge(columns.index(col1), columns.index(col2),
                               color=self.get_color(abs(val), self.color_threshold))

        # 打印索引
        # for i, colname in enumerate(columns):
        #     if (i + 1) % 5 == 0:
        #         print(f"{i} : {colname}")
        #     else:
        #         print(f"{i}: {colname}", end=", ")
        # print()  # 换行

        pos = nx.kamada_kawai_layout(G)

        colors = list(nx.get_edge_attributes(G, 'color').values())
        # If 'weight' was not set, default to 0.5
        weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]

        # plt.figure(figsize=(12, 8))
        # nx.draw(G, pos,
        #         edge_color=colors,
        #         width=weights,
        #         with_labels=True,
        #         node_color='lightgreen',
        #         font_size=10,
        #         font_weight='bold')
        # plt.title('Correlation Network Graph')
        # plt.show()

    def plot_dendrogram_and_auto_select_clusters(self, method='ward'):
        """
        绘制树状图并自动选择聚类数量。

        参数:
        - method (str): linkage 方法，如 'ward', 'average', 'complete' 等。

        返回:
        - int: 自动确定的聚类数量。
        """
        # 对特征进行聚类，转置数据
        data = self.norm_variable_numeric_stats_sample.T  # 每一列代表一个特征

        # 计算特征之间的距离矩阵
        distance_matrix = pdist(data, metric='euclidean')  # 或使用其他距离度量，如 'correlation'

        # 检查距离矩阵是否有非有限值
        if not np.all(np.isfinite(distance_matrix)):
            raise ValueError("Distance matrix contains non-finite values")

        # 进行层次聚类
        Z = linkage(distance_matrix, method=method)

        # # 绘制树状图
        # plt.figure(figsize=(15, 10))
        # dendrogram(Z, labels=self.corr_matrix_sample.columns, leaf_rotation=90)
        # plt.title('Dendrogram for Hierarchical Clustering (Sample Data)')
        # plt.xlabel('Feature Index')
        # plt.ylabel('Distance')
        # plt.tight_layout()
        # plt.show()

        # 设置距离阈值
        distance_threshold = 5  # 可以根据需要调整或通过参数传递

        # 使用 fcluster 根据距离阈值生成聚类标签
        cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

        # 计算聚类数
        n_clusters = len(np.unique(cluster_labels))

        # 设置聚类数
        self.n_clusters = n_clusters

        # print(f"确定的聚类数: {self.n_clusters}")
        return self.n_clusters

    def perform_spectral_clustering(self):
        """执行谱聚类并存储聚类结果。"""
        # 使用相关矩阵作为相似度矩阵
        adj_sample = self.corr_matrix_sample.values  # 使用样本数据的相关矩阵

        # 应用阈值
        adj_sample_thresholded = adj_sample.copy()
        adj_sample_thresholded[np.abs(adj_sample_thresholded) < self.correlation_threshold] = 0  # 应用阈值

        # 确保邻接矩阵没有非有限值
        assert np.all(np.isfinite(adj_sample_thresholded)), "Adjacency matrix contains non-finite values"

        # 执行谱聚类
        sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', n_init=100, random_state=42)
        self.clusters = sc.fit_predict(adj_sample_thresholded)
        # print("聚类结果 (Clusters):")
        # for c in np.unique(self.clusters):
        #     cols = self.norm_variable_numeric_stats_sample.columns[np.where(self.clusters == c)]
        #     print(f"Cluster {c}:")
        #     print(f"Number of features: {cols.size}")
        #     print(cols.tolist())
        #     print("---")

    @staticmethod
    def calculate_feature_importance(cluster_data):
        """
        计算特征的重要度。

        参数:
        - cluster_data (pd.DataFrame): 属于同一聚类的特征数据。

        返回:
        - pd.Series: 各特征的重要度。
        """
        # 标准化数据
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(cluster_data)
        normalized_data = pd.DataFrame(normalized_data, columns=cluster_data.columns)

        # 计算特征的标准差（特征区分性）
        feature_discernibility = normalized_data.std()

        # 计算特征的相关性矩阵
        corr_matrix = normalized_data.corr().abs()  # 使用绝对值相关性矩阵
        np.fill_diagonal(corr_matrix.values, 0)  # 置对角线为0，避免特征与自身的相关性影响

        # 计算 dis_i_max 作为所有特征标准差的最大值
        dis_i_max = feature_discernibility.max()

        # 初始化 ind_i 数组
        n_features = normalized_data.shape[1]
        ind_i = np.zeros(n_features)

        # 使用列索引访问 feature_discernibility
        for idx, column in enumerate(normalized_data.columns):
            dis_i = feature_discernibility[column]

            # 找到每个特征与其他特征的相关系数（不包括自身），并对相关性进行排序
            relevant_k = corr_matrix.iloc[idx].sort_values().index.tolist()

            # 将相关特征的名称转换为整数索引
            relevant_k_indices = [normalized_data.columns.get_loc(col) for col in relevant_k]

            # 找到最小相关系数
            min_r = corr_matrix.iloc[idx].min()

            # 计算 ind_i
            if dis_i_max == dis_i:
                ind_i[idx] = 1 / abs(min_r) if min_r != 0 else 1  # 避免除零错误
            elif relevant_k_indices:
                # 计算与相关特征的加权平均
                sum_corr = np.sum(corr_matrix.iloc[idx, relevant_k_indices])
                if sum_corr != 0:
                    ind_i[idx] = 1 / sum_corr
                else:
                    ind_i[idx] = 0  # 避免除零错误

        # 计算特征重要度
        feature_importance = feature_discernibility * ind_i
        return feature_importance

    def select_most_important_features(self):
        """选择每个聚类中最重要的特征。"""
        self.most_important_features = {}
        for cluster_id in np.unique(self.clusters):
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            cluster_data = self.variable_numeric_stats_sample.iloc[:, cluster_indices]
            # print(f"Cluster {cluster_id}")
            feature_importance = self.calculate_feature_importance(cluster_data)
            feature_importance = feature_importance.sort_values(ascending=False)
            # print("Feature Importance:")
            # print(feature_importance)
            # print(f"Number of features in cluster: {len(feature_importance)}")
            # 获取最重要的特征
            if not feature_importance.empty:
                most_important_feature = feature_importance.idxmax()
                self.most_important_features[cluster_id] = most_important_feature
            #     print(f"Most important feature: {most_important_feature}")
            # print("---")

        # print("Most important features by cluster:")
        # for cluster_id, feature in self.most_important_features.items():
        #     # print(f"Cluster {cluster_id}: {feature}")

    def apply_feature_selection(self):
        """仅保留每个聚类的最重要特征，删除其余特征，并应用到整个数据集。"""
        # 收集所有最重要的特征
        self.features_to_keep = list(self.most_important_features.values())
        # print("Features to keep:", self.features_to_keep)

        # 确保特征存在于整个数值型数据集中
        missing_features = [feature for feature in self.features_to_keep if
                            feature not in self.variable_numeric_stats_sample.columns]
        if missing_features:
            print(f"警告：以下特征在整个数值型数据集中未找到并将被忽略: {missing_features}")
            self.features_to_keep = [feature for feature in self.features_to_keep if
                                     feature in self.variable_numeric_stats_sample.columns]

        # print("Final Features to keep after removing missing features:", self.features_to_keep)

        # 创建一个仅包含最重要特征的简化数据集
        # 1. 保留所有非数值型的列
        reduced_non_numeric = self.season_stats_full[self.non_numeric_cols] if self.non_numeric_cols else pd.DataFrame()

        # 2. 保留零方差的数值型列
        reduced_constant_numeric = self.season_stats_full[
            self.constant_numeric_cols] if self.constant_numeric_cols else pd.DataFrame()

        # 3. 保留选定的数值型特征
        reduced_selected_numeric = self.season_stats_full[self.features_to_keep]

        # 4. 合并非数值型列、零方差的数值型列和选定的数值型特征
        data_frames = []
        if not reduced_non_numeric.empty:
            data_frames.append(reduced_non_numeric.reset_index(drop=True))
        if not reduced_constant_numeric.empty:
            data_frames.append(reduced_constant_numeric.reset_index(drop=True))
        data_frames.append(reduced_selected_numeric.reset_index(drop=True))

        self.reduced_numeric_stats_full = pd.concat(data_frames, axis=1)

        # 检查冗余数据
        all_kept_columns = set(self.features_to_keep + self.non_numeric_cols + self.constant_numeric_cols)
        redundant_columns = [col for col in self.season_stats_full.columns if col not in all_kept_columns]

        if redundant_columns:
            print("冗余数据字段名（未被保留的字段）:")
            print(redundant_columns)

        print("未发现冗余字段的存在")

    # def save_reduced_data(self) -> List[AccountInfo]:
    #     try:
    #         # 将简化后的数据集转换为字典列表
    #         reduced_data_as_dict = self.reduced_numeric_stats_full.to_dict(orient='records')
    #
    #         # 创建符合原始数据格式的列表
    #         account_info_list = []
    #         for row in reduced_data_as_dict:
    #             account_info = {}
    #             for key, value in row.items():
    #                 # 将每个键值对重新包装成 (key, value) 的形式
    #                 account_info[key] = (key, value)
    #             account_info_list.append(account_info)
    #
    #         # 打印返回的前几条数据用于调试
    #         print("Reduced data converted back to original format (first 2 rows):")
    #         print(account_info_list[:2])
    #
    #         return account_info_list
    #
    #     except Exception as e:
    #         raise ValueError(f"Error converting reduced data back to original format: {e}")

    def run_all_steps(self):
        """运行所有步骤的综合方法。"""
        self.read_data()
        self.preprocess_data()
        self.compute_correlation_matrix()
        self.draw_correlation_graph()
        self.plot_dendrogram_and_auto_select_clusters()
        self.perform_spectral_clustering()
        self.select_most_important_features()
        self.apply_feature_selection()
        # return self.save_reduced_data()
