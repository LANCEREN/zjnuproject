from typing import Optional, List
import numpy as np
import torch
from pydantic import BaseModel 
from data_structure import AccountInfo
from data_structure import PostInfo


# 输入数据
class DDQNInterface(BaseModel):
    # 用户属性 UserFeature 类，经过初步处理后，用于表示用户的基本特征信息
    class UserFeature(BaseModel, arbitrary_types_allowed=True):
        account_id: str  # 账号 ID (未确定)
        personal_desc: torch.Tensor  # 用户的个人描述信息的嵌入向量
        followers_count: int  # 用户的粉丝数
        friends_count: int  # 用户的好友数
        platform: int  # 用户所在平台的标识符

    class PostFeature(BaseModel):
        userid: str  # 发布帖子的用户 ID，唯一标识该用户
        relevant_user_id: str  # 转发帖子的用户 ID，唯一标识该用户
        publish_time: str  # 帖子的发布时间，通常为字符串格式的日期时间（例如 "2025-01-08"）

    # 算法所需超参数
    budget: int  # 种子节点的数目

    # 算法所需输入数据
    user_feature: List[UserFeature]  # 用户属性
    post_feature: List[PostFeature]  # 帖子属性

    # 算法将要反馈的输出数据  
    selected_id_nodes: List[str]  # 根据GNN_DDQN算法选出来的目标用户ID（节点）列表

    # 输入接口
    def __init__(self, budget: int, account_info_list: List[AccountInfo], post_info_list: List[PostInfo]):

        self.budget = budget
        self.account_info_list = account_info_list
        self.post_info_list = post_info_list

        for index in account_info_list:
            self.user_feature[index].account_id = account_info_list[index].account_id
            self.user_feature[index].personal_desc = account_info_list[index].personal_desc
            self.user_feature[index].followers_count = account_info_list[index].followers_count
            self.user_feature[index].friends_count = account_info_list[index].friends_count
            self.user_feature[index].platform = account_info_list[index].user_feature.platform

        for index in post_info_list:
            self.post_feature[index].userid = post_info_list[index].userid
            self.post_feature[index].relevant_user_id = post_info_list[index].relevant_user_id
            self.post_feature[index].publish_time = post_info_list[index].publish_time

    # 输出接口
    def output(self):

        for index in self.account_info_list:
            if self.account_info_list[index].account_id in self.selected_id_nodes:
                self.account_info_list[index].state = True

        return self.selected_id_nodes, self.account_info_list
