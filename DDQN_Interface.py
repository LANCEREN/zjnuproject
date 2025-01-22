from typing import Optional, List, Dict, Any 
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
        personal_desc_tensor: np.ndarray  # 用户的个人描述信息的嵌入向量
        # personal_desc: str  # 用户的个人描述信息的嵌入向量
        followers_count: int  # 用户的粉丝数
        friends_count: int  # 用户的好友数
        platform: int  # 用户所在平台的标识符
        state: bool # 传播过程中该用户是否被激活，false 表示未被激活，true 表示是已经被激活
        zhuanfa_pro: float # 转发概率

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
    #算法映射字典
    user_map_dict: Dict[str, str] = {}
    user_dict_reverse: Dict[str, str] = {}

    # 输入接口
    def __init__(self, budget: int, account_info_list: List[AccountInfo], post_info_list: List[PostInfo], user_map_dict: dict = None, user_dict_reverse: dict = None):
        super().__init__(budget=budget, user_feature=[], post_feature=[], selected_id_nodes=[])
        self.user_map_dict = user_map_dict if user_map_dict is not None else {}
        self.user_dict_reverse = user_dict_reverse if user_dict_reverse is not None else {}
        self.budget = budget

        for account_info in account_info_list:
            user_feature = self.UserFeature(
                account_id=account_info.account_id,
                personal_desc_tensor=account_info.personal_desc_tensor,
                followers_count=account_info.followers_count,
                friends_count=account_info.friends_count,
                state = account_info.state,
                platform=account_info.user_feature.platform,  # 访问 UserFeature 子类的属性
                zhuanfa_pro = account_info.retweet_pos_probability
            )
            self.user_feature.append(user_feature)

        for post_info in post_info_list:
            post_feature = self.PostFeature(
                userid=post_info.userid,
                relevant_user_id=post_info.relevant_user_id,
                publish_time=post_info.publish_time
            )
            self.post_feature.append(post_feature)
# 
    # 输出接口
    def output(self):
        for user in self.user_feature:
            if user.account_id in self.selected_id_nodes:
                user.state = True

        return self.selected_id_nodes, self.user_feature
    
    # def __init__(self, budget: int, account_info_list: List[AccountInfo], post_info_list: List[PostInfo]):
    #     self.budget = budget
    #     self.account_info_list = account_info_list
    #     self.post_info_list = post_info_list
    #     for index in account_info_list:
    #         self.user_feature[index].account_id = account_info_list[index].account_id
    #         self.user_feature[index].personal_desc = account_info_list[index].personal_desc
    #         self.user_feature[index].followers_count = account_info_list[index].followers_count
    #         self.user_feature[index].friends_count = account_info_list[index].friends_count
    #         self.user_feature[index].platform = account_info_list[index].user_feature.platform

    #     for index in post_info_list:
    #         self.post_feature[index].userid = post_info_list[index].userid
    #         self.post_feature[index].relevant_user_id = post_info_list[index].relevant_user_id
    #         self.post_feature[index].publish_time = post_info_list[index].publish_time

    # # 输出接口
    # def output(self):

    #     for index in self.account_info_list:
    #         if self.account_info_list[index].account_id in self.selected_id_nodes:
    #             self.account_info_list[index].state = True

    #     return self.selected_id_nodes, self.account_info_list


if __name__ == "__main__":
    from IMPIM_SJTU.DDQN import train
    train.train_IMP()
    print("The ddqn works.")