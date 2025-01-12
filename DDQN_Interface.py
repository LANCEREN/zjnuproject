from typing import Optional, List
import numpy as np
import torch
from pydantic import BaseModel
from pydantic import validator


# 帖子信息模型
class PostInfo(BaseModel):
    topic_id: str  # 帖子的唯一标识符（ID）
    content: str  # 帖子的内容，通常为文本
    publish_time: str  # 帖子的发布时间，通常为字符串格式的日期时间（例如 "2025-01-08 12:00:00"）
    cnt_retweet: int  # 帖子的转发次数，整数值
    cnt_comment: int  # 帖子的评论次数，整数值
    cnt_agree: int  # 帖子的点赞次数，整数值
    page_picture_pat: Optional[str] = None  # 可选字段，帖子的多媒体链接（例如图片或视频的 URL），如果没有则为 None
    cont_languages: Optional[str] = None  # 可选字段，帖子的语言类型（例如 "zh" 表示中文），如果没有则为 None
    page_action_type: int  # 帖子的动作类型，表示该帖子的互动类型（如点赞、评论等）
    nickname: str  # 发布帖子的用户昵称
    userid: str  # 发布帖子的用户 ID，唯一标识该用户
    url: str  # 帖子的 URL，指向帖子的具体网页链接

    relevant_user_id: str  # 若帖子是转发贴，该属性指向转发帖子的id
    if_original: bool  # 判断该帖子是否是原创贴，true 表示原创帖子，false 表示转发贴


# 账号信息模型
class AccountInfo(BaseModel):
    # Friend 类，用于表示与当前账号相关的好友
    class Friend(BaseModel):
        account_id: str  # 好友的账号 ID
        account: str  # 好友的账号名
        homepage: str  # 好友的主页链接,如"https://weibo.com/u/1015805692"

    # Follower 类，用于表示与当前账号粉丝信息
    class Follower(BaseModel):
        account_id: str  # 粉丝的账号 ID
        account: str  # 粉丝的账号名
        homepage: str  # 粉丝的主页链接,如"https://weibo.com/u/1015805692"

    # UserFeature 类，经过第一组初步处理后，用于表示用户的基本特征信息
    class UserFeature(BaseModel):
        account_id: str  # 用户账号 ID
        gender: int  # 用户的性别(编码数据)
        verified: int  # 用户的认证状态(编码数据)
        ip: int  # 用户的 IP 地址(编码数据)
        contents_count: int  # 用户发布的内容数
        friends_count: int  # 用户的好友数
        followers_count: int  # 用户的粉丝数
        platform: int  # 用户所在平台标识符

    # 三所文件直接读取的用户数据
    account_id: str  # 账号的唯一标识符
    account: str  # 账号的名称（用户名）
    account_label: List[str]  # 账号标签的列表，描述账号的相关特性或分类
    verified: bool  # 是否通过认证，布尔值（True = 已认证，False = 未认证）
    verified_type: int  # 认证类型，整数值表示不同的认证类型（例如：-1 表示未认证，0 表示未认证，1 表示普通认证，2 表示企业认证等）
    validate_info: str  # 认证信息，包含有关认证状态的详细信息
    followers_count: int  # 粉丝数
    friends_count: int  # 好友数
    ip: str  # 用户的 IP 地址（通常是字符串形式，如"山西 太原"）
    gender: str  # 用户性别，字符串类型（如 "男" 表示男性，"女" 表示女性）
    contents_count: int  # 用户发布的内容数量
    photo: str  # 用户的头像链接，指向头像图片的 URL
    homepage: str  # 用户的主页链接
    personal_desc: str  # 用户的个人简介
    birthday: str  # 用户的生日，通常为字符串格式（例如 "1994-04-10 白羊座"）
    regis_time: str  # 用户的注册时间，通常为字符串格式的日期时间（例如 "2020-06-01 10:00:00"）
    friends: List[Friend]  # 好友列表，包含一个或多个 Friend 对象
    followers: List[Follower]  # 粉丝列表，包含一个或多个 Follower 对象
    user_feature: UserFeature  # 用户特征，包含关于用户的更多详细信息，使用 UserFeature 类

    # 第二组生成的属性
    influence: float  # 用户的影响力，浮动类型，通常为数值型，表示该账号在社交平台上的影响力

    # 策略处理后，判断是否是种子节点，种子节点直接设为激活
    state: bool  # 传播过程中该用户是否被激活，false 表示未被激活，true 表示是已经被激活


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
