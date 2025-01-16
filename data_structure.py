from typing import Optional, List, Dict, Any 
import torch
from pydantic import BaseModel


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
    is_original: bool  # 判断该帖子是否是原创贴，true 表示原创帖子，false 表示转发贴

    # 贴文情感特征(1.10日添加) 1表示正能量情感，-1表示负能量情感，0表示中性情感
    sentiment: int

# 账号信息模型 
class AccountInfo(BaseModel, arbitrary_types_allowed=True):
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
        verified: Optional[int]  # 用户的认证状态(编码数据)
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
    # 用户静动态网络信息(1.10日添加)，由第二组写入，给第三组嵌入修正使用
    user_embeddings: torch.Tensor

    # 策略处理后，判断是否是种子节点，种子节点直接设为激活
    state: bool  # 传播过程中该用户是否被激活，false 表示未被激活，true 表示是已经被激活

    retweet_probability: float  # 第三组处理的嵌入修正得到的每个节点的转发概率
    retweet_pos_probability: float  # 第三组处理的嵌入修正得到的每个节点的正能量推文转发概率（1.10日增加）
    retweet_neg_probability: float  # 第三组处理的嵌入修正得到的每个节点的负能量推文转发概率（1.10日增加）


# 传播的每一轮的信息模型 
class ICResult(BaseModel):
    seed: List[int]  # 策略生成的种子节点列表
    P_S: List[int]  # 易感状态节点的数量
    P_I1: List[int]  # I1状态节点的累计数量
    P_I2: List[int]  # I2状态节点的累计数量
    P_R: List[int]  # 免疫状态节点的累计数量
    activation_paths_info1: List[str]  # 记录I1传播的激活路径,如['1 2','32 42']代表1激活2,32激活42
    activation_paths_info2: List[str]  # 记录I2传播的激活路径
    step_activations_info1: List[str]  # 记录每步I1状态节点的激活信息，如['1 5 3','9 4 7 8']代表第一个时间步激活1,5,3，第二个时间步激活9,4,7,8
    step_activations_info2: List[str]  # 记录每步I2状态节点的激活信息

