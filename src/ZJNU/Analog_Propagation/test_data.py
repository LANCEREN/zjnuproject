# 创建一个 UserFeature 类的实例
import networkx as nx
import torch
from data_structure import AccountInfo, PostInfo

user_feature1 = AccountInfo.UserFeature(
    account_id="user1",
    gender=1,
    verified=1,
    ip=12345,
    contents_count=120,
    friends_count=50,
    followers_count=200,
    platform=1,
)

user_feature2 = AccountInfo.UserFeature(
    account_id="user2",
    gender=0,
    verified=0,
    ip=67890,
    contents_count=90,
    friends_count=40,
    followers_count=150,
    platform=2,
)

follower1 = AccountInfo.Follower(
    account_id="user1", account="follower_user_1", homepage="https://weibo.com/u/001"
)

follower2 = AccountInfo.Follower(
    account_id="user2", account="follower_user_2", homepage="https://weibo.com/u/002"
)
follower3 = AccountInfo.Follower(
    account_id="user3", account="follower_user_3", homepage="https://weibo.com/u/003"
)

# 创建 AccountInfo 类的实例
account_info1 = AccountInfo(
    account_id="user1",
    account="user_1",
    account_label=["politics", "sports"],
    verified=True,
    verified_type=1,
    validate_info="Verified",
    followers_count=200,
    friends_count=50,
    ip="Beijing",
    gender="男",
    contents_count=120,
    photo="https://image.com/photo1.jpg",
    homepage="https://weibo.com/u/001",
    personal_desc="Tech Enthusiast",
    birthday="1995-07-15",
    regis_time="2020-06-01 10:00:00",
    friends=[],
    followers=[],
    user_feature=user_feature1,
    influence=0.85,
    user_embeddings=torch.randn(10),  # 模拟嵌入向量
    state=False,
    retweet_probability=0.5,
    retweet_pos_probability=0.6,
    retweet_neg_probability=0.4,
)

account_info2 = AccountInfo(
    account_id="user2",
    account="user_2",
    account_label=["entertainment", "tech"],
    verified=False,
    verified_type=0,
    validate_info="Not Verified",
    followers_count=150,
    friends_count=40,
    ip="Shanghai",
    gender="女",
    contents_count=90,
    photo="https://image.com/photo2.jpg",
    homepage="https://weibo.com/u/002",
    personal_desc="Lifestyle Blogger",
    birthday="1992-11-25",
    regis_time="2021-07-10 09:30:00",
    friends=[],
    followers=[follower1],
    user_feature=user_feature2,
    influence=0.75,
    user_embeddings=torch.randn(10),  # 模拟嵌入向量
    state=False,
    retweet_probability=0.7,
    retweet_pos_probability=0.65,
    retweet_neg_probability=0.35,
)

# 你可以创建更多的 AccountInfo 数据，如下所示：
account_info3 = AccountInfo(
    account_id="user3",
    account="user_3",
    account_label=["sports", "lifestyle"],
    verified=True,
    verified_type=1,
    validate_info="Verified",
    followers_count=300,
    friends_count=100,
    ip="Guangzhou",
    gender="男",
    contents_count=150,
    photo="https://image.com/photo3.jpg",
    homepage="https://weibo.com/u/003",
    personal_desc="Sports Enthusiast",
    birthday="1988-02-20",
    regis_time="2019-05-30 11:00:00",
    friends=[],
    followers=[follower1, follower2],
    user_feature=user_feature1,
    influence=0.92,
    user_embeddings=torch.randn(10),  # 模拟嵌入向量
    state=True,  # 这是种子节点
    retweet_probability=0.8,
    retweet_pos_probability=0.85,
    retweet_neg_probability=0.15,
)
account_info4 = AccountInfo(
    account_id="user4",
    account="user_4",
    account_label=["sports", "lifestyle"],
    verified=True,
    verified_type=1,
    validate_info="Verified",
    followers_count=300,
    friends_count=100,
    ip="Guangzhou",
    gender="男",
    contents_count=150,
    photo="https://image.com/photo3.jpg",
    homepage="https://weibo.com/u/003",
    personal_desc="Sports Enthusiast",
    birthday="1988-02-20",
    regis_time="2019-05-30 11:00:00",
    friends=[],
    followers=[follower3],
    user_feature=user_feature2,
    influence=0.92,
    user_embeddings=torch.randn(10),  # 模拟嵌入向量
    state=True,  # 这是种子节点
    retweet_probability=0.8,
    retweet_pos_probability=0.85,
    retweet_neg_probability=0.15,
)
# 将这些账户信息存储到一个列表中
account_info_list = [account_info1, account_info2, account_info3, account_info4]

attention_network = nx.DiGraph()
attention_network.add_edges_from(
    [("user1", "user2"), ("user2", "user3"), ("user3", "user4"), ("user1", "user3")]
)  # user1 关注了 user2, user3，等

# 假设的帖文数据
post_info_list = [  # 示例数据，可以从数据库或其他地方加载
    PostInfo(
        userid="user1",
        relevant_user_id="user2",
        sentiment=1,
        is_original=False,
        topic_id="t1",
        content="content1",
        publish_time="2025-01-08 12:00:00",
        cnt_retweet=10,
        cnt_comment=5,
        cnt_agree=100,
        page_action_type=1,
        nickname="user1",
        url="url1",
    ),
    PostInfo(
        userid="user2",
        relevant_user_id="user3",
        sentiment=-1,
        is_original=False,
        topic_id="t2",
        content="content2",
        publish_time="2025-01-08 13:00:00",
        cnt_retweet=20,
        cnt_comment=10,
        cnt_agree=200,
        page_action_type=2,
        nickname="user2",
        url="url2",
    ),
    PostInfo(
        userid="user3",
        relevant_user_id="user4",
        sentiment=1,
        is_original=False,
        topic_id="t3",
        content="content3",
        publish_time="2025-01-08 14:00:00",
        cnt_retweet=30,
        cnt_comment=15,
        cnt_agree=300,
        page_action_type=3,
        nickname="user3",
        url="url3",
    ),
    PostInfo(
        userid="user4",
        relevant_user_id="user1",
        sentiment=0,
        is_original=False,
        topic_id="t4",
        content="content4",
        publish_time="2025-01-08 15:00:00",
        cnt_retweet=5,
        cnt_comment=2,
        cnt_agree=50,
        page_action_type=4,
        nickname="user4",
        url="url4",
    ),
]
