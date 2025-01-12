import json
from json import JSONDecodeError
from typing import List

import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Feature_Extract.enums import Platform
from Feature_Extract.utils import FeatureClusterSelector
from data_structure import AccountInfo, PostInfo


def _read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except JSONDecodeError:
        print(f'文件格式有误: {file_path}')
        return {}


def _read_user_data(data_path: str, platform: Platform) -> List[AccountInfo]:
    def _load_weibo_data():
        json_content = _read_json_file(data_path)
        if json_content == {}:
            return []
        accounts_info = json_content.get('users_info', [])
        res: List[AccountInfo] = []
        for account in accounts_info:
            res.append(AccountInfo(
                account_id=account.get('account_id', ''),
                account=account.get('account', ''),
                account_label=account.get('account_label', []),
                verified=account.get('verified', False),
                verified_type=account.get('verified_type', 0),
                validate_info=account.get('validate_info', ''),
                followers_count=account.get('followers_count', 0),
                friends_count=account.get('friends_count', 0),
                ip=account.get('ip', ''),
                gender=account.get('gender', ''),
                contents_count=account.get('contents_count', 0),
                user_feature=AccountInfo.UserFeature(
                    account_id=account.get('account_id', ''),
                    gender=0,
                    ip=0,
                    verified=0,
                    contents_count=account.get('contents_count', 0),
                    friends_count=account.get('friends_count', 0),
                    followers_count=account.get('followers_count', 0),
                    platform=platform.value
                ),
                photo=account.get('photo', ''),
                homepage=account.get('homepage', ''),
                personal_desc=account.get('personal_desc', ''),
                birthday=account.get('birthday', ''),
                regis_time=account.get('regis_time', ''),
                friends=[AccountInfo.Friend(
                    account_id=f.get('account_id', ''),
                    account=f.get('account', ''),
                    homepage=f.get('homepage', '')
                ) for f in account.get('friends', [])],
                followers=[AccountInfo.Follower(
                    account_id=f.get('account_id', ''),
                    account=f.get('account', ''),
                    homepage=f.get('homepage', '')
                ) for f in account.get('followers', [])],
                influence=0,
                state=False,
                retweet_probability=0,
                retweet_neg_probability=0,
                retweet_pos_probability=0,
                user_embeddings=torch.Tensor()
            ))
        return res

    if platform == Platform.Weibo:
        return _load_weibo_data()
    elif platform == Platform.Twitter:
        raise NotImplementedError('暂不支持Twitter数据加载')


def _get_user_feature(data: List[AccountInfo]) -> List[AccountInfo]:
    gender_values = LabelEncoder().fit([d.gender for d in data])
    ip_values = LabelEncoder().fit([d.ip for d in data])
    verified_values = LabelEncoder().fit([d.verified for d in data])
    for d in data:
        if d.user_feature.platform == Platform.Weibo.value:
            d.user_feature.gender = gender_values.transform([d.gender])[0]
            d.user_feature.ip = ip_values.transform([d.ip])[0]
            d.user_feature.verified = verified_values.transform([d.verified])[0]
    return data


def _read_post_data(data_path: str, platform: Platform) -> List[PostInfo]:
    def _load_weibo_data():
        json_content = _read_json_file(data_path)
        if json_content == {}:
            return []
        posts_info = json_content.get('posts_info', [])
        res: List[PostInfo] = []
        for post in posts_info:
            res.append(PostInfo(
                topic_id=post.get('topic_id', ''),
                content=post.get('content', ''),
                publish_time=post.get('publish_time', ''),
                cnt_retweet=post.get('cnt_retweet', 0),
                cnt_agree=post.get('cnt_agree', 0),
                cnt_comment=post.get('cnt_comment', 0),
                page_picture_pat=post.get('page_picture_pat', None),
                cont_languages=post.get('cont_languages', None),
                page_action_type=post.get('page_action_type', 0),
                nickname=post.get('nickname', ''),
                userid=post.get('userid', ''),
                url=post.get('url', ''),
                relevant_user_id='',
                is_original=False,
                sentiment=0,
            ))
        return res

    if platform == Platform.Weibo:
        return _load_weibo_data()


def _get_post_feature(data: List[PostInfo], batch_size: int) -> List[PostInfo]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "tabularisai/multilingual-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    def predict_sentiment(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # 0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"
        res = []
        for p in torch.argmax(probabilities, dim=-1):
            if p < 2:
                res.append(-1)
            elif p == 2:
                res.append(0)
            else:
                res.append(1)
        return res

    for i in range(0, len(data), batch_size):
        content_list = [d.content for d in data[i:i + batch_size]]
        sentiments = predict_sentiment(content_list)
        for j in range(len(data[i:i + batch_size])):
            data[i + j].sentiment = sentiments[j]
    return data


def get_user_data(data_path: str, platform: Platform = Platform.Weibo) -> List[AccountInfo]:
    user_data = _read_user_data(data_path, platform)
    user_data = _get_user_feature(user_data)
    FeatureClusterSelector(
        accounts_info=user_data,
        sample_size=2000,  # 根据数据量调整
        unwanted_cols=[],  # 在此添加不需要的列名
        vis_threshold=0.2,
        color_threshold=0.7,
        correlation_threshold=0.7
    ).run_all_steps()  # 特征选择
    return user_data


def get_post_data(data_path: str, batch_size: int = 1024, platform: Platform = Platform.Weibo) -> List[PostInfo]:
    """
    获取帖子数据，并提取特征
    Args:
        data_path: 数据文件路径
        batch_size: 批处理大小
            显存<4G: 32
            显存<8G: 128
            显存<20G: 512 (约占用15G显存)
            显存<32G: 1024（约占用30G显存）
            如果使用cpu，不建议超过512，会变得不幸
        platform: 平台类型
    """
    post_data = _read_post_data(data_path, platform)
    post_data = _get_post_feature(post_data, batch_size)
    return post_data


def get_user_feature(data: List[AccountInfo]) -> List[AccountInfo]:
    return [f.user_feature for f in data]


if __name__ == '__main__':
    DATA_PATH = 'data/generate.json'
    postData = get_post_data(DATA_PATH, 1024, Platform.Weibo)
    userData = get_user_data(DATA_PATH, Platform.Weibo)
    print('数据加载完成')
