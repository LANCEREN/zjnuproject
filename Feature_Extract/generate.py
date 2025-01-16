import json
from json import JSONDecodeError
from typing import List

import numpy as np
from pydantic import BaseModel


def _read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except JSONDecodeError:
        print(f'文件格式有误: {file_path}')
        return {}


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


if __name__ == '__main__':
    json_content = _read_json_file('./data/data.json')
    if json_content == {}:
        exit(0)
    accounts_id = set([])
    posts_info = json_content.get('posts_info', [])
    accounts_info: List[AccountInfo] = []
    for post in posts_info:
        if post.get('userid', '') == '' or post.get('userid', '') in accounts_id:
            continue
        accounts_id.add(post.get('userid', ''))
        accounts_info.append(AccountInfo(
            account_id=post.get('userid', ''),
            account=post.get('username', ''),
            account_label=[],
            verified=False,
            verified_type=0,
            validate_info='',
            followers_count=np.random.randint(0, 200),
            friends_count=np.random.randint(0, 200),
            ip=post.get('site_ip_name', ''),
            gender='男' if np.random.random() > 0.5 else '女',
            contents_count=np.random.randint(0, 1000),
            photo=post.get('url_profile_image', ''),
            homepage='https://weibo.com/u/' + post.get('userid', ''),
            personal_desc=post.get('site_board_name', ''),
            birthday=str(np.random.randint(1980, 2010)) + '-' + str(np.random.randint(1, 13)) + '-' + str(
                np.random.randint(1, 29)),
            regis_time=str(np.random.randint(2000, 2024)) + '-' + str(np.random.randint(1, 13)) + '-' + str(
                np.random.randint(1, 29)),
            friends=[],
            followers=[]
        ))
    for account in accounts_info:
        count1 = 0
        count2 = 0
        while count1 < account.friends_count:
            selected = accounts_info[np.random.randint(0, len(accounts_info))]
            if selected.account_id == account.account_id:
                continue
            account.friends.append(AccountInfo.Friend(
                account_id=selected.account_id,
                account=selected.account,
                homepage=selected.homepage
            ))
            count1 += 1
        while count2 < account.followers_count:
            selected = accounts_info[np.random.randint(0, len(accounts_info))]
            if selected.account_id == account.account_id:
                continue
            account.followers.append(AccountInfo.Follower(
                account_id=selected.account_id,
                account=selected.account,
                homepage=selected.homepage
            ))
            count2 += 1
    json_content['users_info'] = [account.model_dump() for account in accounts_info]
    with open('./data/generate.json', 'w', encoding='utf-8') as file:
        json.dump(json_content, file, ensure_ascii=False, indent=4)
    print('生成用户数据')
