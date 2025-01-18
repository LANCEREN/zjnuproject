# json tools
import os, sys
import json
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any 
from DDQN_Interface import DDQNInterface
from IMPIM_SJTU.DDQN.config import dataPath


# Load JSON data from the file 
def readJSONdata(file_name:str, file_directory:Path):
    file_path = file_directory / Path(file_name)
    with open(file_path, 'r') as file:
        json_file_data = json.load(file)
    return json_file_data

# 预处理三所提供的JSON文件数据
def json_preprocess_sansuo():

    # Load JSON data from the file
    file_name = 'json-1724658998077.json'
    data = readJSONdata(file_name, dataPath.data_sansuo_directory)

    # 定义递归函数来获取所有key并去重
    def get_all_keys(data, parent_key='', seen_keys=set()):
        keys = []
        if isinstance(data, dict):
            for k, v in data.items():
                full_key = f"{parent_key}.{k}" if parent_key else k
                if full_key not in seen_keys:
                    keys.append(full_key)
                    seen_keys.add(full_key)
                keys.extend(get_all_keys(v, full_key, seen_keys))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                keys.extend(get_all_keys(item, parent_key, seen_keys))
        return keys

    # 提取 posts_info 中所有的key
    posts_info = data.get('posts_info', [])
    all_keys = get_all_keys(posts_info)

    # 处理数据，将缺失的数据标记为 None
    processed_data = []
    for post in posts_info:
        processed_post = {}
        for key in all_keys:
            # 分割多层key，逐层获取值
            keys = key.split('.')
            value = post
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, None)
                else:
                    value = None
                    break
            processed_post[key] = value
        processed_data.append(processed_post)

    # 打印整理后的数据
    for post in processed_data:
        print(json.dumps(post, ensure_ascii=False, indent=4))
        print("-" * 40)

    # 保存整理后的数据到新的JSON文件
    output_file_path = dataPath.data_sansuo_directory + './processed_posts_info_with_missing_' + file_name
    with open(output_file_path, 'w') as output_file:
        json.dump(processed_data, output_file, ensure_ascii=False, indent=4)

    print(f"Processed data saved to {output_file_path}")


# 从三所提供的JSON文件数据中挑选数据
def json_selection_process_sansuo():
    
    def save_txt(df_list1, df_list2, save_path):
        if os.path.isfile(save_path):
            os.remove(save_path)

        with open(save_path, "w+") as f:
            for i in range(len(df_list1)):
                f.write('{:} {:}\n'.format(df_list1[i], df_list2[i]))
            f.close()

    import pandas as pd
    # 使用pandas的read_json()方法读取JSON文件
    file_name = Path('processed_posts_info_with_missing_json-1724658998077.json')
    file_path = dataPath.data_sansuo_directory / file_name
    df = pd.read_json(file_path)

    # 打印DataFrame对象
    # print(df)

    # 根据不同方案的输入，选择需要提取特定列的值或者特征
    assert len(df['userid']) == len(df['relevant_userid']), '数据丢失，请检查！'
    userid_list = list()
    relevant_userid_list = list()

    for i in range(len(df['userid'])):

        userid_list.append('None' if pd.isnull(df['userid'][i]) else int(df['userid'][i]))
        relevant_userid_list.append('None' if pd.isnull(df['relevant_userid'][i]) else int(df['relevant_userid'][i]))

    save_path = dataPath.data_sansuo_directory / Path('id_json.txt')
    save_txt(userid_list, relevant_userid_list, save_path)


# 预处理浙师大提供的JSON文件数据,转化为DDQNInterface类
def json_preprocess_zjnu(budget:int, account_info_list_file_name:str, post_info_list_file_name:str):
    from data_structure import AccountInfo
    from data_structure import PostInfo

    account_info_list_file_data = readJSONdata(account_info_list_file_name, dataPath.data_zjnu_directory)
    post_info_list_file_data = readJSONdata(post_info_list_file_name, dataPath.data_zjnu_directory)

    account_info_list = []  # 这里填入实际的 account_info_list 数据
    post_info_list = []  # 这里填入实际的 post_info_list 数据

    # FIXME: zjnu 数据更新后可删除此行
    # 创建一个空字典来存储用户及其对应的编号
    user_dict = {}
    user_dict_reverse = {}
    current_id = 0

    for account_info_dict in account_info_list_file_data:
        # FIXME: zjnu 数据更新后可删除此行
        # 在这个函数里面加
        import numpy as np
        array = np.random.rand(1, 20)
        account_info_dict['personal_desc_tensor'] = array
        # 遍历用户列表，将每个不重复的用户添加到字典中, str -> str(int)
        user_account_id = account_info_dict['account_id']
        if user_account_id not in user_dict:
            user_dict[user_account_id] = str(current_id)
            user_dict_reverse[str(current_id)] = user_account_id
            current_id += 1
        account_info_dict['account_id'] = user_dict[user_account_id]
        # user_embeddings输入格式不对,将 user_embeddings 列表转换为张量
        if 'user_embeddings' in account_info_dict:
            account_info_dict['user_embeddings'] = torch.tensor(account_info_dict['user_embeddings'])

        account_info = AccountInfo(**account_info_dict)
        account_info_list.append(account_info)

    for post_info_dict in post_info_list_file_data:
        # FIXME: 由于三所数据不完整，需要手动调整 id
        import random
        random_keys = random.sample(list(user_dict.keys()), 2)
        post_info_dict['userid'] = user_dict[random_keys[0]]
        post_info_dict['relevant_user_id'] = user_dict[random_keys[1]]

        post_info = PostInfo(**post_info_dict)
        post_info_list.append(post_info)
    

    # 创建 DDQNInterface 实例
    ddqn_interface = DDQNInterface(budget=100, account_info_list=account_info_list, post_info_list=post_info_list, 
                                    user_map_dict=user_dict, user_dict_reverse=user_dict_reverse)
    return ddqn_interface
