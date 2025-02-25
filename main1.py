import json
import pickle
from typing import List
import numpy as np

from src.ZJNU.data_structure import AccountInfo, ICResult, PostInfo
from src.ZJNU.Feature_Extract.main import get_post_data, get_user_data, get_user_feature
from src.ZJNU.Influence_Prediction.Influence_process import Pridict_User_Influence


if __name__ == "__main__":

    # 账号信息列表，里面存放所有账号信息
    account_info_list: List[AccountInfo] = []
    # 帖子信息列表，里面存放所有帖子信息
    post_info_list: List[PostInfo] = []
    # 用户属性列表，里面存放所有用户属性
    user_feature_list: List[AccountInfo.UserFeature] = []
    # 传播模型结果列表
    ic_result_list: List[ICResult] = []

    # 第一组
    DATA_PATH = "data/ZJNU/Feature_Extract/generate.json"
    # 返回账号信息列表
    account_info_list = get_user_data(DATA_PATH)
    # 返回帖子信息列表
    post_info_list = get_post_data(DATA_PATH, 128)
    # 返回AccountInfo的UserFeature内部类
    user_feature_list = get_user_feature(account_info_list)
    print("第一组执行完毕")
    # 第二组
    # 修改account_info_list当中所有AccountInfo对象的influence属性
    # 返回influence和user_embeddings属性修改后的账号信息列表
    account_info_list = Pridict_User_Influence(
        account_info_list, post_info_list
    ).predict_influence()
    print("第二组执行完毕")
    # 将account_info_list写入到JSON文件
    with open('data/ZJNU/account_info_list.json', 'w', encoding='utf-8') as f:
        json.dump([account_info.model_dump() for account_info in account_info_list], f, ensure_ascii=False, indent=4,
                  default=lambda o: int(o) if isinstance(o, (np.int32, np.int64)) else o.__dict__)

    # 将post_info_list写入到JSON文件
    with open('data/ZJNU/post_info_list.json', 'w', encoding='utf-8') as f:
        json.dump([post_info.model_dump() for post_info in post_info_list], f, ensure_ascii=False, indent=4,
                  default=lambda o: int(o) if isinstance(o, (np.int32, np.int64)) else o.__dict__)
    print("数据已成功写入JSON文件")

    def save_account_info(account_info_list, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(account_info_list, f)

    save_account_info(account_info_list, 'data/ZJNU/account_info_list.pkl')
    save_account_info(post_info_list, 'data/ZJNU/post_info_list.pkl')
    print("账号信息已保存")
