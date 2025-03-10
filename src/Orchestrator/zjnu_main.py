import json
import pickle
from typing import List
import numpy as np
# from src.SJTU.Interface import DDQNInterface
from src.ZJNU.Analog_Propagation.Propagation import simulation
from src.ZJNU.data_structure import AccountInfo, ICResult, PostInfo
from src.ZJNU.Feature_Extract.main import get_post_data, get_user_data, get_user_feature
from src.ZJNU.Influence_Prediction.Influence_process import Pridict_User_Influence


def main():
    # 账号信息列表，里面存放所有账号信息
    account_info_list: List[AccountInfo] = []
    # 帖子信息列表，里面存放所有帖子信息
    post_info_list: List[PostInfo] = []
    # 用户属性列表，里面存放所有用户属性
    user_feature_list: List[AccountInfo.UserFeature] = []
    # 传播模型结果列表
    ic_result_list: List[ICResult] = []
    # selected_id_nodes = ["1479040710", "3198471403"]

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
    breakpoint()
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


    save_account_info(account_info_list, 'account_info_list.pkl')
    save_account_info(post_info_list, 'post_info_list.pkl')
    print("账号信息已保存")

    def load_account_info(file_path):
        with open(file_path, 'rb') as f:
            account_info_list = pickle.load(f)
        return account_info_list


    account_info_list = load_account_info('account_info_list.pkl')
    post_info_list = load_account_info('post_info_list.pkl')
    print("账号信息已加载")

#mian
    # 交大
    # budget: int  # 种子节点的数目
    # selected_id_nodes: List[str] 传播策略算法根据GNN_DDQN算法选出来的目标用户ID（节点）列表
    # 修改account_info_list当中AccountInfo对象的state属性
    # selected_id_nodes, account_info_list = DDQNInterface(budget=10, account_info_list=account_info_list,post_info_list=post_info_list).output()
    # 打开并读取种子节点存放的JSON文件
    with open("src/ZJNU/node_features.json", "r") as file:
        selected_id_nodes = json.load(file)
        # 第三组
        # flag:int 1表示正能量增强传播模型结果，2表示负能量抑制传播模型结果，3表示正负能量竞争传播模型结果
    ic_result_list = simulation(selected_id_nodes, account_info_list, post_info_list, flag=1)
    P_I1_first = ic_result_list.P_I1[1] if ic_result_list.P_I1 else 0
    P_I2_first = ic_result_list.P_I2[1] if ic_result_list else 0

    # 计算总节点数，考虑 P_I1 和 P_I2 是否为空
    total_nodes = ic_result_list.P_S[1] + P_I1_first + P_I2_first + ic_result_list.P_R[1]

    # 计算信息1的覆盖率，如果 P_I1 为空或 None，则视为0
    coverage_I1 = ic_result_list.P_I1[-1] / total_nodes if ic_result_list.P_I1 else 0

    # 计算信息2的覆盖率，如果 P_I2 为空或 None，则视为0
    coverage_I2 = ic_result_list.P_I2[-1] / total_nodes if ic_result_list.P_I2 else 0

    # 输出覆盖率
    print("Negative Information Coverage:", coverage_I1)
    print("Positive Information Coverage:", coverage_I2)
    print("第三组执行完毕")


    # # 第三组
    # # flag:int 1表示正能量增强传播模型结果，2表示负能量抑制传播模型结果，3表示正负能量竞争传播模型结果
    # ic_result_list = simulation(
    #     selected_id_nodes, account_info_list, post_info_list, flag=1
    # )
    # print("第三组执行完毕")
