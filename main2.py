import json
import pickle
from typing import List

from src.ZJNU.Analog_Propagation.Propagation import simulation
from src.ZJNU.data_structure import AccountInfo, ICResult, PostInfo



if __name__ == "__main__":
    # 账号信息列表，里面存放所有账号信息
    account_info_list: List[AccountInfo] = []
    # 帖子信息列表，里面存放所有帖子信息
    post_info_list: List[PostInfo] = []
    # 用户属性列表，里面存放所有用户属性
    user_feature_list: List[AccountInfo.UserFeature] = []
    # 传播模型结果列表
    ic_result_list: List[ICResult] = []
    def load_account_info(file_path):
        with open(file_path, 'rb') as f:
            account_info_list = pickle.load(f)
        return account_info_list
    account_info_list = load_account_info('data/ZJNU/account_info_list.pkl')
    post_info_list = load_account_info('data/ZJNU/post_info_list.pkl')
    print("账号信息已加载")
    # 打开并读取种子节点存放的JSON文件
    with open("data/SJTU/DDQN/data_sjtu/node_features.json", "r") as file:
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
