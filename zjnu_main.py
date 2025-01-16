from typing import List

import Feature_Extract.main
from data_structure import AccountInfo
from data_structure import PostInfo
from DDQN_Interface import DDQNInterface
from data_structure import ICResult
from Influence_Prediction.Influence_process import Pridict_User_Influence
from Analog_Propagation.Propagation import simulation

if __name__ == "__main__":
    # 账号信息列表，里面存放所有账号信息
    account_info_list: List[AccountInfo] = []
    # 帖子信息列表，里面存放所有帖子信息
    post_info_list: List[PostInfo] = []
    # 用户属性列表，里面存放所有用户属性
    user_feature_list: List[AccountInfo.UserFeature] = []
    # 传播模型结果列表
    ic_result_list: List[ICResult] = []
    selected_id_nodes=['1479040710','3198471403']

    # 第一组
    DATA_PATH = 'Feature_Extract/data/generate.json'
    # 返回账号信息列表
    account_info_list = Feature_Extract.main.get_user_data(DATA_PATH)
    # 返回帖子信息列表
    post_info_list = Feature_Extract.main.get_post_data(DATA_PATH, 128)
    # 返回AccountInfo的UserFeature内部类
    user_feature_list = Feature_Extract.main.get_user_feature(account_info_list)
    print("第一组执行完毕")
    # 第二组
    # 修改account_info_list当中所有AccountInfo对象的influence属性
    # 返回influence和user_embeddings属性修改后的账号信息列表
    account_info_list = Pridict_User_Influence(account_info_list, post_info_list).predict_influence()
    print("第二组执行完毕")
    # 交大
    # budget: int  # 种子节点的数目
    # selected_id_nodes: List[str] 传播策略算法根据GNN_DDQN算法选出来的目标用户ID（节点）列表
    # 修改account_info_list当中AccountInfo对象的state属性
    # selected_id_nodes, account_info_list = DDQNInterface(budget=10, account_info_list=account_info_list,post_info_list=post_info_list).output()
    # 第三组
    # flag:int 1表示正能量增强传播模型结果，2表示负能量抑制传播模型结果，3表示正负能量竞争传播模型结果
    ic_result_list = simulation(selected_id_nodes,account_info_list,post_info_list,flag=1)
    print("第三组执行完毕")
