from Analog_Propagation.Finetuning_GAT.finetuningGAT import FinetuningGAT
from Analog_Propagation.propagation_model import ICResult

# class Propagation:
#     def __init__(self):
#         pass
#
#     # 第三组
#     # flag:int 1表示正能量增强传播模型结果，2表示负能量抑制传播模型结果，3表示正负能量竞争传播模型结果
#     # ic_result_list = Analog_Propagation11.类名.函数名(selected_id_nodes: List[str],account_info_list: List[AccountInfo],post_info_list: List[PostInfo]，flag=1)
#     @staticmethod
#     def simulation(self,selected_id_nodes,account_info_list,flag):
#         # 1.嵌入修正部分：更新节点的正负转发概率属性
#         #finetuningGAT = FinetuningGAT()
#         #finetuningGAT.update_retweet_prob()
#
#         #模拟传播部分
#         ic_result = ICResult()
#         result = ic_result.simulation_all(selected_id_nodes,account_info_list, post_info_list,flag=1)
#         return result


def simulation(selected_id_nodes, account_info_list, post_info_list, flag):
    # 1.嵌入修正部分：更新节点的正负转发概率属性
    finetuningGAT = FinetuningGAT()

    accounts = finetuningGAT.update_retweet_prob(account_info_list, post_info_list)
    # print("accounts:", accounts)
    # 模拟传播部分
    ic_result = ICResult()
    result = ic_result.simulation_all(selected_id_nodes, accounts, flag)
    return result


if __name__ == "__main__":
    from Analog_Propagation.test_data import account_info_list, post_info_list

    selected_id_nodes = ["user1"]
    result = simulation(selected_id_nodes, account_info_list, post_info_list, flag=1)
    print("P_S:", result.P_S)
    print("P_I1:", result.P_I1)
    print("P_I2:", result.P_I2)
    print("P_R:", result.P_R)
    print("Activation Paths Info 1:", result.activation_paths_info1)
    print("Step Activations Info 1:", result.step_activations_info1)
    print("Activation Paths Info 2:", result.activation_paths_info2)
    print("Step Activations Info 2:", result.step_activations_info2)
