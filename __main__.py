import os
import pathlib
import sys
# 获取当前脚本所在的目录，并上溯到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from src.ZJNU.Influence_Prediction import Data_process
from src.Orchestrator.zjnu_main import main
from src.SJTU.DDQN import train

def ddqn_train():
    train.train_IMP()
    print("The ddqn works.")  # The ddqn works.

if __name__ == "__main__":
    print(sys.path)
    ddqn_train()
