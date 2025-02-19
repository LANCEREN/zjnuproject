import os
import pathlib
import sys

from src.Orchestrator.zjnu_main import main
from src.SJTU.DDQN import train

def ddqn_train():
    train.train_IMP()
    print("The ddqn works.")  # The ddqn works.

if __name__ == "__main__":
    print(sys.path)
    ddqn_train()
