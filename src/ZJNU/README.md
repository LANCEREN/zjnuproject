## 前端的输入：budget即传播策略选择的种子节点数目

# 运行流程：
## docker build -t zjnu_ip -f src/ZJNU/Influence_Prediction/Dockerfile .
## docker build ./src/SJTU/DDQN -t sjtu_DDQN
## docker run --rm -v $(pwd)/data:/app/data zjnu_ip python main1.py  (运行数据特征提取模块和群体影响力估计模块)
## docker run --rm -v $(pwd)/data:/app/data sjtu_DDQN python __main__.py (运行传播策略模块)
## docker run --rm -v $(pwd)/data:/app/data zjnu_ip python main2.py (运行传播仿真环境构建模块)

# Feature_Extract：数据特征提取模块
## Brief
输入三所数据，输出用户和贴文特征。耗时1min20s

## inputs
- generate.json  data/ZJNU/Feature_Extract/generate.json

## outputs
- account_info_list.json	data/ZJNU/account_info_list.json
-post_info_list.json	data/ZJNU/post_info_list.json
-account_info_list：存储用户信息
-post_info_list：存储贴文信息

# Influence_Prediction： 群体影响力估计模块
## Brief
输入用户和贴文特征，输出用户群体影响力（更改account_info_list中用户对象的influence属性值）。耗时3min40s

## inputs
- account_info_list：存储用户信息。

## outputs
- account_info_list：修改用户对象的influence属性值

# DDQN：传播策略模块
## Brief
输入用户和贴文特征，输出传播策略选取的种子节点。耗时：10min：runs on 2080ti

## inputs
- account_info_list.json：data/ZJNU/account_info_list.json
- post_info_list.json：data/ZJNU/post_info_list.json
## outputs
- node_features.json：data/SJTU/DDQN/data_sjtu/node_features.json

# Analog_Propagation：传播仿真环境构建模块
## Brief
输入用户、贴文特征以及传播策略选择的种子节点，输出信息传播路径和覆盖率。耗时1min50s

## inputs
- node_features.json：data/SJTU/DDQN/data_sjtu/node_features.json
-   account_info_list.json	data/ZJNU/account_info_list.json
-  post_info_list.json	data/ZJNU/post_info_list.json

## outputs
-  coverage_I1：负能量抑制场景下信息覆盖率
-  coverage_I2：正能量增强场景下信息覆盖率
-  ic_result_list：信息传播路径

## resource reference（浙师大）
- need a cuda GPU
- runs on Core i5_12 / RTX3090 for 7 mins
- use 16G memory