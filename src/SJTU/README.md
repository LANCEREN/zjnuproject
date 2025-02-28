输入数据：/app/data/ZJNU/account_info_list.json
         /app/data/ZJNU/post_info_list.json
输出数据：/app/data/SJTU/DDQN/data_sjtu/node_features.json
依赖以下步骤：
```bash
    cd zjnuproject
    docker build ./src/SJTU/DDQN -t sjtu_DDQN
    docker run --rm -v $(pwd)/data:/app/data sjtu_DDQN python __mian__.py
```
大致运算时间：
2080ti GPU
10min