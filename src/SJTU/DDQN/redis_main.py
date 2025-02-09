# algorithm1/main.py
import json

import redis

# 连接 Redis
r = redis.Redis(host='redis', port=6379, db=0)

# 从队列中获取任务并处理
def process_data(input_data):
    print(f"The message from Ochestrator is {input_data}.")
    print("sjtu_ddqn_algorithm is processing data.")
    # 模拟算法处理数据
    data = {
        "id": 1,
        "name": "example",
        "active": True,
        "scores": [85, 90, 95],
        "metadata": {
            "created_at": "2023-10-01T12:00:00Z",
            "updated_at": "2023-10-01T12:05:00Z"
        },
        "algorithm": "sjtu_algorithm_ddqn"
    }
    json_data = json.dumps(data)
    print("sjtu_ddqn_algorithm works down.")
    return json_data

if __name__ == "__main__":
    while True:
        # 从队列中获取任务
        task = r.blpop("sjtu_ddqn_algorithm_queue", timeout=0)[1].decode('utf-8')
        result = process_data(task)

        # 将结果发布到结果队列
        r.rpush("sjtu_ddqn_algorithm_result_queue", result)
        