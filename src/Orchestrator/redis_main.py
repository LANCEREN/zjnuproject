# orchestrator/main.py
import time
import json

import redis

# 连接 Redis
r = redis.Redis(host="redis", port=6379, db=0)


def run_algorithm(algorithm_name, input_data):
    # 将任务发布到队列
    queue_name = f"{algorithm_name}_queue"
    r.rpush(queue_name, input_data)
    print(f"Data sent to {queue_name}: {input_data}")
    # 等待结果
    result_queue_name = f"{algorithm_name}_result_queue"
    while True:
        result = r.blpop(result_queue_name, timeout=10)[1].decode('utf-8')
        if result: 
            data = json.loads(result)
            print(f"Data received from {result_queue_name}|{data['algorithm']}:")
            print(f"ID: {data['id']}")
            print(f"Name: {data['name']}")
            print(f"Active: {data['active']}")
            print(f"Scores: {data['scores']}")
            print(f"Metadata: {data['metadata']}")
            return result
        time.sleep(1)


if __name__ == "__main__":
    # 按顺序调用子算法
    input_data = "example input for telecommunication."
    print(input_data)
    result1 = run_algorithm("sjtu_ddqn_algorithm", input_data)