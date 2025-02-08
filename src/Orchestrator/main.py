# orchestrator/main.py
import time

import redis

# 连接 Redis
r = redis.Redis(host="redis", port=6379, db=0)


def run_algorithm(algorithm_name, input_data):
    # 将任务发布到队列
    queue_name = f"{algorithm_name}_queue"
    r.rpush(queue_name, input_data)

    # 等待结果
    result_queue_name = f"{algorithm_name}_result_queue"
    while True:
        result = r.lpop(result_queue_name)
        if result:
            return result.decode("utf-8")
        time.sleep(1)


if __name__ == "__main__":
    # 按顺序调用子算法
    input_data = "example_input"
    result1 = run_algorithm("algorithm1", input_data)
    print(f"Algorithm 1 result: {result1}")

    result2 = run_algorithm("algorithm2", result1)
    print(f"Algorithm 2 result: {result2}")
