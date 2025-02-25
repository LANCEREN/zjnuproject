# 使用官方 Python 镜像
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    # python3-dev \
    # git \
    # build-essential \
    # cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 验证 Python 是否可用
RUN which python3
RUN python3 --version

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY src/ZJNU/requirements.txt ./src/ZJNU/requirements.txt
# 安装依赖
RUN pip3 install --no-cache-dir -r ./src/ZJNU/requirements.txt -v
# 安装 PyTorch 和 torchvision
RUN pip3 install torch torchvision torchaudio
# 安装 PyTorch 相关库
RUN pip3 install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
# 阶段 2：运行环境
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# 设置工作目录
WORKDIR /app

# 从构建阶段复制系统安装的依赖
COPY --from=builder /usr/bin/ /usr/bin/
COPY --from=builder /usr/lib/ /usr/lib/
COPY --from=builder /usr/include/ /usr/include/
COPY --from=builder /usr/local /usr/local/

# 验证 Python 是否可用
RUN which python3
RUN python3 --version

# 复制应用代码
COPY src/ ./src/
COPY __main__.py ./__main__.py
COPY main1.py ./main1.py
COPY main2.py ./main2.py
# 设置默认命令
# CMD ["python3", "src/SJTU/DDQN/redis_main.py"]
# CMD ["python3", "__main__.py"]
#CMD ["/bin/bash"]
#CMD ["python --version"]
#CMD ["python3", "--version"]
#CMD ["python3", "__main__.py"]
#CMD ["python3", "main1.py"]