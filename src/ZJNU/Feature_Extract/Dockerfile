FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04 AS builder
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean
RUN pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r ../requirements.txt

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
COPY --from=builder /usr/bin/ /usr/bin/
COPY --from=builder /usr/lib/ /usr/lib/
COPY --from=builder /usr/include/ /usr/include/
COPY --from=builder /usr/local /usr/local/
WORKDIR /app
COPY . .
# 生成数据
# CMD ["python3", "generate.py"]
CMD ["python3", "main.py"]