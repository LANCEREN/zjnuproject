

222

# 深度学习服务项目

## 项目概述

本项目是一个包含多个不同依赖环境子算法的深度学习服务。每个子算法可能有不同的依赖和环境要求。我们使用 Docker 来管理这些不同的环境，以确保一致性和可重复性。

## 项目结构

```

├── Analog_Propagation
│   ├── data
│   ├── Finetuning_GAT
│   │   ├── data
│   │   │   └── test
│   │   └── output
│   └── output
├── Feature_Extract
│   ├── data
│   ├── enums
│   ├── out
│   └── utils
├── Influence_Prediction
│   ├── GAT
│   └── LINE
├── IMPIM_SJTU                  # 包含各个子算法的目录
│   └── DDQN                    # 子算法1
│       ├── data
│       │   ├── data_sansuo
│       │   ├── data_sjtu
│       │   │   └── subgraphs
│       │   └── data_zjnu
│       ├── output
│       │   └── weights
│       ├── dockerfile          # 子算法1的 Dockerfile
│       └── requirements.txt    # 子算法1的 Python 依赖列表
├── data_structure.py
├── DDQN_Interface.py
├── docker-compose.yml          # Docker Compose 文件，用于管理多个服务
├── readme.MD                   # 项目说明文件
└── zjnu_main.py
```


## 环境要求

- Docker: 20.10.x 或更高版本，SJTU采用27.5.0
- Docker Compose: 1.29.x 或更高版本，SJTU采用2.32.4

## 构建运行

### 1. 克隆项目

```bash
git clone https://git.code.tencent.com/zr197662012/zjnuproject.git
cd zjnuproject
```


### 2. 构建docker镜像

方法一：**docker build指令构建各个子算法镜像**

```bash
docker build -t sjtu_algorithm_ddqn -f IMPIM_SJTU/DDQN/Dockerfile .
docker build -t zjnu_algorithm_FE -f ZJNU/Feature_Extract/Dockerfile .
```

   注：该命令应在主目录下执行，这将根据 `Dockerfile` 文件中的配置，构建子算法服务。`Dockerfile` 中应包含基础环境配置，如 Python 版本、常用库等。

方法二：**docker-compose build指令构建各个子算法镜像** 

   ```bash
   docker-compose build sjtu_algorithm_ddqn
   docker-compose build zjnu_algorithm_FE
   ```

   注：该命令应在主目录下执行，这将根据 `docker-compose.yml` 文件中的配置，索引到子算法的`Dockerfile`文件去构建子算法docker服务。

### 3. 使用 Docker Compose 启动服务

```bash
docker-compose up -d
```

这将根据 `docker-compose.yml` 文件中的配置，启动所有子算法服务。


### 4. 访问服务

根据 `docker-compose.yml` 中的配置，通过docke相关指令访问各个子算法docker容器服务。


## 详细说明

### 1. IMPIM_SJTU/DDQN/**
各单位的各个算法应当以此架构统一组织
- **IMPIM_SJTU** 文件夹：以项目标识+单位缩写命名的文件夹，其中包含该单位开发的各个子算法文件夹。
- **DDQN** 文件夹：以算法缩写命名的文件夹，其中包含DDQN算法的代码和 Dockerfile, requirements.txt。

### 2. requirements.txt
每个子算法文件夹应当根据自己的依赖环境列出所需依赖包和版本号
**并确保可以用类似下发的指令安装无冲突**
```bash
pip3 install --no-cache-dir -r ./IMPIM_SJTU/DDQN/requirements.txt
```

### 3. Dockerfile
每个子算法文件夹应当根据自己的依赖环境构筑自己的dockerfile
- **IMPIM_SJTU/DDQN/Dockerfile**: 用于构建子算法1的镜像，并安装子算法1的特定依赖。
- **ZJNU/Feature_Extract/Dockerfile**: 用于构建子算法2的镜像，并安装子算法2的特定依赖。

```dockerfile
# 使用官方 Python 镜像
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04 AS builder

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY IMPIM_SJTU/DDQN/requirements.txt ./IMPIM_SJTU/DDQN/requirements.txt

# 安装依赖
RUN pip3 install --no-cache-dir -r ./IMPIM_SJTU/DDQN/requirements.txt
# 安装 PyTorch 和 torchvision
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# 安装 PyTorch 相关库
RUN pip3 install torch-scatter==2.0.7 torch-sparse==0.6.9 torch-cluster==1.5.9 torch-spline-conv==1.2.2 torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html

# 阶段 2：运行环境
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

# 设置工作目录
WORKDIR /app

# 从构建阶段复制系统安装的依赖
COPY --from=builder /usr/bin/ /usr/bin/
COPY --from=builder /usr/lib/ /usr/lib/
COPY --from=builder /usr/include/ /usr/include/
COPY --from=builder /usr/local /usr/local/

# 复制应用代码
COPY IMPIM_SJTU/DDQN/ ./IMPIM_SJTU/DDQN/
COPY data_structure.py ./data_structure.py
COPY DDQN_Interface.py ./DDQN_Interface.py
COPY zjnu_main.py ./zjnu_main.py  

# 设置默认命令
CMD ["python3", "DDQN_Interface.py"]
```

### 4. docker-compose.yml

`docker-compose.yml` 文件只有一个并且位于最顶层文件夹，用于根据各个子算法文件夹中的`dockerfile`文件定义和运行多个 Docker 容器。每个子算法作为一个独立的服务运行。

```yaml
version: '3'
services:
  sjtu_algorithm_ddqn:
    build:
      context: .
      dockerfile: IMPIM_SJTU/DDQN/Dockerfile
    container_name: sjtu_algorithm_ddqn
    environment:
      - NVIDIA_VISIBLE_DEVICES=all       # 所有 GPU 对容器可见
      - NVIDIA_DRIVER_CAPABILITIES=all   # 启用所有 NVIDIA 驱动功能 
    volumes:
      - ./IMPIM_SJTU/DDQN:/app/IMPIM_SJTU/DDQN      # 开发时挂载代码,方便开发时实时更新代码（生产环境可移除）
      - ./data_structure.py:/app/data_structure.py
      - ./DDQN_Interface.py:/app/DDQN_Interface.py
      - ./zjnu_main.py:/app/zjnu_main.py  
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # 指定需要 GPU
              count: 1             # 至少需要一块 GPU
    stdin_open: true # docker run -i
    tty: true        # docker run -t
    command: /bin/bash  # 运行 nvidia-smi 命令 
    profiles:
      - development

  zjnu_algorithm_FE:    # zjnu开发容器配置举例（需自己修改本文件和子算法中的dockerfile文件）
    build:
      context: .
      dockerfile: Feature_Extract/Dockerfile
    container_name: zjnu_algorithm_FE
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./Feature_Extract:/app/Feature_Extract      # 开发时挂载代码,方便开发时实时更新代码（生产环境可移除）
      - ./data_structure.py:/app/data_structure.py
      - ./DDQN_Interface.py:/app/DDQN_Interface.py
      - ./zjnu_main.py:/app/zjnu_main.py  
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # 指定需要 GPU
              count: 1             # 至少需要一块 GPU
    stdin_open: true # docker run -i
    tty: true        # docker run -t
    command: /bin/bash  # 运行 nvidia-smi 命令 
    networks:
      - algo_net
    profiles:
      - development

```



## 开发指南

1. **添加新子算法服务**：
   - 在各自单位的目录（如交大是`IMPIM_SJTU`）目录下创建新的子算法目录（如 `DDQN`）。
   - 在新目录中添加 `Dockerfile` 和 `requirements.txt` 文件，配置服务依赖。
   - 更新 `docker-compose.yml` 文件，添加新服务的配置。

2. **修改现有服务**：
   - 进入相应服务目录，修改 `Dockerfile` 或 `requirements.txt` 文件。
   - 返回 `docker-compose.yml` 文件所在目录
   - 重新构建镜像并启动服务：

     ```bash
     docker-compose build <service_name>
     docker-compose up -d <service_name>
     ```

### 数据管理

<!-- 所有数据应放置在 `data/` 目录下，Docker 容器将通过卷挂载访问这些数据。 -->

### 模型管理

<!-- 训练好的模型应放置在 `models/` 目录下，Docker 容器将通过卷挂载访问这些模型。 -->


## 常见问题

1. **依赖库版本冲突**：
   - 每个子算法服务都有独立的 Docker 镜像，可以隔离依赖环境，避免版本冲突。

2. **服务无法启动**：
   - 检查 `docker-compose.yml` 文件中的配置是否正确。
   - 查看 Docker 容器日志，定位问题：

     ```bash
     docker logs <container_id>
     ```

## 贡献指南

欢迎贡献代码、提出问题和建议。请通过 腾讯工蜂 提交贡献，并遵循以下规范：

- 确保代码风格和组织架构一致。
- 创建子算法或单位分支，并在该分支上开发，请不要在主分支上开发。
- 提交前请运行测试，确保功能正常。
- 更新 `README.md` 文件，反映最新变化。


在贡献代码时，请遵循以下步骤：

1. 克隆项目
2. 拉取最新主分支 (`git pull origin master`)
3. 创建或切换到新的分支 (`git checkout -b feature/YourFeatureName`)
4. 提交更改 (`git commit -am 'Add some feature'`)
5. 推送到自己的子分支 (`git push origin feature/YourFeatureName`)
6. 切换到主分支，然后在确保主分支为最新的情况下将其他分支的版本合并到主分支并推送到远程仓库
```bash
git checkout master
git pull origin master
# --no-ff采用非快速合并方法合并代码，方便后续开发
git merge --no-ff -m "fix bugs." feature/YourFeatureName
git push origin master
```
7. **解决冲突**（如果有）：如果在合并过程中发生冲突，Git会暂停合并，并提示你手动解决冲突。你可以使用文本编辑器打开包含冲突的文件，手动解决冲突部分，然后保存文件。解决完所有冲突后，使用git add命令将解决后的文件标记为已解决，最后使用git commit命令提交合并结果。例如：
```bash
git status  # 查看冲突文件列表
# 手动编辑冲突文件，解决冲突
git add <冲突文件>  # 标记冲突文件为已解决
git commit -m "解决冲突并合并主分支代码"  # 提交合并结果
```
如对git操作缺乏基础，可以参考学习以下链接：
[Git教程](https://liaoxuefeng.com/books/git/branch/policy/index.html)
![示例](https://liaoxuefeng.com/books/git/branch/policy/branches.png)

## 联系我们

如有任何问题或建议，请通过以下方式联系我们：

- 邮箱：example@example.com

## 许可证

本项目采用 [MIT 许可证](LICENSE)。



