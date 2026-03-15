FROM tensorflow/tensorflow:2.1.0-gpu-py3

# 基础 Python 依赖
RUN pip install -U Pillow scipy pytest

# requirements（走清华源 + 更大超时 + 重试）
COPY requirements.txt .
RUN pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --default-timeout=300 \
    --retries 10

# （可选但推荐）升级 pip/构建工具，减少老 pip 的坑
RUN pip install --no-cache-dir -U pip setuptools wheel

# 关键：先单独安装 dm-tree，并强制只用二进制 wheel（不允许源码编译）
# 如果这个版本没有 wheel，会立刻失败；那就换成 0.1.6/0.1.7 继续试
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --default-timeout=300 \
    --retries 10 \
    --only-binary=:all: \
    dm-tree==0.1.5

# 安装你本地 clone 的 cleverhans（确保仓库根目录下有 cleverhans/ 文件夹）
COPY cleverhans /opt/cleverhans
RUN pip install --no-cache-dir /opt/cleverhans

WORKDIR /dct
