# AI 合同系统镜像（考核 G 的 Docker 化）
FROM python:3.12-slim

# OpenCV（cv2）运行所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 层缓存，代码变更不用重装依赖）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷业务代码与测试数据
COPY app ./app
COPY data ./data
COPY scripts ./scripts

# 沙箱友好：不写字节码缓存
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# 启动前写入向量库范例，再起服务
CMD ["sh", "-c", "python -B scripts/seed_examples.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
