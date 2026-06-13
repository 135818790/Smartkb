# SmartKB Dockerfile
# 多阶段构建：builder 阶段安装依赖，runtime 阶段运行
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app


# 从 builder 复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY tests/ ./tests/
COPY data/documents/ ./data/documents/
COPY .env.example .

# 模型下载走 ModelScope（国内），HuggingFace 备选
ENV MODELSCOPE_CACHE=/app/models

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
