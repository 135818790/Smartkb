"""
统一配置管理 —— 所有环境变量和路径集中在一个地方
面试要点：配置集中管理是工程化第一步，避免 API Key 散落各处
"""
import os
from pathlib import Path

# 加载 .env 文件（本地开发用）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # 生产环境通过系统环境变量注入

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 嵌入模型
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_MODEL_DIR = str(PROJECT_ROOT / "models" / "BAAI" / "bge-m3")

# 文档
DOCUMENTS_DIR = str(PROJECT_ROOT / "data" / "documents")

# 检索
TOP_K = 3          # 最终返回几篇文档给 LLM
HYBRID_TOP_K = 20  # 混合检索初筛候选数（给 Reranker 精排用）
RRF_K = 60         # RRF 平滑常数

# Reranker 模型
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_DIR = str(PROJECT_ROOT / "models" / "BAAI" / "bge-reranker-v2-m3")

# 文档分块
CHUNK_SIZE = 500   # 每块目标字符数
CHUNK_OVERLAP = 50  # 相邻块重叠字符数（保证上下文连贯）

# ChromaDB 向量存储
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "data" / "vector_store")
CHROMA_COLLECTION_NAME = "smartkb_docs"
