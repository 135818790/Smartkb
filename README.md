# SmartKB —— 企业级智能知识库问答系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

从传统 RAG 到 Agentic RAG 的完整演进实践，解决企业文档检索中"搜不到、答不准、不可信"三大痛点。

## ✨ 核心特性

- **混合检索**：稠密向量（语义匹配）+ 稀疏向量（关键词精确命中）+ RRF 融合
- **Agentic RAG**：LangGraph 状态机编排，Router → Retrieve → Generate → Verify（Self-RAG 自检）
- **RAG 评估管线**：Faithfulness / Answer Relevancy / Context Precision / Context Recall 四项指标自动化评估
- **语义分块**：按段落边界自适应切分，块间上下文重叠，保持语义完整
- **生产就绪**：ChromaDB 持久化、懒加载、DeepSeek API 低成本运行

## 🏗️ 技术架构

```
用户界面 (Chainlit / Swagger)
      │
      ▼
FastAPI 接口层 (POST /chat)
      │
      ▼
LangGraph Agent 编排层
  Router → Retrieve → Generate → Verify
    │                              │
    │                        ┌─ pass ─┴── 返回
    │                        └─ fail ──→ 重试
    │
    ▼
混合检索引擎
  稠密路径: BGE-M3 + ChromaDB（语义）
  稀疏路径: BGE-M3 稀疏向量（关键词）
  RRF 融合 + BGE-Reranker 精排
```

## 📦 技术栈

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek-V3 |
| 嵌入模型 | BGE-M3（稠密+稀疏双输出） |
| 精排模型 | BGE-Reranker-v2-m3 |
| 向量数据库 | ChromaDB |
| Agent 框架 | LangGraph |
| API 框架 | FastAPI |
| UI | Chainlit / Swagger |
| 评估 | 自建评分引擎（LLM-as-Judge） |

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.11+
python --version

# 克隆项目
git clone https://github.com/135818790/smartkb.git
cd smartkb

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 DeepSeek API Key
# https://platform.deepseek.com/api_keys
```

### 3. 放入你的文档

```bash
# 将 .md / .txt 文件放入 data/documents/ 目录
cp /path/to/your/docs/*.md data/documents/
```

### 4. 启动后端

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8001 --reload
```

打开浏览器访问：
- **API 文档**：http://127.0.0.1:8001/docs
- **聊天界面**：先保持后端运行，再执行 `chainlit run frontend/app.py -w --port 8002`

### 5. 运行评估

```bash
python tests/eval.py tests/test_cases.json
```

## 📁 项目结构

```
smartkb/
├── src/
│   ├── api/              # FastAPI 接口层
│   ├── agent/            # LangGraph Agent 编排
│   ├── rag/               # 检索增强生成核心
│   │   ├── document_loader.py   # 文档加载
│   │   ├── chunker.py          # 语义分块
│   │   ├── embedder.py         # BGE-M3 双向量编码
│   │   ├── vector_store.py     # ChromaDB 向量存储
│   │   ├── sparse_index.py     # 稀疏向量索引
│   │   ├── reranker.py         # BGE-Reranker 精排
│   │   ├── retriever.py        # 混合检索编排
│   │   └── generator.py        # LLM 生成
│   ├── core/             # 配置管理
│   └── utils/            # 工具函数
├── frontend/             # Chainlit 聊天界面
├── tests/                # 测试用例与评估脚本
├── data/                 # 文档与向量存储
└── .github/workflows/    # CI/CD 评估自动化
```

## 📊 项目演进

| 迭代 | 内容 | 核心能力 |
|------|------|---------|
| Step 1 | 最简 RAG 教学脚本 | 理解检索→生成全链路 |
| Step 2 | FastAPI 工程化 | 模块化架构 + HTTP 接口 |
| Step 3 | ChromaDB 持久化 | 向量存硬盘，重启不丢 |
| Step 4 | 语义分块 | 长文档自适应切分 |
| Step 5 | 混合检索 | 稠密+稀疏双路召回+RRF+Reranker |
| Step 6 | Agentic RAG | LangGraph 状态机 + Self-RAG |
| Step 7 | 评估管线 | 20 条用例 + 4 项指标自动化评估 |

## 📝 开源许可

MIT License © 2026 曹骏

---

*Built with ❤️ as a learning-to-production RAG project.*
