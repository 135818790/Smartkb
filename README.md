# SmartKB —— 企业级智能知识库问答系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-✓-2496ED)](Dockerfile)
[![Tests](https://img.shields.io/badge/Tests-16%20passed-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

从传统 RAG 到 Agentic RAG 的完整演进实践，解决企业文档检索中"搜不到、答不准、不可信"三大痛点。

## ✨ 核心特性

### RAG 检索
- **混合检索**：稠密向量（BGE-M3 语义匹配）+ 稀疏向量（关键词精确命中）+ RRF 融合 + BGE-Reranker 精排
- **语义分块**：按段落边界自适应切分，块间上下文重叠
- **ChromaDB 持久化**：向量存硬盘，重启不丢

### Agent 编排
- **LangGraph 状态机**：Router → Retrieve → Generate → Verify
- **Self-RAG 自检**：生成后自动核对是否忠实于文档，不通过→改写查询→重新检索
- **多轮对话**：Session 管理 + 对话历史注入检索与生成

### 企业工程化
- **JWT API 鉴权**：无状态 token 验证，`/chat` 等敏感接口受保护
- **结构化日志**：JSON 格式 + trace_id 全链路追踪，兼容 ELK/Loki
- **分层异常体系**：20 个异常类覆盖模型/检索/文档/配置/API 五层
- **SSE 流式输出**：Server-Sent Events 逐字返回，用户感知首字延迟 < 500ms
- **异步文档处理**：Celery + Redis，文件上传后后台解析→分块→嵌入→入库
- **Docker 三容器部署**：API + Redis + Celery Worker，docker-compose 一键启动

### 质量保障
- **RAG 评估管线**：Faithfulness / Answer Relevancy / Context Precision / Context Recall
- **pytest 单元测试**：16 个测试用例覆盖核心模块

## 🏗️ 技术架构

```
用户界面 (Chainlit / Swagger)
      │
      ▼
FastAPI 接口层 (6 个端点)
      │
  ┌───┼───┐
  ▼   ▼   ▼
JWT  trace 异常
鉴权  日志  处理
  │   │   │
  └───┼───┘
      ▼
LangGraph Agent 编排层
  Router → Retrieve → Generate → Verify
    │                              │
    │                        ┌─ pass ─┴── 返回
    │                        └─ fail ──→ 重试
    ▼
混合检索引擎
  稠密路径: BGE-M3 + ChromaDB（语义）
  稀疏路径: BGE-M3 稀疏向量（关键词）
  RRF 融合 + BGE-Reranker 精排
      │
      ▼
DeepSeek-V3 生成回答
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
| 异步任务 | Celery + Redis |
| 鉴权 | PyJWT（HS256） |
| 日志 | python-json-logger（结构化 JSON） |
| UI | Chainlit / Swagger |
| 评估 | LLM-as-Judge（Faithfulness/Relevancy/Precision/Recall） |
| 测试 | pytest（16 用例） |
| 部署 | Docker + docker-compose（3 容器） |

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/135818790/Smartkb.git
cd smartkb
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key: https://platform.deepseek.com/api_keys
```

### 3. 放文档

```bash
# 将 .md / .txt 文件放入 data/documents/
```

### 4. 启动

```bash
# 后端
uvicorn src.api.main:app --host 127.0.0.1 --port 8001 --reload

# 前端（新终端）
chainlit run frontend/app.py -w --port 8002
```

打开浏览器：
- 聊天界面：http://127.0.0.1:8002
- API 文档：http://127.0.0.1:8001/docs

### 5. 完整部署（Docker）

```bash
docker compose up -d
# 启动 3 个容器: API + Redis + Celery Worker
```

### 6. 测试

```bash
# 单元测试
pytest tests/ -m "not integration" -v

# RAG 质量评估
python tests/eval.py tests/test_cases.json
```

## 📁 项目结构

```
smartkb/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI 入口 + JWT middleware + 异常处理
│   │   └── routes.py            # 6 个端点 (/login /chat /chat/stream /upload /status /health)
│   ├── agent/
│   │   └── graph.py             # LangGraph 状态机 (Router→Retrieve→Generate→Verify)
│   ├── rag/
│   │   ├── document_loader.py   # 文档加载
│   │   ├── chunker.py           # 语义分块
│   │   ├── embedder.py          # BGE-M3 稠密+稀疏双输出
│   │   ├── vector_store.py      # ChromaDB 向量存储
│   │   ├── sparse_index.py      # 稀疏向量索引
│   │   ├── reranker.py          # BGE-Reranker 精排
│   │   ├── retriever.py         # 混合检索编排（核心管线）
│   │   └── generator.py         # LLM 生成（流式+非流式）
│   ├── core/
│   │   ├── config.py            # 统一配置
│   │   ├── exceptions.py        # 20 个异常类（5 层体系）
│   │   └── auth.py              # JWT 签发+验证
│   ├── tasks/
│   │   ├── celery_app.py        # Celery 配置（Redis broker）
│   │   └── document_tasks.py    # 异步文档处理
│   └── utils/
│       └── logger.py            # 结构化日志 + trace_id
├── frontend/
│   └── app.py                   # Chainlit 聊天界面
├── tests/
│   ├── test_document_loader.py  # 文档加载测试
│   ├── test_chunker.py          # 分块测试
│   ├── test_retriever.py        # RRF 融合测试
│   ├── test_routes.py           # API 集成测试
│   ├── eval.py                  # RAG 评估脚本
│   └── test_cases.json          # 20 条评估用例
├── Dockerfile                   # 多阶段构建
├── docker-compose.yml           # 三容器编排
├── requirements.txt             # 16 个依赖
└── step1_naive_rag.py           # 教学脚本（80 行起点）
```

## 📊 项目演进

| 阶段 | 内容 | 核心能力 |
|------|------|---------|
| Step 1 | 最简 RAG 教学脚本 | 理解检索→生成全链路 |
| Step 2 | FastAPI 工程化 | 模块化架构 + HTTP 接口 |
| Step 3 | ChromaDB 持久化 | 向量存硬盘，重启不丢 |
| Step 4 | 语义分块 | 长文档自适应切分 |
| Step 5 | 混合检索 | 稠密+稀疏双路召回 + RRF + Reranker |
| Step 6 | Agentic RAG | LangGraph 状态机 + Self-RAG |
| Step 7 | 评估管线 | 20 条用例 + 4 项指标 |
| E1 | Docker 容器化 | Dockerfile + docker-compose |
| E2 | 结构化日志 | JSON + trace_id 全链路追踪 |
| E3 | 流式 + 多轮 | SSE 逐字输出 + Session 管理 |
| E4 | 异常体系 + 测试 | 20 异常类 + 16 pytest 测例 |
| E5 | JWT 鉴权 | Bearer Token + 公开路径白名单 |
| Celery | 异步文档处理 | 上传→后台处理→轮询进度 |

## 📝 开源许可

MIT License © 2026 曹骏
