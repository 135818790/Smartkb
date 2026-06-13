"""
SmartKB —— 企业级智能知识库问答系统 API 入口
Author: 曹骏
启动: uvicorn src.api.main:app --reload
"""
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="SmartKB", description="企业级智能知识库问答系统 - 从RAG到Agentic RAG的完整演进", version="0.1.0")
app.include_router(router)
