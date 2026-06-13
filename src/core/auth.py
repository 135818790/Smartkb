"""
JWT 鉴权 —— 无状态 token 验证
面试要点：JWT 不查数据库，适合微服务架构。Access token 短有效期，生产加 refresh token。

使用方式:
  POST /login  → {"access_token": "eyJ..."}
  POST /chat  → Header: Authorization: Bearer eyJ...
"""
import os
import time
from datetime import datetime, timezone

import jwt

# 密钥（生产环境从环境变量或密钥管理服务获取）
JWT_SECRET = os.getenv("JWT_SECRET", "smartkb-dev-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_SECONDS = 3600 * 8  # 8 小时


def create_token(username: str) -> str:
    """签发 JWT"""
    now = int(time.time())
    payload = {
        "sub": username,                              # 主体（用户名）
        "iat": now,                                   # 签发时间
        "exp": now + JWT_EXPIRE_SECONDS,              # 过期时间
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """
    验证 JWT，返回 payload。
    token 无效或过期 → 抛出异常，上层 FastAPI middleware 转 401
    """
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return payload


def get_token_from_header(auth_header: str) -> str:
    """从 Authorization: Bearer <token> 提取 token"""
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError("缺少 Authorization 头或格式错误，应为 Bearer <token>")
    return auth_header[len("Bearer "):]
