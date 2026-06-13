"""
Celery 应用配置
启动 Worker: celery -A src.tasks.celery_app worker --loglevel=info -P solo
面试说法: "Celery + Redis 做异步任务队列——文档上传后后台处理，用户不等结果。
          -P solo 是 Windows 兼容参数，Linux 用默认 prefork。"
"""
import os
from celery import Celery

# Redis 地址（本地开发用 localhost，Docker 内用 redis 服务名）
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "smartkb",
    broker=REDIS_URL,
    backend=REDIS_URL,  # 结果也存 Redis（task state 查询用）
)

# Celery 配置
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,               # 任务开始时更新状态
    task_acks_late=True,                   # 任务完成后才确认（防止 worker 崩溃丢任务）
    worker_prefetch_multiplier=1,          # 一次只取一个任务（大文件处理避免 OOM）
    task_soft_time_limit=600,              # 10 分钟软超时
    task_time_limit=900,                   # 15 分钟硬超时
    result_expires=3600,                   # 结果保留 1 小时
    broker_connection_retry_on_startup=True,
)

# 自动发现任务
celery_app.autodiscover_tasks(["src.tasks"])
