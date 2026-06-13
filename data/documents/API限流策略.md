# API 限流策略说明

系统对API请求实施基于令牌桶算法的限流。

默认配置：
- 每秒处理100个请求（rate: 100/s）
- 突发峰值允许到200个（burst: 200）

超过限流阈值时，API返回429状态码，响应体中包含Retry-After头指示重试时间。

限流配置可在 config.yaml 的 rate_limit 节点修改以下参数：
- rate: 每秒请求数
- burst: 突发峰值
- retry_after: 重试等待时间（秒）

注意：修改配置后需要重启API网关才能生效。
