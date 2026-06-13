# MySQL 数据库连接池配置指南

连接池最大连接数默认值为20，最小为5。当连接数超过最大值时，新请求将进入等待队列，超时时间为30秒。

可以在 config.yaml 中的 database.pool 节点修改以下参数：
- max_connections: 最大连接数（默认20）
- min_connections: 最小连接数（默认5）
- timeout: 等待超时秒数（默认30）

如果遇到 E10023 错误码，说明连接池已耗尽，需要增大 max_connections 或检查是否有慢查询占用连接。
