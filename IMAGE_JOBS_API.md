# 异步图片任务接口

针对长耗时生图（>100s 容易被 CDN/反代腰斩）提供的异步版本。请求体和同步接口完全一致，只是把 URL 前缀从 `/v1/images/...` 改成 `/v1/jobs/images/...`，提交后立即拿到 `job_id`，再通过 GET 轮询拿结果。

鉴权方式和现有接口一致（`Authorization: Bearer <api_key>`），用同一把 key。

## 接口

```
POST  /v1/jobs/images/generations    提交文生图，请求体同 /v1/images/generations
POST  /v1/jobs/images/edits          提交图生图，请求体同 /v1/images/edits（multipart 也支持）
GET   /v1/jobs/{job_id}              查询任务
```

> **注意**：异步接口下不要带 `stream=true`，提交期会被 400 拒绝。

## 提交（202 Accepted）

```json
{
  "job_id": "job_3f8b...",
  "status": "queued",
  "created_at": 1731600000
}
```

## 轮询

每 2–3 秒 GET 一次（响应里也会带 `Retry-After: 3` 头）。状态机：`queued` → `processing` → `succeeded` / `failed`。

未完成：

```json
{ "job_id": "job_3f8b...", "status": "processing", "created_at": 1731600000 }
```

成功：直接返回标准 images API 响应字段（`data`、`usage`、`size` 等），并附加包装字段 `job_id` / `status` / `created_at` / `completed_at`：

```json
{
  "job_id": "job_3f8b...",
  "status": "succeeded",
  "created_at": 1731600000,
  "completed_at": 1731600045,
  "data": [{ "b64_json": "...", "revised_prompt": "..." }],
  "size": "1024x1024",
  "usage": { ... }
}
```

失败：

```json
{
  "job_id": "job_3f8b...",
  "status": "failed",
  "created_at": 1731600000,
  "completed_at": 1731600007,
  "http_status": 502,
  "error": { "type": "upstream_error", "message": "..." }
}
```

不存在或不属于当前 key：`404 not_found_error`。

## 行为约定

- **结果一次性**：GET 拿到 `succeeded` / `failed` 后服务端立即删除该任务，不能重复 GET。
- **TTL 默认 10 分钟**：超时未取的任务会被自动清理，再 GET 是 404。
- **计费规则与同步接口一致**：上游成功出图就计费；客户端"不来取" / "断开"不影响扣费。失败不计费。
- **总超时 5 分钟**：单个任务上游超过 5 分钟未完成会被标 `failed`。

## curl 示例

```bash
# 提交
JOB=$(curl -s -X POST https://your.host/v1/jobs/images/generations \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-image-2","prompt":"a cyberpunk garfield in primeval forest","size":"2048x2048"}' \
  | jq -r .job_id)

# 轮询
while :; do
  RESP=$(curl -s -H "Authorization: Bearer $KEY" "https://your.host/v1/jobs/$JOB")
  STATUS=$(echo "$RESP" | jq -r .status)
  case "$STATUS" in
    succeeded) echo "$RESP" | jq -r '.data[0].b64_json' | base64 -d > out.png; break;;
    failed)    echo "$RESP" >&2; exit 1;;
    *)         sleep 3;;
  esac
done
```