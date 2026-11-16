# 打开 Kimi 语音输入法（Cursor Ports）

**推荐方式**：远端启动服务 → Cursor 端口转发 → 本机浏览器打开 `http://localhost:7861`。  
不要用 Pod IP（如 `10.233.x.x`）直连；Cursor 内置浏览器也常连不上远端 localhost。

## 一键重启（推荐）

在 Cursor 远端终端执行：

```bash
bash /cmy/cmy/ranAudio/switch_asr_kimi.sh kimi
```

脚本会：停 ASR（可 resume）→ 启 Kimi → 等 SenseVoice ready → 用 Cursor `openExternal` 触发 Ports 转发并尝试打开本机浏览器。

仅重启、不停其它任务：

```bash
bash /cmy/cmy/ranAudio/kimi/scripts/restart_and_open.sh
```

## 手动步骤（与脚本等价）

### 1. 启动服务（远端）

```bash
cd /cmy/cmy/ranAudio/kimi
nohup setsid bash scripts/run_demo.sh >> demo.log 2>&1 < /dev/null &
# 等到 ready
curl -s http://127.0.0.1:7861/api/ready
# 期望：{"ready": true, "asr_engine": "sensevoice", ...}
```

端口：

| 端口 | 用途 |
|------|------|
| **7861** | HTTP（本机 localhost 可录音） |
| **7862** | HTTPS（局域网 / 非 localhost 麦克风） |

### 2. Cursor 转发端口（本机 Cursor）

1. 命令面板：`Ports: Focus on Ports View`（或左侧 **Ports**）
2. 确认有 **7861**（标签 `Kimi-Audio HTTP`）  
   - 没有则点 **Forward a Port**，输入 `7861`
3. 需要麦克风且不用 localhost 时，再转发 **7862**

工作区已配置自动转发：见 [`.vscode/settings.json`](../.vscode/settings.json) 的 `remote.portsAttributes`。

### 3. 本机打开

```text
http://localhost:7861
```

麦克风（HTTPS）：

```text
https://localhost:7862
```

自签名证书：浏览器 → 高级 → 继续访问。

### 4. 触发转发的备用命令（远端）

若 Ports 里没有自动出现 7861，在远端执行（需 Cursor Remote 会话）：

```bash
bash /cmy/cmy/ranAudio/kimi/scripts/open_in_cursor.sh
```

内部调用 Cursor 的 `browser.sh --openExternal http://127.0.0.1:7861/`，让 IDE 做端口转发并打开本机浏览器。

## 状态检查

```bash
bash /cmy/cmy/ranAudio/switch_asr_kimi.sh status
curl -s http://127.0.0.1:7861/api/ready
```

## 白天切回 ASR

```bash
bash /cmy/cmy/ranAudio/switch_asr_kimi.sh asr
```

## 说明

- 页面是单输入框：空框说话 → 转写；有字再说 → 语音修改
- 错误样例：`error_samples/cases.jsonl`（区分 `asr` / `edit`）
- 更完整部署说明：[DEPLOY.md](./DEPLOY.md)

## L1 流式听写（新）

设计：[`STREAMING_ASR_DESIGN.md`](./STREAMING_ASR_DESIGN.md)  
服务（默认 **7863**）：

```bash
bash scripts/run_l1.sh fake   # UI / StableRenderer 冒烟
bash scripts/run_l1.sh qwen   # Qwen3-ASR-1.7B vLLM 流式
# Cursor Ports 转发 7863 → http://localhost:7863/
```
