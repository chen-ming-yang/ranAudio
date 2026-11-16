# Kimi 语音输入法：流式 ASR + 可改写 设计稿

**日期**：2026-10-19  
**现状痛点**：[`demo_web.py`](scripts/demo_web.py) 用 `MediaRecorder` 整段上传 → SenseVoice **非流式**整句转写；说话过程输入框无反馈，停录后才出字，体验差。  
**目标**：边说边出字；字可改；改写链路（规则 / Kimi LLM）仍可用。  
**本文**：把「第一层流式 partial」补全为可落地的分层方案与接口约定。

---

## 0. 产品约束（不要破坏）

| 约束 | 含义 |
|---|---|
| 单输入框 | 空框说话 = 听写；有草稿再说 = 语音修改指令 |
| 规则优先 | 改写仍走 `voice_edit_lib` 级联，LLM 懒加载 |
| 书面化可选 | 转写后规则链 → 可选 spoken2written；**不得**对每个 partial 跑 LLM |
| 可切换引擎 | SenseVoice 保留为离线/对照；流式为默认听写路径 |

---

## 1. 分层总览

```text
┌─────────────────────────────────────────────────────────────┐
│ L0 音频采集                                                  │
│   AudioWorklet / ScriptProcessor → PCM 16k 帧（20–40ms）    │
│   取代「整段 MediaRecorder → 一次 POST」                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ WebSocket 二进制帧
┌──────────────────────────▼──────────────────────────────────┐
│ L1 流式 ASR（必须）                                          │
│   Qwen3-ASR-1.7B（默认；rollback 可调）或 streaming-paraformer │
│   输出 PartialHypothesis { text, stable_len, is_final }     │
│   StableRenderer：只 patch 变化区间，防光标跳动               │
│   UI：stable（正文色） / provisional（灰斜体）                │
└──────────────────────────┬──────────────────────────────────┘
                           │ endpoint / 用户松键 / is_final
┌──────────────────────────▼──────────────────────────────────┐
│ L2 句末固化 + 轻规则（听写模式）                              │
│   normalize_display → 姓名/列表/filler（与现网一致）         │
│   书面化：仅 final 后可选跑一次（禁止 per-partial）          │
└──────────────────────────┬──────────────────────────────────┘
                           │ 框内已有 stable 草稿
┌──────────────────────────▼──────────────────────────────────┐
│ L3 语音修改（编辑模式）                                       │
│   指令侧可短流式 ASR → final instruction                     │
│   apply_edit_rules →（未命中）Kimi LLM                       │
│   改写结果整体替换草稿（或 diff 高亮），不进 provisional 闪烁   │
└─────────────────────────────────────────────────────────────┘
```

第一层是体验地基；L2/L3 复用现有能力，只改**触发时机**（final 才跑重逻辑）。

---

## 2. 第一层（必须）：流式 ASR 出 partial

### 2.1 引擎选型

| 候选 | 优点 | 风险 / 代价 | 建议角色 |
|---|---|---|---|
| **Qwen3-ASR-1.7B**（开源） | 官方大档；本机已有 `/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B`；与 CDSD/Rest90 同族；支持 streaming/offline 统一推理；**rollback 可调** | 显存/延迟高于 0.6B 与 paraformer | **服务端默认**（体验机 / GPU） |
| Qwen3-ASR-0.6B | 更轻、吞吐更高 | 质量通常弱于 1.7B | 低显存备选，非默认 |
| **streaming-paraformer**（~226MB） | 端侧友好、延迟低、流式成熟（FunASR 系） | 与现网 SenseVoice 同门；质量/领域需 A/B | **端侧 / 低显存** 备选 |
| SenseVoice-Small | 已接线、质量已知 | **非流式**，只作 final 对照或编辑短指令兜底 | 对照 / 回退 |

**拍板（已定）**：

1. Demo 默认：**Qwen3-ASR-1.7B 流式**（本地权重 `Qwen3-ASR-1.7B`）。  
2. 启动参数 `--asr-engine=qwen17|qwen06|paraformer|sensevoice`；`sensevoice` = 旧整段路径。  
3. 验收：首字延迟、rollback 频率、final CER（同批录音对照 SenseVoice）。

### 2.2 假设协议：`PartialHypothesis`

服务端每推出一版假设，推一条 JSON（或同等字段的二进制头 + UTF-8）：

```jsonc
{
  "type": " partial",           // 或 "final"
  "session_id": "…",
  "text": "今天天气不错适合",    // 当前完整假设（含 provisional 尾）
  "stable_len": 6,             // 前 N 个字符已稳定，其后可 rollback
  "audio_ms": 1840,            // 已消费音频时长
  "is_final": false,
  "engine": "qwen3-asr-1.7b"
}
```

约定：

- `text[:stable_len]` = **stable**；`text[stable_len:]` = **provisional**。  
- 引擎 rollback 时：只允许改 `stable_len` 之后的内容；若必须改更早区间，应先把 `stable_len` 回退再发（客户端按新 `stable_len` 重绘）。  
- `is_final=true`：本 utterance 结束；客户端把全文升为 stable，再触发 L2。

**Rollback 可调（Qwen3-ASR-1.7B）**：暴露 `rollback_frames` / `stable_lag_ms`（例如 400–800ms 音频对应的假设才升 stable），在「少抖」与「少拖尾」之间扫参。

### 2.3 StableRenderer（客户端）

目标：用户正在看/改框内文字时，**不要**每次 partial 都 `textarea.value = fullText`（会打光标、选区、IME 组合态）。

伪代码：

```js
class StableRenderer {
  constructor(rootEl) {
    this.root = rootEl;          // contenteditable 或双 span 容器
    this.stable = "";
    this.provisional = "";
    this.userEditing = false;    // 用户正在键盘改 stable 区
  }

  onPartial({ text, stable_len, is_final }) {
    if (this.userEditing) {
      // 策略 A（推荐）：只更新 provisional 尾；stable 以用户文本为准
      // 策略 B：暂停流式渲染直到 blur
      this._setProvisional(text.slice(Math.min(stable_len, text.length)));
      return;
    }
    const nextStable = text.slice(0, stable_len);
    const nextProv = text.slice(stable_len);
    if (nextStable !== this.stable) {
      this._patchStable(this.stable, nextStable); // 按公共前缀/后缀做 DOM patch
      this.stable = nextStable;
    }
    this._setProvisional(nextProv);
    this.provisional = nextProv;
    if (is_final) this.promoteAllStable();
  }

  promoteAllStable() {
    this._setProvisional("");
    this.stable = this.stable + this.provisional;
    this.provisional = "";
    this._renderStableOnly(this.stable);
  }
}
```

DOM 结构建议（比单 `textarea` 更易分区样式）：

```html
<div id="draft" contenteditable="true" role="textbox">
  <span class="stable">今天天气不错</span><span class="provisional">适合出门</span>
</div>
```

```css
.provisional { color: #8a8a8a; font-style: italic; }
.stable { color: inherit; font-style: normal; }
```

光标策略：

- 默认光标钉在 **provisional 末尾**（听写态）。  
- 用户点进 stable 中间 → `userEditing=true`，流式只改 provisional；松键/失焦后可选「从光标后重新听写」或「合并用户改字为新 stable」。

### 2.4 传输与采集（L0）

| 现状 | 目标 |
|---|---|
| 点按录音 → 整段 webm/wav POST `/api/infer` | 按住/点按开始 → **WebSocket** 推 PCM；停录发 `end` |
| 无中间态 | 连续收 `partial` / `final` |

建议端点：

```text
WS /ws/asr?mode=dictation|edit&spoken2written=0|1
  → client: {type:"start", sample_rate:16000}
  → client: binary PCM s16le frames
  → client: {type:"end"}           // 或静音 VAD endpoint
  → server: PartialHypothesis…
  → server: {type:"final", …} 后可选再发 {type:"postprocess", text, vias…}  // L2
```

编辑模式：`mode=edit` + 首包带 `draft`（或仍用 header 语义的 JSON）；指令音频流式识别，**仅 final instruction** 进 `apply_edit_rules`。

保留 HTTP `/api/infer`、`/api/edit` 作样例/评测/回退。

### 2.5 Endpoint（何时 final）

优先级：

1. 用户松键 / 再点结束（与现网一致，最稳）。  
2. 可选客户端 VAD（webrtcvad / Silero）：静音 ≥ \(T\) ms 自动 `end`。  
3. 服务端若模型自带 endpoint，可双信号：用户未松键也可出句级 `final`，再开下一 utterance（连续听写）。

连续听写时：每句 `final` → L2 → 追加到 draft（句间加标点/空格策略另定）；下一句 provisional 接在后面。

---

## 3. 第二层：句末固化（听写）

仅在 `is_final`（或用户确认）后跑，**禁止**对 provisional 跑：

1. `normalize_display`  
2. 姓名纠字 / 列表 / filler（现网顺序）  
3. 若书面化开：`llm_spoken2written` **一次**  

结果写回：替换该 utterance 对应的 stable 段（或整框若单句模式）。UI 上书面化前后可短暂保留对照（现有 asr/written 双栏可复用）。

---

## 4. 第三层：可修改（与流式的关系）

| 场景 | 行为 |
|---|---|
| 键盘改字 | 改的是 stable；流式只动 provisional（§2.3） |
| 语音改写（框内有草稿） | 指令流式 ASR → final 文本当 instruction → 规则/LLM → **整段草稿替换**（可高亮 diff） |
| 改写中又来 partial | 忽略或排队；改写事务持锁，避免和听写抢框 |

指令侧是否流式：**短指令可流式预览灰字**，但规则引擎只吃 final，避免「把灯改成…」说到一半误触发。

---

## 5. 与现网模块对照

| 模块 | 流式改造后 |
|---|---|
| `SenseVoiceASR` | 保留；`--asr-engine=sensevoice` 整段路径 |
| `normalize_*` / `apply_edit_rules` | 不变；挂到 final / edit final |
| Kimi LLM | 仍懒加载；永不订阅 partial |
| `MediaRecorder` 路径 | 降级/样例；主路径改 PCM + WS |
| `eval_voice_edit` | 继续 HTTP；另加 streaming 延迟与 rollback 指标 |

---

## 6. 验收指标（第一层）

| 指标 | 目标（初值，可调） |
|---|---|
| 首 partial 延迟 | ≤ 300–500 ms（键下/开口后） |
| UI 更新 | 仅变更区间 DOM patch；听写态光标不无故跳到段首 |
| Rollback 率 | 稳定段被回退的次数 / 句；扫 `stable_lag_ms` |
| Final CER | 同音频 vs SenseVoice 差距可接受（先定相对阈值，如 +1–2 pp 内或持平） |
| 编辑回归 | `voice_edit_lib` self_check + oracle-text 全绿 |

---

## 7. 落地顺序（建议）

| 步 | 内容 | 产出 |
|---|---|---|
| P0 | WS + PCM 采集 + 假 ASR（回显渐长字符串） | StableRenderer / 灰斜体可用 |
| P1 | 接入 **Qwen3-ASR-1.7B** 流式（主）；paraformer / 0.6B 可选 | 对照 SenseVoice CER |
| P2 | `stable_len` + rollback 参数扫描 | 定默认 lag |
| P3 | final → 现网 L2 规则链 | 与现体验能力对齐 |
| P4 | edit 模式指令流式 + 规则/LLM | 完整「可改」 |
| P5 | 连续听写 VAD / 多 utterance | 增强体验 |

### 7.1 L1 已实现（可跑）

| 文件 | 作用 |
|---|---|
| [`scripts/qwen_streaming_engine.py`](scripts/qwen_streaming_engine.py) | Qwen3-ASR-1.7B vLLM 流式 → `PartialHypothesis`（含 `stable_len`） |
| [`scripts/l1_server.py`](scripts/l1_server.py) | FastAPI：`/` StableRenderer 页 + `WS /ws/asr` + `/api/ready` |
| [`scripts/run_l1.sh`](scripts/run_l1.sh) | 启动脚本（`qwen` / `fake`） |
| [`scripts/smoke_l1.py`](scripts/smoke_l1.py) | 无 GPU 自检 |

```bash
# UI 冒烟（无 GPU）
bash scripts/run_l1.sh fake
# 真模型（需 wenet-asr-compare + 空闲显存；默认端口 7863）
bash scripts/run_l1.sh qwen
# 浏览器：Cursor Ports 转发 7863 → http://localhost:7863/
```

环境：`PYTHON=/cmy/cmy/miniconda3/envs/wenet-asr-compare/bin/python`（含 vLLM）。  
注意：官方流式 **仅 vLLM backend**；与 Rest90 FireRed 同卡时注意显存（`--gpu-mem` 默认 0.35）。  
依赖：系统需有 `gcc`（vLLM/triton 会编译小内核）；`run_l1.sh` 已 `export CC/CXX`。加载参数默认 `enforce_eager=True`、`max_model_len=8192`。

---

## 8. 第一层条文（定稿口径，可直接贴进评审）

**第一层（必须）：流式 ASR 出 partial**

1. 引擎：默认 **Qwen3-ASR-1.7B**（rollback 可调；本地 `/cmy/cmy/Qwen3-ASR/Qwen3-ASR-1.7B`）；端侧/低显存备选 **streaming-paraformer（~226MB）** 或 **0.6B**；SenseVoice 仅整段回退。  
2. 协议：持续输出 `text` + `stable_len` + `is_final`；稳定前缀不可随意改写，rollback 须显式收缩 `stable_len`。  
3. 渲染：**StableRenderer** 只 patch 变化区间，禁止整框赋值导致光标跳动。  
4. UI：**stable** 正文样式，**provisional** 灰斜体；用户编辑 stable 时流式仅更新 provisional。  
5. 重逻辑（规则 / 书面化 / 语音改写）只消费 **final**，不消费 partial。

---

## 9. 待决（OPEN）

- [ ] Qwen3-ASR-1.7B 流式 runtime（官方 streaming toolkit / vLLM / 自研 sliding window）接线确认  
- [ ] 1.7B 首字延迟是否达标；不达标再降级 0.6B / paraformer  
- [ ] `contenteditable` vs 双层 textarea 兼容性（移动端）  
- [ ] 连续听写句间标点是否由模型出，还是客户端插入  

---

**【完】**
