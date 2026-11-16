# 语义转写与修正能力总览

实现主文件：[`scripts/voice_edit_lib.py`](scripts/voice_edit_lib.py)  
接线：[`scripts/demo_web.py`](scripts/demo_web.py)（SenseVoice ASR → 规则 → 可选 Kimi LLM）  
评测分桶：[`eval_voice_edit/schema.md`](eval_voice_edit/schema.md)  
问题与修法记录：[`eval_voice_edit/ISSUES.md`](eval_voice_edit/ISSUES.md)

产品形态：**单输入框** — 空框说话 → 转写；有字再说 → 语音修改。

---

## 一、转写后自动处理（用户不说修改指令也会做）

| 类型 | via / 函数 | 做什么 | 例 |
|------|------------|--------|----|
| **显示归一** | `normalize_display` | 中文数字、时间、百分比、口语算式、符号词 → 阿拉伯数字/符号 | `拨打一二零`→`拨打120`；`一加一等于二`→`1 + 1 = 2`；`井号`→`#` |
| **姓名纠字** | `rule_name_clarify` / `resolve_name_clarify` | 句内「X是…的X」（组词/偏旁）纠正姓名与说明字；偏旁口误别名（铁手旁→提手旁、羊→扬） | `我叫陈明阳，名是名字的名，杨是提手旁的杨`→`…陈名扬…扬是提手旁的扬` |
| **列表结构** | `rule_list_*` / `normalize_list_structure` | 口语枚举 → `1.` `2.` `3.` 换行列表（一是/第一/顿号菜名/半编号） | `下面有三点，一是开会，二是吃饭`→带序号列表 |
| **去填充词** | `rule_filler` / `strip_fillers` | 剥高置信停顿填充（嗯/呃/那个…）；**不折叠**故意叠词 | `嗯，打开窗帘`→`打开窗帘`；`好好好好`不变 |
| **书面化（可选）** | `llm_spoken2written` | 参数开时转写后默认 LLM 口语→书面 | 见下节 |

姓名纠字规则失败时才懒加载 Kimi；列表规范化与去填充词**只用规则**。书面化见参数。

---

## 一-b、书面化参数（口语→书面 / CS2W 风格）

| 开关 | 说明 |
|------|------|
| 环境变量 `KIMI_SPOKEN2WRITTEN=1` 或 `--spoken2written` | 服务端默认开 |
| 请求 `?spoken2written=1` / header `X-Spoken2Written` | 覆盖默认 |
| 页面勾选「书面化」 | 转写请求带上；`localStorage` 记住 |

- **关（默认）**：自动只做 filler；用户再说「书面化一下 / 改成书面语…」才走 LLM。
- **开**：转写后规则链结束 → 默认 LLM 语句规范化（去填充、口误重复、语法/口语词；**故意强调重复保留**）。
- **不做自动叠词折叠**（避免误伤「好好好好」）。
- 对照 [CS2W](https://github.com/guozishan/CS2W)：filler≈规则；R-type/语法/口语词≈LLM；评测默认 **val**，**test 封存**。

```bash
KIMI_SPOKEN2WRITTEN=1 bash scripts/run_demo.sh
python scripts/eval_spoken2written.py          # 默认 val
# python scripts/eval_spoken2written.py --split test  # 仅最终 held-out
```

---

## 二、语音修改指令（框内已有草稿时）

规则级联（`apply_edit_rules`），命中则不调 LLM：

```
列表规范化指令 → noop 无关指令 → toggle → 偏旁替换 → 精确替换 → 模糊替换 →（否则）Kimi LLM
```

另：指令命中「书面化…」时跳过上列，直接 `llm_spoken2written`。

| 类型 | via | 评测 bucket | 做什么 | 例 |
|------|-----|-------------|--------|----|
| **书面化指令** | `llm_spoken2written` | — | 口语→书面（同参数开时的 LLM） | 「书面化一下」 |
| **列表规范化指令** | `rule_list_*` / `noop` | — | 指令含「规范化/加上序号/整理成列表…」时对草稿做列表结构改写 | 「加上序号」 |
| **无关指令保留** | `noop` | `noop` | 指令与草稿无关则原样保留，避免误改 | 草稿「打开窗帘」+「今天天气怎么样」 |
| **开/关对立** | `rule_toggle` | `toggle` | 打开↔关闭、开始↔停止、锁屏↔解锁等 | 「改成关闭」 |
| **偏旁/拆字纠错** | `rule_radical` | `glyph` | 双侧偏旁、只描述目标、拆字口诀 | 「把白字旁的的改成提土旁的地」 |
| **单/多句替换** | `rule` | `replace` / `multi` | `把A改成/变成/换成B`，支持多对 | 「把灯改成空调，把窗帘改成窗户」 |
| **近邻模糊替换** | `rule_fuzzy` | `asr_fix` | ASR 把源字听偏时，在草稿找近邻再换 | 指令 src 与草稿差 1～2 字 |
| **LLM 兜底** | `llm` | `paraphrase` / `delete` / `insert` / `hard` 等 | 规则未命中时由 Kimi 改写 + `clean_rewrite` 清洗泄漏/复读 | 删词、插词 |

---

## 三、后处理与门禁（改写结果清洗）

| 能力 | 函数 | 说明 |
|------|------|------|
| 剥提示泄漏 / 截断复读 | `clean_rewrite` | 去掉 `【改写结果】` 等；打断 A B A B 循环 |
| 比分归一 | `norm_text` | 去空白与常见标点再比 exact/CER |
| 回声失败检测 | `is_echo_fail` | 预测几乎等于指令本身 → 失败 |

---

## 四、评测分桶 vs 实现状态

| bucket | 含义 | 规则侧 | 状态（见 ISSUES） |
|--------|------|--------|-------------------|
| `replace` | 单替换 | `rule` | 稳 |
| `multi` | 多替换 | `rule` | 稳 |
| `toggle` | 开/关 | `rule_toggle` | 稳 |
| `noop` | 无关指令 | `noop` | 稳 |
| `glyph` | 偏旁纠错 | `rule_radical` | 稳 |
| `asr_fix` | 听错近邻 | `rule_fuzzy` | 多数好 |
| `num` | 数字显示 | `normalize_display` | 稳（转写侧） |
| `insert` | 插入 | 多靠 LLM | 清洗后可用 |
| `delete` | 删除 | 弱 | 待加强 |
| `paraphrase` | 换说法 | 不规则化 | 待评测口径/LLM |
| `hard` | 难例 | LLM | 少量 |

姓名纠字、列表规范化目前主要靠产品链路 + `voice_edit_lib` self_check，尚未全部进 `cases.jsonl` 分桶。

---

## 五、相关文档索引

| 文档 | 内容 |
|------|------|
| **本文** `CAPABILITIES.md` | 能力类型总览 |
| [`OPEN.md`](OPEN.md) | Cursor Ports 打开方式 |
| [`DEPLOY.md`](DEPLOY.md) | 部署与启动 |
| [`eval_voice_edit/ISSUES.md`](eval_voice_edit/ISSUES.md) | 评测问题、修法、门禁 |
| [`eval_voice_edit/schema.md`](eval_voice_edit/schema.md) | 题集字段 |
| [`eval_voice_edit/reports/latest.md`](eval_voice_edit/reports/latest.md) | 最近一次 HTTP 评测 |
| [`eval_spoken2written/`](eval_spoken2written/) | CS2W 离线评测（默认 val） |

回归：

```bash
cd /cmy/cmy/ranAudio/kimi
python scripts/voice_edit_lib.py          # 规则自检
python scripts/eval_spoken2written.py     # CS2W val（勿用 test 调参）
bash scripts/run_voice_edit_eval.sh       # 端到端（需 demo ready）
```
