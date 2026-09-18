# MiniCPM5-2B Function Call：输出格式与触发方式

**日期**：2026-09-13  
**范围**：裸 2B 文本侧实测（无 LoRA、无音频 encoder）。对照 v1.0 XML 基线与 S2 路由（`<|FC|>` / `<|NOFC|>`）。  
**复现**：[`tmp/probe_minicpm5_fc.py`](../tmp/probe_minicpm5_fc.py) → [`tmp/minicpm5_fc_probe.json`](../tmp/minicpm5_fc_probe.json)

---

## 1. 结论

1. 底座**已经会**出 function call，格式是官方 XML，不是 MiniCPM-o 双工的 `<tool_call>{json}</tool_call>`。
2. **真正的开关**是 `apply_chat_template(..., tools=schema列表)`。template 把 22 个工具写进 `<tools>`，并规定调用时输出 `<function name=…>`。
3. 自定义 system（「只输出 function call」）不是必需的；有 `tools=` 就够出 XML。没有 `tools=` 时，模型仍可能乱吐 `<function`，或退回普通闲聊。
4. 词表里**没有** `<|FC|>` / `<|NOFC|>`。S2 路由 token 要自己加。本地执行命令应复用现成的 `<function`…`</function>`，不要另造一套。
5. 听写 / 闲聊不是底座的原生能力：同一套 `tools=` 下，「请转写：…」会被误打成某个工具。

---

## 2. 词表里已有的标签

`models/MiniCPM5-2B` 的 special token（与 FC 相关）：

| token | id | 本次裸 2B 是否吐出 |
|---|---|---|
| `<function` | 18 | **是**（命令的首 token） |
| `</function>` | 19 | 是 |
| `<param` | 20 | 有参工具时 |
| `</param>` | 21 | 有参工具时 |
| `<tool_call>` / `</tool_call>` | 2 / 3 | 否 |
| `<tools>` / `</tools>` | 12 / 13 | 只出现在 **prompt**，不在生成 |
| `<tool_response>` / `</tool_response>` | 10 / 11 | 否 |
| `<|tool_call|>` | 130077 | 否 |
| `<|FC|>` / `<|NOFC|>` | — | **词表里没有** |

开标签是半截的 `<function` / `<param`，后面的 ` name="foo">` 是普通 BPE。  
`generate` / decode 必须 `skip_special_tokens=False`，否则只剩 `name="enable_bluetooth">`，解析会断（v1.0 已踩过）。

官方 chat template（[`chat_template.jinja`](../../MiniCPM-o-Demo/models/MiniCPM5-2B/chat_template.jinja)）写明的调用体：

```xml
<function name="function-name"><param name="param-name">param-value</param></function>
```

无参：

```xml
<function name="enable_bluetooth"></function>
```

收尾再跟 `<|im_end|>`。ranAudio 解析：[`tools.py`](../ranAudio/tools.py) 的 `parse_tool_calls` / `format_tool_xml`。

---

## 3. 目前如何触发

代码路径：[`model.py`](../ranAudio/model.py) 的 `_apply_chat` → `tokenizer.apply_chat_template`。

```text
messages = [
  {role: system, content: fc_system_prompt() 或空},
  {role: user,   content: 文本 或 <|audio|> pads},
]
prompt = apply_chat_template(messages, tools=load_tools(), add_generation_prompt=True)
llm.generate(...)
```

- `tools=` 传入 22 个 OpenAI 风格 schema（[`finetune_fc/tools/*.json`](../../MiniCPM-o-Demo/finetune_fc/tools)）。
- template **有 tools** 时：在 system 段追加 `# Tools` + `<tools>` JSON 列表 + 英文 guideline（「不需要就正常回答；需要则输出 `<function …>`」）。
- template **无 tools** 时：不插入工具块。
- 默认 system（[`fc_system_prompt`](../ranAudio/tools.py)）：「只输出 function call，不要寒暄」。有 `tools=` 时这句可有可无；没 `tools=` 时它会逼模型编造工具名。
- 推理入口：`generate_from_text`（纯文本）、`generate_text`（语音 pads）。v1.0 再叠 `runs/fc` LoRA。

`enable_thinking=False` 时，template 仍可能在 assistant 前插入空的 `<think></think>`，不影响随后吐 XML。

---

## 4. 裸 2B 实测（2026-09-13）

greedy，`max_new_tokens=256`，22 工具 schema。用户句是**文本**，不是 wav。

| 条件 | 用户 | 输出（摘要） | 解析 |
|---|---|---|---|
| `tools=` + 车机 system | 打开蓝牙 | `<function name="enable_bluetooth"></function>` | PASS |
| **只有 `tools=`**，无自定义 system | 打开蓝牙 | 同上 | PASS |
| `tools=` + system | 导航去公司 | `navigate_to`，`destination=company`（英文，不是「公司」） | 名对、槽漂 |
| 只有 system，**不传 tools** | 打开蓝牙 | `<function name="bluetooth_device_list">`（编造） | 形状对、名错 |
| 无 tools、无 system | 打开蓝牙 | 中文教用户怎么开蓝牙 | 无 FC |
| `tools=` + system | 今天天气怎么样 | 「我无法获取实时天气…」 | 正确不上工具 |
| `tools=` + system | 请转写：今天会议取消了 | `<function name="answer_call">` | **误触发** |

要点：

- **出不出 XML，看有没有把 schema 喂进 template。**
- **出哪个工具，看用户话 + schema 描述**；没列表时会幻觉名字。
- 槽位不稳（公司 → `company`）。语音 22/22 是 v1.0 LoRA，不是底座默认就会。
- 「请转写」在底座里不是理解任务，会被现有工具吸走。S2 必须把听写标成 `<|NOFC|>`，不能指望不训就对。

首 token 实例（「打开蓝牙」）：

```text
18 <function>  2546  name  943 ="  26346 enable  …  19 </function>  130073 <|im_end|>
```

对照 tokenize：同一句 XML 的 `<function` / `</function>` 就是这两个 id。`<tool_call>\n{"name":…}` 能编进词表（id 2/3），但 **greedy 不会走这条**。

---

## 5. 和现有版本的关系

| 版本 | 2B 吐什么 | 和本次实测 |
|---|---|---|
| v1.0 | 只要 XML FC | 与底座格式一致；LoRA 把语音也锁在这条路上 |
| v1.1 | JSON：`asr` + `need_fc`（+ `reply`） | 底座**不会**自发这套；要另训 |
| S2 本版 | `<|FC|><|BASE|>` + 转写，或 `<|NOFC|>` + 理解 | token **要新增**。本版不上 XML。完整计划：[S2_FC_NOFC_训练计划.md](./S2_FC_NOFC_训练计划.md) |

S2 语义（已拍板）：

- `<|FC|>` = **命令**。本版第二 token 固定 `<|BASE|>`，后面是转写，然后上抛。
- `<|LOCAL|>` 只进词表，本版不训。以后简单命令才复用本节 `<function>` XML。
- `<|NOFC|>` = **理解**（闲聊 / 听写）。听写不是 FC。
- 光杆「请转写我说的话」→ NOFC 引导；「请转写：今天会议取消了」→ `<|NOFC|>今天会议取消了`。

---

## 6. 实现注意

- 解析优先 MiniCPM5 XML；`parse_tool_calls` 也能读 `<tool_call>` JSON / Qwen XML，**训练标签只写 XML**。
- 不要把 MiniCPM-o 双工的 `INTENT_PREFIX` + `<tool_call>` JSON 当成 2B 的原生格式。
- 评测 / 日志必须保留 special token。
- 以后 LOCAL 样本：`"<|FC|><|LOCAL|>" + format_tool_xml(name, args)`，不要再发明第三套括号。本版不要灌 XML。
