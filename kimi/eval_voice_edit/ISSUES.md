# 语音修改评测问题清单

基线报告：`reports/latest.md`。  
相关：`cases.jsonl`、`reports/latest.json`、`scripts/voice_edit_lib.py`、`scripts/demo_web.py`、`scripts/eval_voice_edit.py`。

## 总览（HTTP edge n=60）

| 指标 | 初测 | +清洗 | +multi | +noop | +toggle | 说明 |
|------|------|-------|--------|-------|---------|------|
| exact / ok | 0.433 | 0.633 | 0.733 | 0.800 | **0.817** | 完全正确 |
| instr_cer | 0.174 | 0.036 | 0.036 | 0.036 | 0.036 | 去标点后门禁过 |
| echo_fail | 0.017 | 0.017 | 0.017 | 0.017 | 0.017 | 已过门 |
| prompt_leak | 多条 | 0 | 0 | 0 | 0 | 无 `【…】` 残留 |
| replace | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 规则稳 |
| multi | 0.000 | 0.000 | **1.000** | 1.000 | 1.000 | `extract_replace_pairs` |
| noop | 0.167 | 0.167 | 0.167 | **1.000** | 1.000 | `should_keep_draft` |
| toggle | 0.167 | 0.833 | 0.833 | 0.833 | **1.000** | `apply_toggle` |
| insert | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 | 泄漏修掉 |
| paraphrase / delete | 差 | 差 | 差 | 差 | 仍差 | 见下方 P2 |

门禁（当前）：

- `replace_toggle_exact ≥ 0.85` → **PASS**
- `noop_exact ≥ 0.90` → **PASS**
- `echo_fail ≤ 0.05` → PASS
- `instr_cer ≤ 0.15` → **PASS**

**不需要微调**才能修的问题：P0 后处理、P1 规则。FT 仅在规则 + 清洗后语义仍系统性错时再议。

---

## P0 — 后处理强清洗（已修）

**现象**：LLM 泄漏 `【改写结果】`、句/AB 复读、尾句号假失败。

**样例 id**：`tog_007`, `tog_009`, `di_008`, `nop_010`, `par_*`, `fix_005` / `013` / `014`, `tog_008`

**修改方式**（文件：`scripts/voice_edit_lib.py`；`demo_web` 已调用，重启 demo 生效）：

1. **`clean_rewrite` — 剥提示词泄漏**
   - 若含 `【改写结果】`：取最后一个标记之后；若之后为空则取标记之前。
   - 若仍含 `【草稿】` / `【修改指令】`：取第一个 `【` 之前。
   - 全局删残留 `【[^】]*】`；再剥前缀 `改写结果：` 等。
2. **`clean_rewrite` — 截断循环**
   - 按句切分：相邻同句只留首次。
   - `A B A B…`（≥2 周期）截成稳定短句（相对 draft 选对立侧）。
   - 保留整串 prefix 重复检测 + `len > max(48, 3*len(draft))` 硬截。
   - 只剥尾部 `。！？!?`（UI 保留句内标点）。
3. **`norm_text` — 比分归一**
   - 去空白 + 常见标点；供 exact / CER / `is_echo_fail` 共用（避免「多一个句号」假失败）。
4. **`looks_like_garbage`**
   - 清洗后仍含 `【】`、或同句重复 ≥3、或过长 → 当垃圾，走 rule_fallback / keep。

不需要微调。

---

## P1 — 多替换规则错误（已修）

**现象**：`fullmatch` 把多子句 dst 吃到句末，后半指令写进草稿。

**样例**：`mul_001` — 草稿 `打开客厅的灯和窗帘` + 指令 `把灯改成空调，把窗帘改成窗户` → 错成把后半指令写进正文。

**修改方式**（`scripts/voice_edit_lib.py`）：

1. **`extract_replace_pairs(instruction)`**
   - 用 `finditer`（禁止整句 `fullmatch` 吃多子句）：
     - `(?:把|将)(.+?)(?:[，、]*)(?:变成|改成|换成)(.+?)(?=(?:，|、)?(?:把|将)|$|[。！？!?])`
   - 允许 ASR 在「改成」前插入逗号（`把我到了，改成晚点到`）。
   - src/dst 剥尾标点；丢弃 `src==dst` 或空对。
2. **`apply_simple_replace`**
   - pairs 非空：按 `len(src)` **降序** 依次 `draft.replace(src, dst, 1)`（避免「明天」误伤「明天下午」）。
   - 至少命中 1 次且结果 ≠ 草稿才返回，否则 `None` → LLM。
   - pairs 为空：保留单条宽松 `fullmatch`（无「把」的短指令）。
3. **调用点不用改**：`demo_web` 已走 `_apply_simple_replace`；重启 demo。
4. **回归**：`python scripts/voice_edit_lib.py` self_check 覆盖全部 `multi` + 单条 `rep`；edge HTTP `multi` 6/6 exact。

---

## P1 — noop 不会「别改」（已修）

**现象**

- 闲聊未 keep 草稿：`nop_001`（天气）、`nop_006`（嗯）、`nop_007`（好的收到）、`nop_008`（哈哈哈）
- 形似替换的无关句被当真替换：`nop_010`（把香蕉变成苹果 → 曾改成 `发送苹果`）

**修改方式**（已落地）：

1. **`should_keep_draft(draft, instruction)`**（`voice_edit_lib.py`）
   - 闲聊/语气：`norm_text` 后命中 `_KEEP_EXACT`（嗯/好的/收到/哈哈/谢谢/你是谁…）或 `_KEEP_SUBSTR`（今天天气/讲个笑话/无关指令…）。
   - 极短语气：`≤2` 且全是 `嗯啊哦…`。
   - **无实体重叠**：`extract_replace_pairs` 非空但所有 `src` 都不在草稿中 → keep（`nop_010`）。
2. **接线**：`demo_web` 在规则/LLM **之前**；命中则 `text=draft`、`via=noop`，跳过清洗误伤。
3. **oracle**：`eval_voice_edit.run_oracle` 同样先 `should_keep_draft`。
4. **回归**：edge HTTP `noop` 6/6；`noop_exact` 门禁 PASS。

---

## P1 — toggle 无规则（已修）

**现象**：曾全走 LLM；语义常对，但被标点/泄漏/循环拖死（仅靠清洗约 0.83）。

**样例**：`tog_001`～`tog_010`（开↔关、暂停↔继续、锁屏↔解锁等）

**修改方式**（已落地）：

1. **`apply_toggle(draft, instruction)`**（`voice_edit_lib.py`）
   - 对立词表 `_TOGGLE_PAIRS`（长短语优先）：打开/关闭/关掉、打开热点/关闭热点、暂停播放/继续播放、锁屏/解锁…
   - 剥指令前缀 `改成/变成/换成`，在指令中找目标极性短语（长优先），再在草稿中替换对立词。
   - 长匹配失败则回退短词（如 `改成关闭静音` + `静音模式打开` → 用「关闭」替换「打开」）。
2. **接线顺序**：noop → **toggle** → `apply_simple_replace` → LLM → `clean_rewrite`；`via=rule_toggle`。
3. **oracle / self_check**：全部 `tog_*` 文本断言。
4. **回归**：edge HTTP toggle 6/6 exact。

---

## P2 — 听着对、字不对（已修：偏旁描述 + 草稿近邻）

**现象**：同音字落成错字。用户常用偏旁描述纠错，而不是再读一遍；ASR 也可能把「把A改成B」听成 A→A 或近音 src。

**修改方式**（已落地，`apply_edit_rules` 顺序：noop → toggle → radical → exact → fuzzy → LLM）：

1. **`apply_radical_describe_replace`**：双侧偏旁（`把白字旁的的修改为提土旁的地`）、只描述目标（`把的改成提土旁的地`）、拆字口诀（`把的改成土也地`）。不做「只描述错字」（`把女字旁的她改成他`）。题集 `glyph` 19 条，oracle exact 1.0。
2. **`apply_fuzzy_replace`**：草稿滑窗编辑距离。`把周杰伦改成周杰伦` 把「周杰轮」换成「周杰伦」；`把宫词改成公司` 把「公词」换成「公司」。正确字形已在草稿（`护身→护身`）则不改，交 LLM。
3. **`should_keep_draft`**：偏旁规则能命中、或模糊能命中时不 keep。

HTTP edge：`fix_012` / `fix_014` 走 `rule_fuzzy` 正确；`glyph` 6 条里 4 条 `rule_radical` 正确。`gly_004` / `gly_015` 的 ASR 把源字本身听成了目标同音字（她/在），规则看到 A→A 且单字不做模糊，仍 keep。`fix_002`（护身→护身，正确词「沪深」已丢）仍交 LLM。

---

## 显示归一 — 数字、符号、算式（已修）

**现象**：ASR 常把数字听成汉字（`一二零`、`三点十分`），口语算式和符号也落成文字（`一加一等于二`、`井号`）。产品上要一律显示成阿拉伯数字和符号，且**不需要用户再下指令**。

**样例**

| 输入 | 输出 |
|------|------|
| `拨打一二零` | `拨打120` |
| `二十七` / `三百` / `七点` | `27` / `300` / `7点` |
| `三点十分` | `3:10` |
| `两点五` | `2.5` |
| `百分之二十` | `20%` |
| `一加一等于二` | `1 + 1 = 2` |
| `二乘三等于六` / `十减三等于七` / `八除以二等于四` | `2 × 3 = 6` / `10 - 3 = 7` / `8 ÷ 2 = 4` |
| `一加一`（无等于） | `1 + 1` |
| `井号` / `艾特` / `百分号` / `斜杠` | `#` / `@` / `%` / `/` |
| `三点水`（偏旁说法） | **不改** `三点水` |

**修改方式**（已落地，不微调）：

1. **`normalize_display(text)`**（`scripts/voice_edit_lib.py`）
   - 先保护 `三点水/旁/头/底/边/框`，避免把偏旁口误里的数字改掉。
   - `百分之N` → `N%`；`N点M分` → `N:MM`；`N点M`（小数）→ `N.M`。
   - 口语算式：`N加/减/乘/乘以/除/除以M(等于K)?` → `N + M (= K)`，运算符两侧留空格。
   - 剩余中文数字串 → 阿拉伯数字（含 `幺`、`两`、十百千万亿）。
   - 符号整词（长词优先）：百分号、井号、艾特、斜杠、反斜杠、下划线、加号/减号/乘号/除号/等号、星号、冒号、问号、感叹号、省略号、和号。
2. **接线**
   - `demo_web`：ASR 返回文本、改写草稿入参、指令文本、改写结果返回前均调用（幂等）。
   - 规则前草稿/指令已归一，页面上是 `120` 时再说「把一二零改成一一零」仍可对上。
3. **评测**：`score_one` 对 **pred 与 gold 都先 `normalize_display`** 再比 exact/CER；旧题 gold 仍可写汉字数字。
4. **回归**：`voice_edit_lib` self_check；题集 `num_001`～`num_005`（含算式）oracle exact 1.0。

**为何不微调**：数字、符号、算式是封闭替换，正确写法唯一；规则稳定可测。微调适合没有唯一标准句的 paraphrase。

---

## P2 — paraphrase 字面严、意思近（未修）

**现象**：exact 全灭，但产品上可能可用（关窗 → `请把窗户关上` vs gold `请关闭窗户`）。

**样例**：`par_001`～`par_010`

**修改方式**（评测口径为主，不硬改模型输出对齐某一 gold）：

1. **题集**：`paraphrase` 条设 `"strict": false`，`metric: "cer"`，门禁用 CER≤0.15 / 人工抽检，不用 exact。
2. **报告**：`eval_voice_edit` 对 `strict:false` 已有 `ok` 逻辑时，在 `latest.md` 的 paraphrase 行同时展示 exact 与 ok。
3. **产品**：不强行规则化「说正式一点」；若要提分，再考虑少样本 prompt 或 FT（优先级低于 noop/toggle）。

---

## 优先修改顺序

1. ~~强清洗（`clean_rewrite` / `norm_text`）~~ **已修**
2. ~~multi 规则解析（`extract_replace_pairs`）~~ **已修**
3. ~~noop keep（`should_keep_draft`）~~ **已修**
4. ~~toggle 对立词（`apply_toggle`）~~ **已修**
5. ~~偏旁描述 + 草稿近邻（`rule_radical` / `rule_fuzzy`）~~ **已修**
6. ~~数字 / 符号 / 口语算式显示（`normalize_display`）~~ **已修**
7. paraphrase / delete 评测与规则（下一刀）
8. 仍系统性语义错再考虑微调

## 复现

```bash
cd /cmy/cmy/ranAudio/kimi
# demo ready: curl -s http://127.0.0.1:7861/api/ready
bash scripts/run_voice_edit_eval.sh
# 报告: eval_voice_edit/reports/latest.md
# 规则自检: python scripts/voice_edit_lib.py
```
