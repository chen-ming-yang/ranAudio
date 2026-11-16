# Voice-edit eval schema

Each line in `cases.jsonl` is one JSON object.

| field | required | meaning |
|---|---|---|
| `id` | yes | unique id, also wav stem |
| `bucket` | yes | `replace` / `toggle` / `delete` / `insert` / `asr_fix` / `multi` / `noop` / `paraphrase` / `hard` |
| `draft` | yes | ASR draft text before edit |
| `instruction` | yes | spoken edit instruction (gold transcript) |
| `gold` | yes | expected edited text |
| `engine` | yes | TTS engine: `edge` / `qwen_custom` / `qwen_design` / `index_clone` |
| `voice` | no | engine-specific voice / speaker |
| `instruct` | no | for `qwen_design` style prompt |
| `strict` | no | if true, primary metric is exact match |
| `metric` | no | `exact` (default) or `cer` |
| `instruction_wav` | no | override path to wav; else `wavs/{engine}/{id}.wav` |
| `notes` | no | free text |

Human recordings: place `wavs_human/{id}.wav`; eval prefers them over synthesized audio.

Reports: `reports/latest.json` + `reports/latest.md`.
