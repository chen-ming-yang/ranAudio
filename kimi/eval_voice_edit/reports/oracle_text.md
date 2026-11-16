# Voice-edit eval report

- time: 2026-09-28T16:15:59+00:00
- mode: oracle-text
- base: http://127.0.0.1:7861
- n: 124

## Gates

- PASS: `replace_toggle_exact_ge_0.85`
- PASS: `noop_exact_ge_0.90`
- PASS: `echo_fail_le_0.05`
- FAIL: `instr_cer_le_0.15`

## All

```json
{
  "n": 124,
  "exact": 0.7983870967741935,
  "ok": 0.7983870967741935,
  "edit_cer": 0.09414711632453569,
  "echo_fail": 0.0,
  "over_edit": 0.0,
  "tts_skip": 0.0,
  "error": 0.0,
  "instr_cer": 0.0,
  "via": {
    "rule": 0.4435483870967742,
    "rule_toggle": 0.08064516129032258,
    "rule_miss": 0.20161290322580644,
    "noop": 0.12096774193548387,
    "rule_radical": 0.1532258064516129
  }
}
```

## By bucket

- **asr_fix**: exact=1.000 edit_cer=0.000 n=15
- **delete**: exact=0.000 edit_cer=0.427 n=8
- **glyph**: exact=1.000 edit_cer=0.000 n=19
- **hard**: exact=1.000 edit_cer=0.000 n=5
- **insert**: exact=0.000 edit_cer=0.499 n=7
- **multi**: exact=1.000 edit_cer=0.000 n=10
- **noop**: exact=1.000 edit_cer=0.000 n=10
- **num**: exact=1.000 edit_cer=0.000 n=5
- **paraphrase**: exact=0.000 edit_cer=0.476 n=10
- **replace**: exact=1.000 edit_cer=0.000 n=25
- **toggle**: exact=1.000 edit_cer=0.000 n=10

## By engine

- **edge**: exact=0.803 instr_cer=0.000 edit_cer=0.095 n=66
- **oracle**: exact=1.000 instr_cer=0.000 edit_cer=0.000 n=5
- **qwen_custom**: exact=0.775 instr_cer=0.000 edit_cer=0.100 n=40
- **qwen_design**: exact=0.769 instr_cer=0.000 edit_cer=0.106 n=13
