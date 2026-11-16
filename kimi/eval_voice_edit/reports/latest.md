# Voice-edit eval report

- time: 2026-09-28T15:22:12+00:00
- mode: http
- base: http://127.0.0.1:7861
- n: 66

## Gates

- PASS: `replace_toggle_exact_ge_0.85`
- PASS: `noop_exact_ge_0.90`
- PASS: `echo_fail_le_0.05`
- PASS: `instr_cer_le_0.15`

## All

```json
{
  "n": 66,
  "exact": 0.8333333333333334,
  "ok": 0.8333333333333334,
  "edit_cer": 0.09031986531986531,
  "echo_fail": 0.0,
  "over_edit": 0.030303030303030304,
  "tts_skip": 0.0,
  "error": 0.0,
  "instr_cer": 0.03638583638583638,
  "via": {
    "rule": 0.4393939393939394,
    "rule_toggle": 0.09090909090909091,
    "llm": 0.21212121212121213,
    "rule_fuzzy": 0.07575757575757576,
    "noop": 0.12121212121212122,
    "rule_radical": 0.06060606060606061
  }
}
```

## By bucket

- **asr_fix**: exact=0.909 edit_cer=0.030 n=11
- **delete**: exact=0.000 edit_cer=0.875 n=2
- **glyph**: exact=0.667 edit_cer=0.097 n=6
- **hard**: exact=1.000 edit_cer=0.000 n=1
- **insert**: exact=1.000 edit_cer=0.000 n=5
- **multi**: exact=1.000 edit_cer=0.000 n=6
- **noop**: exact=1.000 edit_cer=0.000 n=6
- **paraphrase**: exact=0.000 edit_cer=0.549 n=6
- **replace**: exact=1.000 edit_cer=0.000 n=17
- **toggle**: exact=1.000 edit_cer=0.000 n=6

## By engine

- **edge**: exact=0.833 instr_cer=0.036 edit_cer=0.090 n=66
