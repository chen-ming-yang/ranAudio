# ranAudio 版本迭代

记录 MiniCPM5 单轮语音相关契约（按版本追加，不覆盖旧版）。

与平级仓库 [`MiniCPM-o-Demo/docs/function_call/`](../../MiniCPM-o-Demo/docs/function_call/README.md) **版本号独立**：那边是 MiniCPM-o **全双工** FC；这边是 **ranAudio 离线单轮**。二者共享 22 个车机工具与 Qwen Slave 执行层。

架构标准：[MiniCPM5-2B_音频门控架构定义.md](./MiniCPM5-2B_音频门控架构定义.md)（**Draft v0.3** · 旁路 ASR + FastLM 三路由；§3.2）  
实现对照：[architecture.md](./architecture.md)（协议已改文档，**代码未跟**）

2B 原生 FC 格式与触发（2026-09-13 实测）：[minicpm5_fc_format.md](./minicpm5_fc_format.md)

S1.5（SenseVoice + 新 projector）：[S1.5_SenseVoice_训练计划.md](./S1.5_SenseVoice_训练计划.md)（a / b-S / **b-MS 已完成**；数字在 §9）  
S2（v0.3 目标：`NOFC` / `FC+LOCAL` / `DELEGATE`；§8/§9 为旧 FC+BASE 实测）：[S2_FC_NOFC_训练计划.md](./S2_FC_NOFC_训练计划.md)

| 版本 | 文件 | 范围 | 状态 |
|------|------|------|------|
| v1.0 | [v1.0.md](./v1.0.md) | 离线单轮 wav→MiniCPM5 XML FC；2750 样本；22 工具矩阵评测 | **已达成** 22/22（基线保留） |
| v1.1 | [v1.1.md](./v1.1.md) | 2B：ASR + need_fc；无 FC 自带 reply；有 FC 才交 Slave | 契约与代码已对齐；权重未强制开训 |
| v2.0 | [MiniCPM5-2B_音频门控架构定义.md](./MiniCPM5-2B_音频门控架构定义.md) | Draft **v0.3**：旁路 ASR 始终开；NOFC / FC+LOCAL / DELEGATE | **协议文档已定**；代码/权重仍旧 S2 |

追加新版本时：新建版本文件，只改本表；**不要改**已发布版本正文。
