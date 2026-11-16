"""Step-Audio FastLM + Qwen3.5-2B SlowLM routing (S2 FC/NOFC protocol).

Canonical doc: ranAudio/STEP_ROUTE.md
"""

from .pipeline import StepRoutePipeline, TurnResult
from .protocol_step import parse_s2_output

__all__ = ["StepRoutePipeline", "TurnResult", "parse_s2_output"]
