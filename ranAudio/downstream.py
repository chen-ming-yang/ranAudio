"""Route judge output: FC -> Qwen Slave; no-FC reply stays on 2B."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from finetune_fc.slave_client import DEFAULT_SLAVE_URL, SlaveClient, SlaveResult


@dataclass
class DownstreamResult:
    need_fc: bool
    asr: str
    mode: str
    reply: str
    slave: Optional[SlaveResult] = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "need_fc": self.need_fc,
            "asr": self.asr,
            "mode": self.mode,
            "reply": self.reply,
            "ok": True,
            "name": "",
            "arguments": {},
            "result": None,
            "error": None,
            "slave_ms": None,
        }
        if self.slave is not None:
            out.update(
                {
                    "ok": self.slave.ok,
                    "name": self.slave.name,
                    "arguments": self.slave.arguments,
                    "result": self.slave.result,
                    # Prefer slave confirmation when FC ran; else keep 2B reply.
                    "reply": self.slave.confirmation or self.reply,
                    "error": self.slave.error,
                    "slave_ms": self.slave.slave_ms,
                }
            )
        return out


async def route_async(
    *,
    asr: str,
    need_fc: bool,
    reply: str = "",
    slave_url: str = DEFAULT_SLAVE_URL,
    client: Optional[SlaveClient] = None,
) -> DownstreamResult:
    """need_fc=true: pass ASR to Slave. need_fc=false: use 2B reply, no Slave chat."""
    if need_fc:
        cli = client or SlaveClient(base_url=slave_url)
        slave = await cli.plan_and_execute(asr)
        return DownstreamResult(
            need_fc=True,
            asr=asr,
            mode="fc",
            reply=slave.confirmation or "",
            slave=slave,
        )
    return DownstreamResult(
        need_fc=False,
        asr=asr,
        mode="chat",
        reply=reply,
        slave=None,
    )


def route(
    *,
    asr: str,
    need_fc: bool,
    reply: str = "",
    slave_url: str = DEFAULT_SLAVE_URL,
) -> DownstreamResult:
    return asyncio.run(
        route_async(asr=asr, need_fc=need_fc, reply=reply, slave_url=slave_url)
    )
