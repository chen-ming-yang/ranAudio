#!/usr/bin/env python3
import asyncio
import json
import sys

import numpy as np
import websockets


async def main(url: str = "ws://127.0.0.1:7863/ws/asr") -> None:
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "start", "language": "Chinese"}))
        print("sess", await ws.recv())
        pcm = (np.random.randn(3200).astype(np.float32) * 0.05)
        b = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
        for _ in range(20):
            await ws.send(b)
            try:
                print("msg", await asyncio.wait_for(ws.recv(), timeout=0.05))
            except asyncio.TimeoutError:
                pass
        await ws.send(json.dumps({"type": "end"}))
        print("final", await ws.recv())


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:7863/ws/asr"))
