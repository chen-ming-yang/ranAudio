#!/usr/bin/env python3
"""Probe: does uvicorn accept websockets from websockets-client?"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import websockets
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
import uvicorn

app = FastAPI()


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    print("HANDLER ENTER", flush=True)
    await ws.accept()
    print("HANDLER ACCEPT", flush=True)
    await ws.send_json({"type": "session", "ok": True})
    while True:
        msg = await ws.receive()
        if msg.get("type") == "websocket.disconnect":
            break
        if msg.get("text"):
            data = json.loads(msg["text"])
            if data.get("type") == "end":
                await ws.send_json({"type": "final", "text": "hi"})
                break


def main() -> None:
    with TestClient(app) as c:
        with c.websocket_connect("/ws/asr") as ws:
            print("tc", ws.receive_json())
            ws.send_json({"type": "end"})
            print("tc2", ws.receive_json())

    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="info")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    time.sleep(1.5)

    async def client():
        async with websockets.connect("ws://127.0.0.1:8765/ws/asr") as ws:
            print("ws", await ws.recv())
            await ws.send(json.dumps({"type": "end"}))
            print("ws2", await ws.recv())

    asyncio.run(client())
    server.should_exit = True
    time.sleep(0.5)
    print("probe ok")


if __name__ == "__main__":
    main()
