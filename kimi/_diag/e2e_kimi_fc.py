"""端到端: Kimi-Audio(听音频->ASR) -> Qwen Slave(FC 规划->工具执行)

显存策略: Kimi 与 Slave 不能同时驻留(21G 空闲 < Kimi 22G 需求),
因此分两阶段串行: 先 ASR(释放显存) -> 再启动 Slave 做 FC -> 用完即停。

用法: python e2e_kimi_fc.py <wav> [expected_tool]
"""
import gc
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, "/cmy/cmy/MiniCPM-o-Demo")

WAV = sys.argv[1]
EXPECT = sys.argv[2] if len(sys.argv) > 2 else None
SLAVE = "http://127.0.0.1:22600"
DEMO = "/cmy/cmy/MiniCPM-o-Demo"
SLAVE_LOG = "/tmp/slave_e2e.log"


def gpu_free_gib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    return int(out) / 1024.0


def wait_slave(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = urllib.request.urlopen(SLAVE + "/health", timeout=3).read().decode()
            if "ready" in r:
                return True
        except Exception:
            pass
        time.sleep(4)
    return False


def start_slave():
    log = open(SLAVE_LOG, "w")
    p = subprocess.Popen(
        ["/cmy/cmy/miniconda3/envs/qwen3-compare/bin/python", "-u",
         "-m", "finetune_fc.slave.openai_server",
         "--port", "22600",
         "--model-path", "/cmy/cmy/Qwen/Qwen3.5-2B",
         "--engine", "hf"],
        cwd=DEMO, stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid)
    print("      启动 Slave: pid=" + str(p.pid), flush=True)
    return p


def stop_slave(p):
    if p is None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    print("      Slave 已停止", flush=True)


def slave_fc(text, tools):
    payload = {"model": "qwen3.5-2b",
               "messages": [{"role": "user", "content": text}],
               "tools": tools}
    req = urllib.request.Request(
        SLAVE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def main():
    print("=" * 60, flush=True)
    print("[阶段1] Kimi-Audio ASR", flush=True)
    print("  空闲显存: " + str(round(gpu_free_gib(), 1)) + " GiB", flush=True)
    from kimia_infer.api.kimia import KimiAudio
    model = KimiAudio(
        model_path="/cmy/cmy/ranAudio/kimi/models/Kimi-Audio-7B-Instruct",
        load_detokenizer=False)
    sp = {"audio_temperature": 0.8, "audio_top_k": 10,
          "text_temperature": 0.0, "text_top_k": 5,
          "audio_repetition_penalty": 1.0, "audio_repetition_window_size": 64,
          "text_repetition_penalty": 1.0, "text_repetition_window_size": 16}
    msgs = [{"role": "user", "message_type": "text",
             "content": "Please transcribe the following audio:"},
            {"role": "user", "message_type": "audio", "content": WAV}]
    _, asr_text = model.generate(msgs, **sp, output_type="text")
    print("  ASR 输出: " + str(asr_text), flush=True)
    del model
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    print("  Kimi 卸载后空闲: " + str(round(gpu_free_gib(), 1)) + " GiB", flush=True)

    print("=" * 60, flush=True)
    print("[阶段2] Qwen Slave FC 规划", flush=True)
    from finetune_fc.fc_format import load_tools
    tools = load_tools()
    print("  工具数: " + str(len(tools)), flush=True)
    slave_proc = start_slave()
    try:
        if not wait_slave():
            print("  Slave 启动失败, 见 " + SLAVE_LOG, flush=True)
            return
        d = slave_fc(asr_text, tools)
        m = d["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        names = [t["function"]["name"] for t in tcs]
        print("  tool_calls   = " + str(names), flush=True)
        print("  confirmation = " + str(m.get("content"))[:150], flush=True)
        for r in d.get("tool_results", []):
            print("  tool_result: " + r["name"] + " -> " + str(r["content"])[:100], flush=True)
        print("  device = " + json.dumps(d.get("device"), ensure_ascii=False), flush=True)
        if EXPECT:
            print("\n[RESULT] expect=" + EXPECT + " got=" + str(names)
                  + " -> " + ("PASS" if EXPECT in names else "FAIL"), flush=True)
    finally:
        stop_slave(slave_proc)


main()
