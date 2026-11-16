#!/usr/bin/env bash
# Trigger Cursor Ports forward + open local browser to Kimi demo.
# Must run inside a Cursor Remote / SSH session (needs VSCODE_IPC_HOOK_CLI).
set -euo pipefail
set +o noclobber

PORT="${KIMI_DEMO_PORT:-7861}"
URL="${KIMI_OPEN_URL:-http://127.0.0.1:${PORT}/}"

# Prefer Cursor remote browser helper (openExternal → local browser + port forward)
if [[ -z "${BROWSER:-}" || ! -x "${BROWSER}" ]]; then
  # Resolve current cursor-server helper
  CAND=$(ls -d /root/.cursor-server/bin/linux-x64/*/bin/helpers/browser.sh 2>/dev/null | tail -1 || true)
  if [[ -n "${CAND:-}" && -x "$CAND" ]]; then
    BROWSER="$CAND"
  fi
fi

if [[ -z "${VSCODE_IPC_HOOK_CLI:-}" ]]; then
  # pick newest ipc sock if env missing
  SOCK=$(ls -t /tmp/vscode-ipc-*.sock 2>/dev/null | head -1 || true)
  if [[ -n "${SOCK:-}" ]]; then
    export VSCODE_IPC_HOOK_CLI="$SOCK"
  fi
fi

echo "[open] waiting for http://127.0.0.1:${PORT}/api/ready ..."
for i in $(seq 1 40); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/api/ready" 2>/dev/null | grep -q '"ready": true'; then
    break
  fi
  sleep 2
done

if ! curl -s -m 2 "http://127.0.0.1:${PORT}/api/ready" 2>/dev/null | grep -q '"ready": true'; then
  echo "[open] error: demo not ready on :${PORT}" >&2
  exit 1
fi

if [[ -n "${BROWSER:-}" && -x "${BROWSER}" && -n "${VSCODE_IPC_HOOK_CLI:-}" ]]; then
  echo "[open] Cursor openExternal → ${URL}"
  "$BROWSER" "$URL" >/dev/null 2>&1 || true
  echo "[open] done. If browser did not open: Ports → Forward ${PORT} → http://localhost:${PORT}"
else
  echo "[open] warn: not in Cursor Remote session (no BROWSER/IPC)."
  echo "[open] Manual: Ports → Forward a Port → ${PORT} → open http://localhost:${PORT}"
  exit 0
fi
