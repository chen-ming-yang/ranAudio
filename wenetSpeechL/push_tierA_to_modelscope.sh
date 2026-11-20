#!/usr/bin/env bash
# 把 Tier-A staging 目录发布到 ModelScope 数据集仓库
#   仓库: chenhaha12315/wenet-speech-L-filter
#
# 用法:
#   MODELSCOPE_TOKEN=xxxx bash push_tierA_to_modelscope.sh            # 正式上传
#   MODELSCOPE_TOKEN=xxxx bash push_tierA_to_modelscope.sh --dry-run  # 只校验
#
# 说明:
#   - 单文件 393MB < 50GB 上限；>5MB 由平台自动走 Git LFS
#   - 17 文件，远低于 10 万文件 / 单目录 1 万文件限制
set -euo pipefail

STAGE=/tmp/ms_stage/wenet-speech-L-filter
REPO=chenhaha12315/wenet-speech-L-filter
LOG=/cmy/cmy/.migrate-logs/ms_upload.log
mkdir -p "$(dirname "$LOG")"
export PATH=$PATH:/usr/local/bin

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY=1

if [[ -z "${MODELSCOPE_TOKEN:-}" ]]; then
  echo "缺少 MODELSCOPE_TOKEN"; exit 1
fi

echo "=== 预检 ==="
[[ -d "$STAGE" ]] || { echo "staging 不存在: $STAGE"; exit 1; }
NF=$(find "$STAGE" -type f | wc -l)
SZ=$(du -sh "$STAGE" | cut -f1)
MAXF=$(find "$STAGE" -type f -printf '%s\n' | sort -rn | head -1)
echo "文件数 : $NF        (限制 100,000)"
echo "总体积 : $SZ"
echo "最大单文件: $MAXF B (限制 50 GB)"
if (( MAXF > 50000000000 )); then echo "超过单文件上限"; exit 1; fi

if [[ -n "$DRY" ]]; then echo "--dry-run: 预检通过，不执行上传"; exit 0; fi

echo "=== 开始上传 -> $REPO ==="
python3 - "$STAGE" "$REPO" "$MODELSCOPE_TOKEN" <<'PY' 2>&1 | tee -a "$LOG"
import sys
from modelscope.hub.api import HubApi

stage, repo, token = sys.argv[1], sys.argv[2], sys.argv[3]
api = HubApi()
api.login(token)
print("login OK")

api.upload_folder(
    repo_id=repo,
    repo_type='dataset',
    folder_path=stage,
    commit_message='Upload WenetSpeech-L Tier-A Agree3 (10 shards, 10,156,566 utt / 6190.72 h)',
    ignore_patterns=['.git/*', '**/__pycache__/*'],
)
print("UPLOAD DONE")
PY

echo "=== 完成 $(date '+%F %T') ==="
echo "https://www.modelscope.cn/datasets/$REPO"
