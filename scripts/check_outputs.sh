#!/usr/bin/env bash
# check_outputs.sh — OFFLINE. Validates a downloaded final MP4 on this machine.
# Usage: ./scripts/check_outputs.sh <path/to/final.mp4> [--task-spec <spec.json>]
# Runs h3_video_qa.validate_final_video: defaults to 274 frames at 24 fps and
# 1080x1920, or derives expected dimensions/fps/frames from --task-spec.
set -euo pipefail
CALLER_PWD="$(pwd -P)"
cd "$(dirname "$0")/.."
REPO_REAL="$(pwd -P)"
. "./scripts/_colab_env.sh"

_resolve_caller_path() {
  local p="$1"
  case "$p" in
    ""|/*) printf '%s' "$p"; return ;;
  esac
  if [ "$CALLER_PWD" = "$REPO_REAL" ]; then
    printf '%s' "$p"
  elif [ -e "$CALLER_PWD/$p" ]; then
    printf '%s' "$CALLER_PWD/$p"
  elif [ -e "$REPO/$p" ]; then
    printf '%s' "$REPO/$p"
  else
    printf '%s' "$CALLER_PWD/$p"
  fi
}

MP4_PATH=""
TASK_SPEC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --task-spec)
      if [ $# -lt 2 ]; then
        echo "Usage: $0 <path/to/final.mp4> [--task-spec <spec.json>]" >&2
        exit 2
      fi
      TASK_SPEC="$(_resolve_caller_path "$2")"
      shift 2
      ;;
    --task-spec=*)
      TASK_SPEC="$(_resolve_caller_path "${1#--task-spec=}")"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 <path/to/final.mp4> [--task-spec <spec.json>]"
      exit 0
      ;;
    -*)
      echo "Usage: $0 <path/to/final.mp4> [--task-spec <spec.json>]" >&2
      exit 2
      ;;
    *)
      if [ -n "$MP4_PATH" ]; then
        echo "Usage: $0 <path/to/final.mp4> [--task-spec <spec.json>]" >&2
        exit 2
      fi
      MP4_PATH="$(_resolve_caller_path "$1")"
      shift
      ;;
  esac
done

if [ -z "$MP4_PATH" ]; then
  echo "Usage: $0 <path/to/final.mp4> [--task-spec <spec.json>]" >&2
  exit 2
fi

"$PY" - "$MP4_PATH" "$TASK_SPEC" <<'PY'
import json, sys
from pathlib import Path
from h3_task_spec import load_task_spec
from h3_video_qa import validate_final_video

mp4_path = Path(sys.argv[1])
spec_arg = sys.argv[2]
if spec_arg:
    spec = load_task_spec(Path(spec_arg))
    result = validate_final_video(
        mp4_path,
        expected_frames=spec.total_frames,
        expected_fps=spec.fps,
        expected_width=spec.output_width,
        expected_height=spec.output_height,
    )
else:
    result = validate_final_video(mp4_path)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
