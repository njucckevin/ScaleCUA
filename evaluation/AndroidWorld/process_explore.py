# 将 random walk 探索的轨迹处理为 state_transfer 的形式
#
# 用法示例：
#   python evaluation/AndroidWorld/process_explore.py \
#     --traj_dir /Users/cckevin/ScaleCUA/evaluation/AndroidWorld/explore_results/trajectories \
#     --out /Users/cckevin/ScaleCUA/evaluation/AndroidWorld/state_transfer_from_explore.json
#
# 说明：
# - 输入轨迹由 evaluation/AndroidWorld/random_walk_aw.py 生成，单个 trajectory 文件是一个 JSON list，
#   每个 step dict 里包含 task_uuid/app/task/screen_before/screen_after/action/action_element 等字段。
# - 输出为一个 JSON array（通常单行），每个 item 类似 evaluation/AndroidWorld/state_transfer_mobile.json：
#   {
#     "task_id": "...",          # = task_uuid
#     "step_id": 1,              # 固定从 1 开始
#     "app": "...",
#     "task": "...",
#     "screen_before": "xxx.png",# 只保留文件名（basename）
#     "screen_after": "yyy.png",
#     "action_type": "CLICK|LONG_PRESS|PRESS_BACK|SCROLL DOWN|SCROLL UP|TYPE '...'",
#     "bbox": {"x_min":..,"x_max":..,"y_min":..,"y_max":..} 或 null
#   }

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm


def _escape_single_quotes(text: str) -> str:
    # state_transfer_mobile.json 里 TYPE action 采用单引号包裹
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _format_action_type(action: Dict[str, Any]) -> str:
    """把 random_walk_aw.py 的 action_sample 映射到 state_transfer_mobile.json 的 action_type。"""
    a = (action.get("action_type") or "").strip()

    if a == "click":
        return "CLICK"
    if a == "long_press":
        return "LONG_PRESS"
    if a == "navigate_back":
        return "PRESS_BACK"
    if a == "scroll":
        direction = (action.get("direction") or "").strip().lower()
        if direction == "down":
            return "SCROLL DOWN"
        if direction == "up":
            return "SCROLL UP"
        # 兜底：保留原 direction
        return f"SCROLL {direction}".strip().upper()
    if a == "input_text":
        text = action.get("text")
        if text is None:
            text = ""
        text = str(text)
        return f"TYPE '{_escape_single_quotes(text)}'"

    # 兜底：尽量不崩溃，返回一个可读的 action_type
    # （如果你后续发现还有别的动作类型，可以在上面继续补映射）
    return a.upper() if a else "UNKNOWN"


def _extract_bbox(action_element: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not action_element:
        return None
    bbox = action_element.get("bbox_pixels")
    if not isinstance(bbox, dict):
        return None
    keys = ("x_min", "x_max", "y_min", "y_max")
    if not all(k in bbox for k in keys):
        return None
    try:
        return {k: int(bbox[k]) for k in keys}
    except Exception:
        return None


def _convert_one_trajectory(traj_path: Path) -> List[Dict[str, Any]]:
    with traj_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Trajectory JSON is not a list: {traj_path}")

    out: List[Dict[str, Any]] = []
    for i, step in enumerate(data):
        if not isinstance(step, dict):
            continue

        task_uuid = step.get("task_uuid")
        app = step.get("app")
        task = step.get("task")
        screen_before = step.get("screen_before")
        screen_after = step.get("screen_after")
        action = step.get("action") or {}

        # 兼容少数字段缺失：缺啥就跳过该 step
        if not task_uuid or not app or not task or not screen_before or not screen_after or not isinstance(action, dict):
            continue

        item = {
            "task_id": str(task_uuid),
            "step_id": int(i + 1),
            "app": str(app),
            "task": str(task),
            "screen_before": Path(str(screen_before)).name,
            "screen_after": Path(str(screen_after)).name,
            "action_type": _format_action_type(action),
            "bbox": _extract_bbox(step.get("action_element")),
        }
        out.append(item)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    here = Path(__file__).resolve().parent
    default_traj_dir = here / "explore_results" / "trajectories"
    default_out = here / "state_transfer_from_explore.json"

    parser = argparse.ArgumentParser(description="Convert explore trajectories to state_transfer format.")
    parser.add_argument("--traj_dir", type=str, default=str(default_traj_dir), help="Trajectory JSON directory.")
    parser.add_argument("--out", type=str, default=str(default_out), help="Output JSON file path.")
    parser.add_argument("--indent", type=int, default=None, help="JSON indent (default: single-line, compact).")
    args = parser.parse_args(argv)

    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists() or not traj_dir.is_dir():
        print(f"[ERROR] traj_dir not found or not a directory: {traj_dir}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(traj_dir.glob("*.json"))
    if not traj_files:
        print(f"[ERROR] no *.json found in: {traj_dir}", file=sys.stderr)
        return 2

    all_items: List[Dict[str, Any]] = []
    bad_files = 0
    for p in tqdm(traj_files):
        try:
            all_items.extend(_convert_one_trajectory(p))
        except Exception as e:
            bad_files += 1
            print(f"[WARN] failed to process {p.name}: {e!r}", file=sys.stderr)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=args.indent)
        if args.indent is None:
            f.write("\n")

    print(f"[OK] trajectories: {len(traj_files)} files ({bad_files} failed), items: {len(all_items)}")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())