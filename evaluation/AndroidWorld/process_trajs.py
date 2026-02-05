# 对探索得到的轨迹进行过滤
# 1. 过滤执行成功的
# 2. 过滤agent认为任务已完成的
# 3. 如果末尾有多个answer动作，第一个answer出现就结束
# Good：154_RetroCreatePlaylist、557_SimpleSmsSendClipboardContent、180_SimpleDrawProCreateDrawing


from __future__ import annotations
import argparse
import json
import re
import os
from pathlib import Path


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _find_result_json(traj_dir: Path) -> Path | None:
    # 兼容两种命名：result.json / results.json
    for name in ("result.json", "results.json"):
        p = traj_dir / name
        if p.exists():
            return p
    return None


def _iter_tool_calls(response_text: str):
    for m in TOOL_CALL_RE.finditer(response_text or ""):
        try:
            yield json.loads(m.group(1))
        except Exception:
            continue


def _is_success_response(response_text: str) -> bool:
    """成功判定（agent认为成功）：terminate(status=success) 或 answer。"""
    for call in _iter_tool_calls(response_text):
        if (call or {}).get("name") != "mobile_use":
            continue
        args = (call or {}).get("arguments") or {}
        action = args.get("action")
        if action == "answer":
            return True
        if action == "terminate" and args.get("status") == "success":
            return True
    return False


def _is_answer_step(step_obj: dict) -> bool:
    for call in _iter_tool_calls(step_obj.get("response", "")):
        if (call or {}).get("name") == "mobile_use" and ((call.get("arguments") or {}).get("action") == "answer"):
            return True
    return False


def _is_success_terminal(result_path: Path) -> bool:
    """检查 result(s).json 的最后一步是否为 success（terminate success 或 answer）。"""
    data = json.loads(result_path.read_text(encoding="utf-8"))
    traj = data.get("trajectory") or []
    if not traj:
        return False

    return _is_success_response(traj[-1].get("response", ""))


def _trim_answer_tail(traj: list[dict]) -> list[dict]:
    """若末尾是连续多个 answer step，只保留第一个 answer（其后的全部舍弃）。"""
    if len(traj) < 2 or not _is_answer_step(traj[-1]):
        return traj

    i = len(traj) - 1
    while i >= 0 and _is_answer_step(traj[i]):
        i -= 1
    first_answer_idx = i + 1  # 连续 answer 段的起点
    return traj[: first_answer_idx + 1]


def _image_relpath(traj_dir: Path, step_id: int) -> str:
    # 不做文件存在性/后缀兜底：统一按 png 命名，后续你自行校验即可
    return f"{traj_dir.name}/screenshot_step{step_id}.png"


def _with_image_keys(traj_dir: Path, data: dict, base_dir=None) -> dict:
    traj = data.get("trajectory") or []
    traj = _trim_answer_tail(traj)
    new_traj: list[dict] = []
    for idx, step_obj in enumerate(traj):
        step_id = step_obj.get("step", idx)
        step_id = int(step_id)
        s = dict(step_obj)
        s["image"] = _image_relpath(traj_dir, step_id)
        if base_dir:
            if not os.path.exists(os.path.join(base_dir, s["image"])):
                print(f"image not found: {s['image']}")
        new_traj.append(s)
    out = dict(data)
    # merged 数据里只保留子文件夹名，去掉前面的绝对路径
    out["save_dir"] = traj_dir.name
    out["trajectory"] = new_traj
    return out


def main():
    parser = argparse.ArgumentParser(description="Process AndroidWorld trajectories and count successes.")
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("/Users/chengkanzhi/Desktop/ScaleCUA/evaluation/AndroidWorld/runs/diy_0126"),
        help="runs 根目录（每个子目录是一条轨迹）",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.exists():
        raise SystemExit(f"runs_dir 不存在：{runs_dir}")

    total_dirs = 0
    has_result = 0
    terminate_success = 0
    merged: list[dict] = []

    for traj_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        total_dirs += 1
        result_path = _find_result_json(traj_dir)
        if not result_path:
            continue
        has_result += 1
        try:
            if _is_success_terminal(result_path):
                terminate_success += 1
                data = json.loads(result_path.read_text(encoding="utf-8"))
                merged.append(_with_image_keys(traj_dir, data, base_dir=runs_dir))
        except Exception:
            # 解析失败视为非 success（但仍算“有 result 文件”）
            print(f"parse failed: {result_path}")
            continue

    print(f"runs_dir: {runs_dir}")
    print(f"total_traj_dirs: {total_dirs}")
    print(f"with_result_json: {has_result}")
    print(f"agent_success(terminate_success_or_answer): {terminate_success}")

    out_path = runs_dir / "data_merge_0126.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path} (items={len(merged)})")

    # 统计总step数和平均step数
    total_steps = 0
    for item in merged:
        total_steps += len(item["trajectory"])
    print(f"total_steps: {total_steps}")
    print(f"average_steps: {total_steps / len(merged):.2f}")


if __name__ == "__main__":
    main()

