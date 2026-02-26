# 对探索得到的轨迹进行过滤
# 1. 过滤执行成功的
# 2. 过滤agent认为任务已完成的
# 3. 如果末尾有多个answer动作，第一个answer出现就结束

from __future__ import annotations
import argparse
import json
import re
import os
import hashlib
from pathlib import Path
from PIL import Image, ImageChops
from tqdm import tqdm


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


def _load_metadata_json(traj_dir: Path) -> dict:
    p = traj_dir / "metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"metadata.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _metadata_steps_map(metadata: dict) -> dict[int, dict]:
    steps = (metadata or {}).get("steps") or []
    out: dict[int, dict] = {}
    for s in steps:
        try:
            out[int(s.get("step"))] = s
        except Exception:
            continue
    return out


def _extract_mobile_use_action_and_coord(response_text: str) -> tuple[str | None, list[float] | None]:
    """从 <tool_call> 中抽取 action 与 coordinate（若无则返回 None）。"""
    for call in _iter_tool_calls(response_text or ""):
        if (call or {}).get("name") != "mobile_use":
            continue
        args = (call or {}).get("arguments") or {}
        action = args.get("action")
        coord = args.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            try:
                return str(action) if action is not None else None, [float(coord[0]), float(coord[1])]
            except Exception:
                return str(action) if action is not None else None, None
        return str(action) if action is not None else None, None
    return None, None


def _images_identical_pixels(p1: Path, p2: Path) -> bool:
    """严格像素级一致（与 visualization.ipynb 相同逻辑）。"""
    try:
        with Image.open(p1) as im1, Image.open(p2) as im2:
            im1 = im1.convert("RGB")
            im2 = im2.convert("RGB")
            if im1.size != im2.size:
                return False
            return ImageChops.difference(im1, im2).getbbox() is None
    except Exception:
        raise SystemExit(f"error in images_identical_pixels: {p1} or {p2}")


def _coord_to_logical_pixels(coord: list[float], logical_w: int, logical_h: int) -> tuple[int, int]:
    """将 tool_call 的坐标映射到 metadata 的 logical_screen_size 像素坐标。

    - 若坐标看起来是 0-1000（强模型/Gemini/Qwen3 风格），按 /1000 映射到 [0, W/H)。
    - 否则认为已经是像素坐标（仅做 round + clamp）。
    """
    if not coord or len(coord) != 2:
        return (0, 0)
    x, y = float(coord[0]), float(coord[1])

    # Heuristic: 0-1000 normalized coords.
    if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0 and logical_w > 0 and logical_h > 0:
        xp = int(round(x / 1000.0 * float(logical_w)))
        yp = int(round(y / 1000.0 * float(logical_h)))
    else:
        xp = int(round(x))
        yp = int(round(y))

    # clamp to screen
    if logical_w > 0:
        xp = max(0, min(logical_w - 1, xp))
    else:
        xp = 0
    if logical_h > 0:
        yp = max(0, min(logical_h - 1, yp))
    else:
        yp = 0
    return (xp, yp)


def _pick_min_area_bbox(ui_elements: list[dict], x: int, y: int) -> dict | None:
    """从 ui_elements 的 bbox_pixels 中选出包含点 (x,y) 且面积最小的 bbox。"""
    best = None
    best_area = None
    for el in ui_elements or []:
        bbox = (el or {}).get("bbox_pixels")
        if not isinstance(bbox, dict):
            continue
        try:
            x_min = int(bbox.get("x_min"))
            x_max = int(bbox.get("x_max"))
            y_min = int(bbox.get("y_min"))
            y_max = int(bbox.get("y_max"))
        except Exception:
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        area = (x_max - x_min) * (y_max - y_min)
        if best_area is None or area < best_area:
            best_area = area
            best = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    return best


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


def _with_image_keys(traj_dir: Path, data: dict, metadata: dict, base_dir=None) -> dict:
    traj = data.get("trajectory") or []
    traj = _trim_answer_tail(traj)
    meta_map = _metadata_steps_map(metadata)
    new_traj: list[dict] = []
    for idx, step_obj in enumerate(traj):
        step_id = step_obj.get("step", idx)
        step_id = int(step_id)
        s = dict(step_obj)
        s["image"] = _image_relpath(traj_dir, step_id)
        # bbox：仅 click/long_press（兼容 left_click）时才填
        action, coord = _extract_mobile_use_action_and_coord(step_obj.get("response", ""))
        bbox = None
        if action in {"click", "left_click", "long_press"} and coord is not None:
            meta_step = meta_map.get(step_id)
            if not meta_step:
                raise SystemExit(f"metadata step not found: {traj_dir}/metadata.json step={step_id}")
            logical = meta_step.get("logical_screen_size") or []
            if not (isinstance(logical, (list, tuple)) and len(logical) == 2):
                raise SystemExit(f"bad logical_screen_size in metadata: {traj_dir}/metadata.json step={step_id}")
            logical_w, logical_h = int(logical[0]), int(logical[1])
            x, y = _coord_to_logical_pixels(coord, logical_w, logical_h)
            bbox = _pick_min_area_bbox(meta_step.get("ui_elements") or [], x, y)
        s["bbox"] = bbox
        if base_dir:
            if not os.path.exists(os.path.join(base_dir, s["image"])):
                print(f"image not found: {s['image']}")
        new_traj.append(s)

    # is_valid：若本步操作导致“下一步截图完全没变化”，则 False。
    # 例外：若下一步 action 是 type，则本步仍保留为 True（可能是点击输入框）。
    # 仅在两张截图均存在且能做像素级判断时生效；否则默认 True。
    base_path = Path(base_dir) if base_dir else None
    for i in range(len(new_traj)):
        new_traj[i]["is_valid"] = True
    for i in range(len(new_traj) - 1):
        cur = new_traj[i]
        nxt = new_traj[i + 1]
        if not base_path:
            continue
        p1 = base_path / str(cur.get("image", ""))
        p2 = base_path / str(nxt.get("image", ""))
        if not (p1.exists() and p2.exists()):
            raise SystemExit(f"image not found: {p1} or {p2}")
        if _images_identical_pixels(p1, p2):
            next_action, _ = _extract_mobile_use_action_and_coord(nxt.get("response", ""))
            if next_action != "type":
                cur["is_valid"] = False
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

    for traj_dir in tqdm(sorted(p for p in runs_dir.iterdir() if p.is_dir())):
        total_dirs += 1
        result_path = _find_result_json(traj_dir)
        if not result_path:
            continue
        has_result += 1
        try:
            if _is_success_terminal(result_path):
                terminate_success += 1
                data = json.loads(result_path.read_text(encoding="utf-8"))
                metadata = _load_metadata_json(traj_dir)  # success 必须有 metadata.json，否则直接报错
                merged.append(_with_image_keys(traj_dir, data, metadata, base_dir=runs_dir))
        except FileNotFoundError as e:
            # 成功轨迹缺 metadata.json：按需求直接报错退出
            raise SystemExit(str(e))
        except Exception:
            # 解析失败视为非 success（但仍算“有 result 文件”）
            print(f"parse failed: {result_path}")
            continue

    print(f"runs_dir: {runs_dir}")
    print(f"total_traj_dirs: {total_dirs}")
    print(f"with_result_json: {has_result}")
    print(f"agent_success(terminate_success_or_answer): {terminate_success}")

    out_path = runs_dir / "data_merge_0224_refine.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path} (items={len(merged)})")

    # 统计总step数和平均step数
    total_steps = 0
    for item in merged:
        total_steps += len(item["trajectory"])
    print(f"total_steps: {total_steps}")
    print(f"average_steps: {total_steps / len(merged):.2f}")
    # 统计is_valud=False数量
    invalid_steps = 0
    for item in merged:
        for step in item["trajectory"]:
            if step.get("is_valid", True) is False:
                invalid_steps += 1
    print(f"invalid_steps: {invalid_steps}")

    # 统计每个app的轨迹数量
    data_meta = json.load(open("/Users/chengkanzhi/Desktop/ScaleCUA/evaluation/AndroidWorld/synthesized_tasks_0219_final.json", "r"))
    dir_2_app = {}
    for item in data_meta:
        app_name = item["app"]
        dir_name = str(item["sample_id"])+"_"+item["base_task_name"]
        dir_2_app[dir_name] = app_name
    app_2_num = {}
    for item in merged:
        dir_name = item["save_dir"]
        app_name = dir_2_app[dir_name]
        if app_name not in app_2_num:
            app_2_num[app_name] = 1
        app_2_num[app_name] += 1
    print("Num per app:")
    for app, num in app_2_num.items():
        print(f"{app}: {num}")

    # 统计weak/strong模型步数
    weak_steps = 0
    strong_steps = 0
    unknown_policy_steps = 0
    for item in merged:
        for step in item.get("trajectory", []):
            src = step.get("policy_source")
            if src == "weak":
                weak_steps += 1
            elif src == "strong":
                strong_steps += 1
            else:
                unknown_policy_steps += 1

    print(f"weak_steps: {weak_steps}")
    print(f"strong_steps: {strong_steps}")
    if unknown_policy_steps:
        print(f"unknown_policy_steps: {unknown_policy_steps}")
    

if __name__ == "__main__":
    main()

