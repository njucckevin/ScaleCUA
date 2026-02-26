#!/usr/bin/env python3
"""
将 AndroidWorld 的指令-轨迹（Gemini-3-Pro / Qwen3VL tool-call）数据转换为 LLaMA-Factory 可用的 ShareGPT 格式（包含 system）。

核心对齐点（按你的需求）：
1) 输出为 ShareGPT：messages=[system,user,assistant] + images
2) 单轮样本：每个 step 一个样本，包含 system_prompt + user_prompt + answer
3) history 处理：训练样本的 history 必须等于“上一步的 step_history”（step0 为空），与 Qwen2.5VL 测试时一致
4) assistant 输出：严格输出 <thinking>...</thinking> + <tool_call>{...}</tool_call>
5) 坐标系转换：将 Gemini/Qwen3 的 0-1000 归一化坐标，转换为 Qwen2.5VL smart_resize 后的像素坐标

用法示例：
  python evaluation/AndroidWorld/convert_traj.py \
    --input evaluation/AndroidWorld/runs/diy_0126/data_merge_0126.json \
    --output evaluation/AndroidWorld/runs/diy_0126/train_sharegpt.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm


# -----------------------------
# Qwen2.5VL smart_resize (与 seeact_v.py 保持一致的本地实现)
# -----------------------------
def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """
    Rescale height/width to fit within [min_pixels, max_pixels] while keeping
    aspect ratio and ensuring dimensions are divisible by `factor`.

    该实现等价于：
    transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize
    """
    import math

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


# -----------------------------
# 解析 response：抽出 thinking + tool_call JSON
# -----------------------------
_RE_THINKING = re.compile(r"<thinking>\s*([\s\S]*?)\s*</thinking>", re.I)
_RE_TOOL_CALL = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.I)


def _extract_thinking_and_tool_call(response: str) -> tuple[str, Dict[str, Any]]:
    """
    返回 (thinking_text, tool_call_dict)
    - 优先使用 <thinking>...</thinking>
    - 若缺失 thinking：取 <tool_call> 之前的文本作为 thinking（再剥离可能的 XML）
    """
    m_tool = _RE_TOOL_CALL.search(response or "")
    if not m_tool:
        raise ValueError("No <tool_call>...</tool_call> found in response.")
    tool_payload = m_tool.group(1).strip()
    try:
        tool_call = json.loads(tool_payload)
    except Exception as e:
        raise ValueError(
            f"Failed to parse JSON inside <tool_call>: {e}\nPayload={tool_payload[:200]}..."
        )

    m_th = _RE_THINKING.search(response or "")
    if m_th:
        thinking = m_th.group(1).strip()
    else:
        prefix = (response or "")[: m_tool.start()].strip()
        # 粗暴去掉可能的标签
        prefix = re.sub(r"<[^>]+>", " ", prefix)
        thinking = re.sub(r"\s+", " ", prefix).strip()
    if not thinking:
        thinking = " "
    return thinking, tool_call


# -----------------------------
# Qwen3VL → Qwen2.5VL tool-call 坐标转换
# -----------------------------
def _scale_coord_0_1000_to_resized(
    coord: List[float], resized_w: int, resized_h: int
) -> List[int]:
    """
    Gemini/Qwen3 工具坐标约定（见 PROMPT.py）：坐标范围 0~999（近似 0~1000）。
    代码里统一按 /1000 映射（与 qwen3vl_action_transform 一致），这里也按 /1000。
    """
    if not coord or len(coord) != 2:
        return [0, 0]
    x, y = coord
    xr = int(round(float(x) / 999.0 * float(resized_w)))
    yr = int(round(float(y) / 999.0 * float(resized_h)))
    # clamp
    xr = max(0, min(resized_w - 1, xr)) if resized_w > 0 else 0
    yr = max(0, min(resized_h - 1, yr)) if resized_h > 0 else 0
    return [xr, yr]


def convert_tool_call_qwen3_to_qwen25(
    tool_call: Dict[str, Any], resized_w: int, resized_h: int
) -> Dict[str, Any]:
    """
    将 Gemini/Qwen3 的 tool_call（0-1000 坐标系）转换为 Qwen2.5VL 期望的 tool_call（smart_resize 后像素坐标系）。

    注意：Qwen2.5VL 的执行端会用：
      x_orig = x_resized / resized_w * width
      y_orig = y_resized / resized_h * height
    因此我们需要输出“resized 空间”的坐标。
    """
    if not isinstance(tool_call, dict):
        return {"name": "mobile_use", "arguments": {"action": "wait"}}

    name = tool_call.get("name", "mobile_use")
    args = (
        tool_call.get("arguments", {})
        if isinstance(tool_call.get("arguments", {}), dict)
        else {}
    )
    action = args.get("action", "")

    out_args: Dict[str, Any] = dict(args)  # shallow copy
    # 仅对涉及坐标的动作转换
    if action in {"click", "left_click", "long_press", "swipe"}:
        if "coordinate" in out_args:
            out_args["coordinate"] = _scale_coord_0_1000_to_resized(
                out_args.get("coordinate", [0, 0]), resized_w, resized_h
            )
        if action == "swipe" and "coordinate2" in out_args:
            out_args["coordinate2"] = _scale_coord_0_1000_to_resized(
                out_args.get("coordinate2", [0, 0]), resized_w, resized_h
            )

    # 其余动作：type/answer/system_button/open/wait/terminate/key 直接透传即可
    return {"name": name, "arguments": out_args}


# -----------------------------
# ShareGPT 样本构建
# -----------------------------
@dataclass
class ConvertStats:
    total_steps: int = 0
    written: int = 0
    skipped: int = 0
    invalid_filtered: int = 0
    thinking_empty_filtered: int = 0
    policy_filtered: int = 0


def _load_image_size(path: Path) -> Tuple[int, int]:
    """
    读取图片宽高。
    你的数据里图片全部是 PNG，因此这里只解析 PNG 的 IHDR 来获取宽高（无需 Pillow）。
    """
    if path.suffix.lower() != ".png":
        raise ValueError(f"Expected .png image, got: {path}")
    data = path.read_bytes()
    # PNG signature + IHDR (宽高在固定偏移 16..24)
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG header: {path}")
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    return w, h


def _find_androidworld_root(start_dir: Path) -> Path:
    """
    找到包含 `android_world/` 包目录的根路径（需要被加入 sys.path）。
    预期脚本位于 evaluation/AndroidWorld/ 下或其子目录下。
    """
    cur = start_dir.resolve()
    for p in [cur, *cur.parents]:
        if (p / "android_world").is_dir():
            return p
    raise RuntimeError(
        f"Cannot locate AndroidWorld root containing `android_world/` from: {start_dir}"
    )


def build_sharegpt_samples(
    input_path: Path,
    image_root: Path,
    min_pixels: int,
    max_pixels: int,
    limit: Optional[int] = None,
    absolute_images: bool = False,
    strict: bool = False,
    refine: bool = False,
    model_format: str = "qwen25vl",
) -> tuple[List[Dict[str, Any]], ConvertStats]:
    # 确保能 import android_world（脚本放在 evaluation/AndroidWorld/ 时，这里会返回该目录）
    aw_root = _find_androidworld_root(Path(__file__).resolve().parent)
    if str(aw_root) not in sys.path:
        sys.path.insert(0, str(aw_root))

    from android_world.agents.PROMPT import (
        Qwen25VL_SYSTEM_PROMPT,
        QWEN25VL_USER_PROMPT,
        QWEN3VL_SYSTEM_PROMPT,
        QWEN3VL_USER_PROMPT,
    )

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list.")

    samples: List[Dict[str, Any]] = []
    stats = ConvertStats()

    def _history_from_conclusions(
        traj_list: List[Dict[str, Any]], cur_idx: int, quote_conclusion: bool
    ) -> str:
        parts: List[str] = []
        for j in range(cur_idx):
            c = str((traj_list[j] or {}).get("conclusion", "") or "")
            if c and quote_conclusion:
                c = "\"" + c + "\""
            parts.append(f"Step {j + 1}: {c}; ")
        return "".join(parts)

    for ep in tqdm(raw):
        save_dir = ep.get("save_dir", "")
        goal = ep.get("goal", "")
        traj = ep.get("trajectory", [])
        if not isinstance(traj, list):
            continue

        for i, step in enumerate(traj):
            if limit is not None and stats.written >= limit:
                return samples, stats

            stats.total_steps += 1
            try:
                img_rel = step.get("image", "")
                if not img_rel:
                    raise ValueError("missing step['image']")
                img_path = (
                    (image_root / img_rel).resolve()
                    if absolute_images
                    else (image_root / img_rel)
                )
                if not img_path.exists():
                    raise FileNotFoundError(f"image not found: {img_path}")

                if model_format == "qwen25vl":
                    # system prompt: resolution 来自 smart_resize(width,height) 的 resized_w x resized_h
                    w, h = _load_image_size(img_path)
                    resized_h, resized_w = smart_resize(
                        h, w, min_pixels=min_pixels, max_pixels=max_pixels
                    )
                    system_prompt = Qwen25VL_SYSTEM_PROMPT.format(
                        resolution=f"{resized_w}x{resized_h}"
                    )
                    user_prompt_template = QWEN25VL_USER_PROMPT
                else:
                    system_prompt = QWEN3VL_SYSTEM_PROMPT
                    user_prompt_template = QWEN3VL_USER_PROMPT

                # history:
                # - default: 用“上一步的 step_history”
                # - refine: 用“之前所有步的 conclusion”拼接，模拟 Qwen25VL 的推理 history
                if refine:
                    history = _history_from_conclusions(
                        traj,
                        i,
                        quote_conclusion=(model_format != "qwen3vl"),
                    )
                else:
                    history = ""
                    if i > 0:
                        history = str(traj[i - 1].get("step_history", "") or "")
                user_prompt = user_prompt_template.format(
                    instruction=goal, history=history
                )

                # ShareGPT 多模态：用 <image> 占位符关联 images[]
                # 对齐 seeact_v.py 的 message content 顺序：先 text，后 image
                user_content = user_prompt + "<image>"

                # assistant: thinking + tool_call（按 model_format 控制输出模板与坐标转换）
                response = str(step.get("response", "") or "")
                thinking_raw, tool_call = _extract_thinking_and_tool_call(response)
                if refine:
                    thinking = str(step.get("thinking_refine", "") or "")
                    if not str(thinking).strip():
                        stats.thinking_empty_filtered += 1
                        continue
                else:
                    thinking = thinking_raw
                    # thinking 为空：不写入训练样本，但仍保留在轨迹中用于后续 history 对齐
                    if not str(thinking).strip():
                        stats.thinking_empty_filtered += 1
                        continue
                if model_format == "qwen25vl":
                    tool_call_out = convert_tool_call_qwen3_to_qwen25(
                        tool_call, resized_w, resized_h
                    )
                else:
                    tool_call_out = tool_call
                tool_call_json = json.dumps(tool_call_out, ensure_ascii=False)

                conclusion = str(step.get("conclusion", "") or "")
                if model_format == "qwen3vl":
                    assistant_content = (
                        "Thought: "
                        + thinking.strip()
                        + "\nAction: "
                        + json.dumps(conclusion.strip(), ensure_ascii=False)
                        + "\n<tool_call>\n"
                        + tool_call_json
                        + "\n</tool_call>"
                    )
                elif refine:
                    assistant_content = (
                        "<thinking>\n"
                        + thinking.strip()
                        + "\n</thinking>\n"
                        + "<tool_call>\n"
                        + tool_call_json
                        + "\n</tool_call>\n"
                        + "<conclusion>\n"
                        + ("\"" + conclusion.strip() + "\"" if conclusion.strip() else "")
                        + "\n</conclusion>"
                    )
                else:
                    assistant_content = (
                        "<thinking>\n"
                        + thinking.strip()
                        + "\n</thinking>\n"
                        + "<tool_call>\n"
                        + tool_call_json
                        + "\n</tool_call>"
                    )

                step_idx = step.get("step", i)
                sample_id = f"{save_dir}_step{step_idx}"

                # images 字段：默认保持相对路径（与 input file 同目录时最方便）
                images_field = [str(img_path) if absolute_images else img_rel]

                # 过滤无效 step：不写入训练样本，但仍然让其“存在于轨迹里”，以保证后续 step 的 history 对齐不变
                # （history 仍然来自 traj[i-1].step_history，其中可能包含 is_valid=false 的步骤）
                if step.get("is_valid", True) is False:
                    stats.invalid_filtered += 1
                    continue

                # Mixed-policy 数据：若该 step 带 policy_tag，则仅保留 strong 作为训练样本。
                # 注意：这里仅过滤“写样本”，不会改变上面 history 的构造（history 仍基于完整轨迹）。
                if "policy_tag" in step and step.get("policy_source") != "strong":
                    stats.policy_filtered += 1
                    continue

                samples.append(
                    {
                        "id": sample_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ],
                        "images": images_field,
                        "bbox": step.get("bbox", []),
                    }
                )
                stats.written += 1
            except Exception as e:
                stats.skipped += 1
                if strict:
                    raise
                print(
                    f"[WARN] skip sample (save_dir={save_dir}, step={step.get('step', i)}): {e}",
                    file=sys.stderr,
                )

    return samples, stats


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert AndroidWorld trajectory json to ShareGPT (with system) for LLaMA-Factory."
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="input json path, e.g. .../data_merge_0126.json",
    )
    p.add_argument("--output", required=True, type=Path, help="output path (.json or .jsonl)")
    p.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="root dir for step['image'] (default: input.parent)",
    )
    p.add_argument(
        "--min-pixels",
        type=int,
        default=3136,
        help="Qwen2.5VL smart_resize min_pixels (default: 3136)",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=12845056,
        help="Qwen2.5VL smart_resize max_pixels (default: 12845056)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="max number of step-samples to write (for debugging)"
    )
    p.add_argument(
        "--absolute-images",
        action="store_true",
        help="write absolute image paths into `images`",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail fast on any bad record instead of skipping",
    )
    p.add_argument(
        "--refine",
        action="store_true",
        help="use refined fields (thinking_refine/conclusion) and conclusion-based history",
    )
    p.add_argument(
        "--model-format",
        type=str,
        default="qwen25vl",
        choices=["qwen25vl", "qwen3vl"],
        help="target model format (qwen25vl or qwen3vl). default: qwen25vl",
    )
    args = p.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    image_root: Path = args.image_root or input_path.parent

    samples, stats = build_sharegpt_samples(
        input_path=input_path,
        image_root=image_root,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        limit=args.limit,
        absolute_images=args.absolute_images,
        strict=args.strict,
        refine=args.refine,
        model_format=args.model_format,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    else:
        output_path.write_text(
            json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(
        f"[OK] written={stats.written}, skipped={stats.skipped}, "
        f"invalid_filtered={stats.invalid_filtered}, "
        f"thinking_empty_filtered={stats.thinking_empty_filtered}, "
        f"policy_filtered={stats.policy_filtered}, "
        f"total_steps={stats.total_steps}\n"
        f"     output={output_path}\n"
        f"     image_root={image_root}"
    )


if __name__ == "__main__":
    main()

