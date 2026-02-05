# from __future__ import annotations

# import argparse
# import base64
# import concurrent.futures as cf
# import json
# import os
# import re
# import time
# from pathlib import Path
# from threading import local
# from typing import Any, Dict, Tuple

# from openai import OpenAI
# from tqdm import tqdm

# _RE_THINKING = re.compile(r"<thinking>\s*([\s\S]*?)\s*</thinking>", re.I)
# _RE_TOOL_CALL = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.I)


# def _parse_thinking_tool_call(response: str) -> Tuple[str, Dict[str, Any], str]:
#     """返回 (thinking_text, tool_call_dict, tool_call_raw_json_str)."""
#     m_tool = _RE_TOOL_CALL.search(response or "")
#     if not m_tool:
#         raise ValueError("No <tool_call>...</tool_call> found.")
#     tool_raw = m_tool.group(1).strip()
#     tool_obj = json.loads(tool_raw)

#     m_th = _RE_THINKING.search(response or "")
#     thinking = m_th.group(1).strip() if m_th else ""
#     if not thinking:
#         thinking = " "
#     return thinking, tool_obj, tool_raw


# def _png_to_data_url(png_path: Path) -> str:
#     b = png_path.read_bytes()
#     return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")


# def _rewrite_thinking_to_action(
#     client: OpenAI,
#     model: str,
#     thinking: str,
#     tool_call_raw: str,
#     image_data_url: str,
#     request_timeout: float,
#     retries: int,
# ) -> str:
#     """
#     Call the model to extract ONE short English action sentence from <thinking>, e.g.:
#       I opened the Clock app to set an alarm for tomorrow morning.

#     Return ONLY that single sentence (no extra output).
#     """
#     system_prompt = (
#         "You are a rewriting assistant.\n"
#         "Your goal is to extract ONE short action sentence from a GUI agent's step-level thinking.\n"
#         "You will be given a model's <thinking>, the action in tool_call format and the screenshot for current step. Summarize the action taken/intended in ONE short English sentence.\n"
#         "Example: I opened the Clock app to set an alarm for tomorrow morning.\n"
#         "Example: I clicked the Confirm button to delete the selected item.\n\n"
#         "Rules:\n"
#         "- Output EXACTLY one English sentence.\n"
#         "- Do NOT output any prefixes like 'Thought:' or 'Action:'.\n"
#         "- Do NOT output <tool_call>.\n"
#         "- Do NOT add any extra text.\n"
#     )
#     user_text = (
#         "Given the current screenshot, the model's <thinking>, and the <tool_call>, extract the core action as ONE short English sentence.\n\n"
#         f"<thinking>\n{thinking}\n</thinking>\n"
#         f"\n<tool_call>\n{tool_call_raw}\n</tool_call>\n"
#     )

#     last_err: Exception | None = None
#     for _ in range(max(1, int(retries) + 1)):
#         try:
#             resp = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": user_text},
#                             {"type": "image_url", "image_url": {"url": image_data_url}},
#                         ],
#                     },
#                 ],
#                 temperature=0,
#                 timeout=float(request_timeout),
#             )
#             break
#         except Exception as e:
#             last_err = e
#             time.sleep(1.0)
#     else:
#         raise last_err  # pragma: no cover
#     out = (resp.choices[0].message.content or "").strip()
#     # 兜底：只取第一行非空内容
#     lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
#     if not lines:
#         raise ValueError("Empty rewrite output.")
#     return lines[0]


# def _one_line(text: str) -> str:
#     """Collapse whitespace into one line (avoid breaking Thought/Action lines)."""
#     t = re.sub(r"\s+", " ", (text or "").strip())
#     return t if t else " "


# def main() -> None:
#     ap = argparse.ArgumentParser(description="Rewrite Gemini <thinking> into Qwen3VL Thought/Action and save a new json.")
#     ap.add_argument("--input", type=Path, required=True, help="e.g. runs/diy_0126/data_merge_0126.json")
#     ap.add_argument("--output", type=Path, required=True, help="e.g. runs/diy_0126/data_merge_0126_qwen3vl.json")
#     ap.add_argument("--image-root", type=Path, default=None, help="default: input.parent")
#     ap.add_argument("--model", type=str, default=os.getenv("REWRITE_MODEL", "gemini-3-pro-preview"), help="OpenAI-compatible model name")
#     ap.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.ppchat.vip/v1"))
#     ap.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "sk-msxoHjQv4oPTtngKR9VzDLQn4WQ1Ge9b5H12YHUF7aeosivs"))
#     ap.add_argument("--sleep", type=float, default=0.0, help="optional sleep seconds between requests")
#     ap.add_argument("--workers", type=int, default=8, help="thread workers (default: 8)")
#     ap.add_argument("--request-timeout", type=float, default=90.0, help="per-request timeout seconds (default: 180)")
#     ap.add_argument("--stall-timeout", type=float, default=180.0, help="stall watchdog: no completion within this time => cancel remaining (default: 300)")
#     ap.add_argument("--retries", type=int, default=1, help="retries per request on exception (default: 1)")
#     args = ap.parse_args()

#     if not args.api_key:
#         raise RuntimeError("Missing OPENAI_API_KEY (or pass --api-key).")

#     input_path = args.input
#     output_path = args.output
#     image_root = args.image_root or input_path.parent

#     data = json.loads(input_path.read_text(encoding="utf-8"))
#     if not isinstance(data, list):
#         raise ValueError("Input JSON must be a list.")

#     _tl = local()

#     def _get_client() -> OpenAI:
#         c = getattr(_tl, "client", None)
#         if c is None:
#             c = OpenAI(api_key=args.api_key, base_url=args.base_url)
#             _tl.client = c
#         return c

#     def _process_one(step: Dict[str, Any]) -> str:
#         resp_txt = str(step.get("response", "") or "")
#         thinking, _tool_obj, tool_raw = _parse_thinking_tool_call(resp_txt)

#         img_rel = step.get("image", "")
#         if not img_rel:
#             raise ValueError("Missing step['image'].")
#         img_path = (image_root / img_rel)
#         if not img_path.exists():
#             raise FileNotFoundError(f"Image not found: {img_path}")

#         action_short = _rewrite_thinking_to_action(
#             client=_get_client(),
#             model=args.model,
#             thinking=thinking,
#             tool_call_raw=tool_raw,
#             image_data_url=_png_to_data_url(img_path),
#             request_timeout=args.request_timeout,
#             retries=args.retries,
#         )

#         if args.sleep > 0:
#             time.sleep(args.sleep)

#         return (
#             "Thought: "
#             + _one_line(thinking)
#             + "\nAction: \""
#             + _one_line(action_short)
#             + "\"\n<tool_call>\n"
#             + tool_raw
#             + "\n</tool_call>"
#         )

#     steps: list[Dict[str, Any]] = []
#     for ep in data:
#         traj = ep.get("trajectory", [])
#         if isinstance(traj, list):
#             for step in traj:
#                 if isinstance(step, dict):
#                     steps.append(step)

#     done = 0
#     with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
#         fut_to_step = {ex.submit(_process_one, s): s for s in steps}
#         pending = set(fut_to_step.keys())
#         pbar = tqdm(total=len(fut_to_step))
#         while pending:
#             it = cf.as_completed(pending, timeout=float(args.stall_timeout))
#             try:
#                 fut = next(it)
#             except cf.TimeoutError:
#                 # 卡住：取消剩余
#                 for pf in list(pending):
#                     pf.cancel()
#                 break

#             pending.discard(fut)
#             step = fut_to_step[fut]
#             try:
#                 step["response_qwen3vl"] = fut.result()
#             except Exception as e:
#                 # 保留错误信息，方便事后排查；不中断整体转换
#                 step["response_qwen3vl"] = None
#                 step["response_qwen3vl_error"] = str(e)
#             done += 1
#             pbar.update(1)
#         pbar.close()

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
#     print(f"[OK] wrote: {output_path}  (rewritten_steps={done})")


# if __name__ == "__main__":
#     main()


# Convert to Qwen3-VL training recipe

import argparse
import json
import re
import sys
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List


def _ensure_androidworld_on_syspath() -> None:
    # This file lives in evaluation/AndroidWorld/
    aw_root = Path(__file__).resolve().parent
    if str(aw_root) not in sys.path:
        sys.path.insert(0, str(aw_root))


def _extract_action_text_qwen3vl(response_text: str) -> str:
    """
    Match seeact_v.py (_extract_action_text_qwen3vl):
    extract Action: ... for step history (does not affect execution).
    """
    m = re.search(r"Action:\s*(.+?)(?:\n<tool_call>|$)", response_text or "", flags=re.S)
    if not m:
        return ""
    text = m.group(1).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    return text.replace("\n", " ").strip()


def _user_text_with_n_images(user_prompt: str, n_images: int) -> str:
    # Align seeact_v.py ordering: text first, then images
    if n_images <= 0:
        return user_prompt
    return user_prompt + "<image>" + ("\n<image>" * (n_images - 1))


def convert_to_sharegpt_qwen3vl(
    input_path: Path,
    output_path: Path,
    image_root: Path,
    last_n: int = 3,
) -> None:
    _ensure_androidworld_on_syspath()
    from android_world.agents.PROMPT import (
        QWEN3VL_SYSTEM_PROMPT,
        QWEN3VL_SYSTEM_PROMPT_LASTN,
        QWEN3VL_USER_PROMPT,
    )

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")

    out: List[Dict[str, Any]] = []

    for ep in tqdm(data):
        save_dir = ep.get("save_dir", "")
        goal = ep.get("goal", "")
        traj = ep.get("trajectory", [])
        if not isinstance(traj, list):
            continue

        step_his = ""  # Qwen3VL history string (Action line only)

        for i, step in enumerate(traj):
            step_idx = step.get("step", i)
            sample_id = f"{save_dir}_step{step_idx}"

            # system prompt selection matches seeact_v.py (LASTN when last_N>1)
            system_prompt = (
                QWEN3VL_SYSTEM_PROMPT_LASTN if last_n and last_n > 1 else QWEN3VL_SYSTEM_PROMPT
            )
            user_prompt = QWEN3VL_USER_PROMPT.format(instruction=goal, history=step_his)

            # last N screenshots (oldest -> newest), include current step screenshot
            start = max(0, i - (last_n - 1))
            imgs: List[str] = []
            for j in range(start, i + 1):
                img_rel = traj[j].get("image", "")
                if not img_rel:
                    raise ValueError(f"Missing trajectory[{j}].image for {save_dir}")
                img_path = image_root / img_rel
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                imgs.append(img_rel)

            user_content = _user_text_with_n_images(user_prompt, len(imgs))

            assistant = step.get("response_qwen3vl", "")
            if not isinstance(assistant, str) or not assistant.strip():
                # Some steps may have failed during rewriting; skip them.
                continue

            out.append(
                {
                    "id": sample_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant},
                    ],
                    "images": imgs,
                }
            )

            # update history for next step (use current step's Action line)
            action_txt = _extract_action_text_qwen3vl(assistant)
            if action_txt:
                step_his += f"Step {i+1}: {action_txt}; "

    print(f"total steps: {len(out)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert data_merge_0126_qwen3vl.json to ShareGPT trainset for Qwen3VL.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("runs/diy_0126/data_merge_0126_qwen3vl.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("runs/diy_0126/0126_train_qwen3vl.json"),
    )
    ap.add_argument("--image-root", type=Path, default=Path("runs/diy_0126"))
    ap.add_argument("--last-n", type=int, default=3)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    convert_to_sharegpt_qwen3vl(
        input_path=(base / args.input),
        output_path=(base / args.output),
        image_root=(base / args.image_root),
        last_n=int(args.last_n),
    )
    print(f"[OK] wrote: {base / args.output}")


if __name__ == "__main__":
    main()


import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


def _ensure_androidworld_on_syspath() -> None:
    aw_root = Path(__file__).resolve().parent
    if str(aw_root) not in sys.path:
        sys.path.insert(0, str(aw_root))


def _extract_action_text_qwen3vl(block: str) -> str:
    """Match seeact_v.py: extract the 'Action:' line for step history."""
    m = re.search(r"Action:\s*(.+?)(?:\n<tool_call>|$)", block or "", flags=re.S)
    if not m:
        return ""
    text = m.group(1).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    return text.replace("\n", " ").strip()


def _user_content_with_images(user_prompt: str, n_images: int) -> str:
    if n_images <= 0:
        return user_prompt
    # user_prompt 本身以 \n 结尾（见 PROMPT.py），这里直接追加多个 <image>
    return user_prompt + "<image>" + ("\n<image>" * (n_images - 1))


### (Removed) old duplicate entrypoint & converter.
### Kept only the new converter below "# Convert to Qwen3-VL training recipe".