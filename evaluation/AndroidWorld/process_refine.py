from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import random
import re
import threading
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINKING_RE = re.compile(r"<thinking>\s*([\s\S]*?)\s*</thinking>", re.I)

CONCLUSION_SYSTEM_PROMPT = """You are a high-quality GUI trajectory annotator.

Task:
- You will receive one Android GUI step with:
  1) user instruction (overall goal),
  2) step history text,
  3) the model response (maybe include thinking and conclusion) for the current step,
  4) current screenshot (possibly with a red marker if the step is click/long_press).
- Your task is to write or rewrite a concise conclusion of the action executed in the current step.

Output rule:
- The conclusion should summarize this step operation.
- Write 1-2 English sentences as <conclusion> for this step.
- For simple thinking, directly describe what was done in this step.
- For complex thinking (e.g., reflection after a wrong previous move, or trying a new strategy), describe the cause and effect around this step.
- If the thinking contains task-instruction-related information that should be remembered for later steps, include it in the conclusion.
- The conclusion should summarize the intended action and immediate progress/result for this step.
- Keep it factual and grounded in the provided text + screenshot.
- Output plain conclusion text only (no <conclusion> tags, no quotes, no extra wrappers)

Examples:
1) I opened the Audio Recorder app from the app drawer.
2) I mistakenly entered text in the filename field in the previous step, so in this step I tried to use a long press to select all and delete it.
"""

THINKING_SYSTEM_PROMPT = """You are a high-quality GUI trajectory annotator.

Task:
- You will receive one Android GUI step with:
  1) user instruction (overall goal),
  2) step history text,
  3) the model response for the current step,
  4) current screenshot (possibly with a red marker if the step is click/long_press).
- In previous data, the model response may miss <thinking> entirely or contain overly brief thinking.
- Your task is to synthesize from instruction, history, response, and screenshot, and reconstruct a detailed, in-depth thinking with clear reasoning.
- Note that previous steps may contain mistakes; if correction is needed, analyze it carefully in the thinking.

Output rule:
- Write a refined, in-depth thinking paragraph for this step.
- The thinking should explain the user instruction, analyze the current screen state, analyze prior history (including possible reflection), and then reason toward the current action.
- The thinking should simulate the agent's forward reasoning process, not post-hoc verification.
- DO NOT use patterns like "because the action is xxx, it means xxx"; instead, provide a plausible step-by-step forward rationale.
- DO NOT include any specific coordinates in the thinking.
- The thinking length should be around 100-200 words.
- Keep it factual and grounded in the provided text + screenshot.
- Output plain thinking text only (no <thinking> tags, no quotes, no extra wrappers).
"""

EVAL_SYSTEM_PROMPT = """You are a Android GUI action evaluator.

Task:
- You will receive one Android GUI step with:
  1) user instruction (overall goal),
  2) step history text,
  3) the model response for the current step,
  4) current screenshot (possibly with a red marker if the step is click/long_press).

Evaluate whether the current step action is reasonable and correct in context.
Common unreasonable cases include:
- wrong click target / wrong region,
- repeated meaningless action,
- obviously unreasonable action for current UI state,
- there exists an obvious better immediate action.

Output rule:
- Output exactly one token: True or False.
- True means the step is reasonable and correct enough.
- False means the step is not reasonable.
- Do not output any other text.
"""

PATTERN_PROMPT = """You are encouraged to generate a richer thinking pattern by following the selected pattern directives below.
Your thinking length can be extended to around 200-300 words.
No matter which patterns are selected, your final reasoning result must be strictly consistent with the action implied by the current step response.
The final chosen action in your reasoning must align with the step action, and must not contradict it.
Your final output should be one coherent chain-of-thought paragraph.
Do not explicitly mention which thinking pattern(s) you used."""

PATTERN_SELF_CHECK = """[Pattern: Self-check]
After proposing a plausible action, add a self-check using wording like "Wait, let me check again ...", then re-verify key evidence and confirm the same action."""

PATTERN_OUTCOME_PREDICTION = """[Pattern: Outcome prediction]
After proposing an action, reason about likely immediate outcomes and possible UI transitions, then confirm why this action is still the best choice now."""

PATTERN_SELF_DOUBT_CORRECTION = """[Pattern: Self-doubt and correction]
First consider a related but likely wrong action, then use "Wait, ..." to correct yourself and converge to the final correct action with clear justification."""

PATTERN_MULTI_PATH = """[Pattern: Multi-path comparison]
List multiple plausible options visible on the current screen, analyze consequences of each, and then select and justify the final action."""

PATTERN_POOL = [
    PATTERN_SELF_CHECK,
    PATTERN_OUTCOME_PREDICTION,
    PATTERN_SELF_DOUBT_CORRECTION,
    PATTERN_MULTI_PATH,
]


def _iter_tool_calls(response_text: str):
    for m in TOOL_CALL_RE.finditer(response_text or ""):
        yield json.loads(m.group(1))


def _extract_action_and_coord(response_text: str) -> tuple[str | None, list[float] | None]:
    for call in _iter_tool_calls(response_text or ""):
        if call.get("name") != "mobile_use":
            continue
        args = call.get("arguments") or {}
        action = args.get("action")
        coord = args.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            return (str(action) if action is not None else None, [float(coord[0]), float(coord[1])])
        return (str(action) if action is not None else None, None)
    return (None, None)


def _extract_thinking_from_response(response_text: str) -> str:
    m = THINKING_RE.search(response_text or "")
    return m.group(1).strip() if m else ""


def _word_count_by_space(text: str) -> int:
    t = (text or "").strip()
    return len(t.split()) if t else 0


def _avg_original_thinking_words(data: list[dict]) -> float:
    total_steps = 0
    total_words = 0
    for ep in data:
        for step in ep.get("trajectory", []):
            total_steps += 1
            th = _extract_thinking_from_response(str(step.get("response", "") or ""))
            total_words += _word_count_by_space(th)
    return (total_words / total_steps) if total_steps > 0 else 0.0


def _avg_refined_thinking_words(data: list[dict]) -> float:
    total_steps = 0
    total_words = 0
    for ep in data:
        for step in ep.get("trajectory", []):
            total_steps += 1
            total_words += _word_count_by_space(str(step.get("thinking_refine", "") or ""))
    return (total_words / total_steps) if total_steps > 0 else 0.0


def _coord_0_999_to_pixel(coord: list[float], width: int, height: int) -> tuple[int, int]:
    x = float(coord[0])
    y = float(coord[1])
    xp = int(round(x / 999.0 * float(width)))
    yp = int(round(y / 999.0 * float(height)))
    xp = max(0, min(width, xp))
    yp = max(0, min(height, yp))
    return xp, yp


def _image_to_data_url_with_optional_marker(
    image_path: Path,
    action: str | None,
    coord: list[float] | None,
    marked_output_path: Path | None = None,
) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        marked = False
        if action in {"click", "long_press"} and coord is not None:
            x, y = _coord_0_999_to_pixel(coord, im.width, im.height)
            draw = ImageDraw.Draw(im)
            r = max(18, int(min(im.width, im.height) * 0.04))
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                outline=(255, 0, 0),
                width=max(4, int(r * 0.28)),
            )
            marked = True

        if marked and marked_output_path is not None:
            marked_output_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(marked_output_path)

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


def _normalize_bool(text: str) -> bool:
    t = (text or "").strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    raise ValueError(f"Bad eval output: {text}")


def _build_user_text(instruction: str, step_history: str, response: str, mode: str) -> str:
    if mode == "conclusion":
        tail = "Now output ONE concise conclusion for this step."
    elif mode in {"thinking", "thinking_pattern"}:
        tail = "Now output ONE refined in-depth thinking paragraph for this step."
    else:
        tail = "Now evaluate whether this step action is reasonable and correct. Output exactly True or False."
    return (
        "Instruction:\n"
        f"{instruction}\n\n"
        "Step history:\n"
        f"{step_history}\n\n"
        "Current step model response:\n"
        f"{response}\n\n"
        + tail
    )


def _normalize_one_line(text: str) -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    if not out:
        raise ValueError("Empty conclusion returned by model.")
    return out


def _normalize_paragraph(text: str) -> str:
    out = (text or "").strip()
    if not out:
        raise ValueError("Empty thinking returned by model.")
    return out


def _gen_text(
    client: OpenAI,
    model: str,
    instruction: str,
    step_history: str,
    response: str,
    image_data_url: str,
    request_timeout: float,
    mode: str,
    interactive: bool = False,
) -> str:
    if mode == "conclusion":
        system_prompt = CONCLUSION_SYSTEM_PROMPT
    elif mode == "thinking":
        system_prompt = THINKING_SYSTEM_PROMPT
    elif mode == "thinking_pattern":
        k = random.choice([1, 2]) if random.random() < 0.8 else 3
        picked = random.sample(PATTERN_POOL, k=k)
        system_prompt = THINKING_SYSTEM_PROMPT + "\n\n" + PATTERN_PROMPT + "\n\n" + "\n\n".join(picked)
    else:
        system_prompt = EVAL_SYSTEM_PROMPT
    user_text = _build_user_text(instruction, step_history, response, mode=mode)

    # 如果response本身包含<conclusion>，删除这部分

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        temperature=0.4,
        timeout=request_timeout,
    )
    content = completion.choices[0].message.content
    if interactive:
        print(user_text)
        print("\n--- SYSTEM PROMPT ---\n")
        print(system_prompt)
        print("\n--- MODEL OUTPUT ---\n")
        print(content)
        input()
    if mode == "eval":
        return _normalize_bool(content)
    if mode == "conclusion":
        return _normalize_one_line(content)
    return _normalize_paragraph(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine per-step conclusion or thinking for *_refine.json.")
    parser.add_argument("--input", type=Path, required=True, help="e.g. runs/diy_0208/data_merge_0208_refine.json")
    parser.add_argument("--output", type=Path, default=None, help="e.g. runs/diy_0208/data_merge_0208_refine_conclusion.json")
    parser.add_argument("--image-root", type=Path, default=None, help="default: input.parent")
    parser.add_argument("--model", type=str, default=os.getenv("CONCLUSION_MODEL", "gemini-3-pro-preview"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.ppchat.vip/v1"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "sk-msxoHjQv4oPTtngKR9VzDLQn4WQ1Ge9b5H12YHUF7aeosivs"))
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mode", type=str, default="conclusion", choices=["conclusion", "thinking", "thinking_pattern", "eval"])
    parser.add_argument("--interactive", action="store_true", help="run step-by-step with print/input")
    parser.add_argument(
        "--save-temp-images",
        action="store_true",
        help="save click/long_press marked images under <image_root>/temp_images",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if args.output is not None:
        output_path: Path = args.output
    elif args.mode == "conclusion":
        output_path = input_path.with_name(f"{input_path.stem}_conclusion.json")
    elif args.mode == "thinking":
        output_path = input_path.with_name(f"{input_path.stem}_thinking.json")
    elif args.mode == "thinking_pattern":
        output_path = input_path.with_name(f"{input_path.stem}_thinking_pattern.json")
    else:
        output_path = input_path.with_name(f"{input_path.stem}_eval.json")
    image_root: Path = args.image_root or input_path.parent
    temp_images_root = image_root / "temp_images" if args.save_temp_images else None
    if temp_images_root is not None:
        temp_images_root.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or pass --api-key).")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    tl = threading.local()
    if args.mode in {"thinking", "thinking_pattern"}:
        print(f"avg_words_original_thinking: {_avg_original_thinking_words(data):.2f}")

    jobs: list[tuple[int, int, str, str, str, str]] = []
    for ep_idx, ep in enumerate(data):
        instruction = ep["goal"]
        traj = ep["trajectory"]
        for i, step in enumerate(traj):
            response = step["response"]
            if i == 0:
                step_history = ""
            else:
                step_history = traj[i - 1]["step_history"]
            image_rel = step["image"]
            jobs.append((ep_idx, i, instruction, step_history, response, image_rel))

    def _get_client() -> OpenAI:
        c = getattr(tl, "client", None)
        if c is None:
            c = OpenAI(api_key=args.api_key, base_url=args.base_url)
            tl.client = c
        return c

    def _process_one(job: tuple[int, int, str, str, str, str]) -> tuple[int, int, str]:
        ep_idx, step_idx, instruction, step_history, response, image_rel = job
        action, coord = _extract_action_and_coord(response)
        image_path = image_root / image_rel
        marked_output_path = (temp_images_root / image_rel) if temp_images_root is not None else None
        image_data_url = _image_to_data_url_with_optional_marker(
            image_path, action, coord, marked_output_path=marked_output_path
        )
        text_out = _gen_text(
            client=_get_client(),
            model=args.model,
            instruction=instruction,
            step_history=step_history,
            response=response,
            image_data_url=image_data_url,
            request_timeout=float(args.request_timeout),
            mode=args.mode,
            interactive=args.interactive,
        )
        if args.sleep > 0:
            time.sleep(args.sleep)
        return ep_idx, step_idx, text_out

    if args.interactive:
        for job in tqdm(jobs, total=len(jobs), desc="steps"):
            try:
                ep_idx, step_idx, text_out = _process_one(job)
                if args.mode == "conclusion":
                    key = "conclusion"
                elif args.mode == "thinking":
                    key = "thinking_refine"
                elif args.mode == "thinking_pattern":
                    key = "thinking_pattern"
                else:
                    key = "is_reasonable"
                data[ep_idx]["trajectory"][step_idx][key] = text_out
            except Exception:
                ep_idx, step_idx, *_ = job
                if args.mode == "conclusion":
                    key = "conclusion"
                    data[ep_idx]["trajectory"][step_idx][key] = ""
                elif args.mode == "thinking":
                    key = "thinking_refine"
                    data[ep_idx]["trajectory"][step_idx][key] = ""
                elif args.mode == "thinking_pattern":
                    key = "thinking_pattern"
                    data[ep_idx]["trajectory"][step_idx][key] = ""
                else:
                    key = "is_reasonable"
                    data[ep_idx]["trajectory"][step_idx][key] = False
    else:
        with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            fut_to_job = {ex.submit(_process_one, job): job for job in jobs}
            for fut in tqdm(cf.as_completed(fut_to_job), total=len(fut_to_job), desc="steps"):
                try:
                    ep_idx, step_idx, text_out = fut.result()
                    if args.mode == "conclusion":
                        key = "conclusion"
                    elif args.mode == "thinking":
                        key = "thinking_refine"
                    elif args.mode == "thinking_pattern":
                        key = "thinking_pattern"
                    else:
                        key = "is_reasonable"
                    data[ep_idx]["trajectory"][step_idx][key] = text_out
                except Exception:
                    ep_idx, step_idx, *_ = fut_to_job[fut]
                    if args.mode == "conclusion":
                        key = "conclusion"
                        data[ep_idx]["trajectory"][step_idx][key] = ""
                    elif args.mode == "thinking":
                        key = "thinking_refine"
                        data[ep_idx]["trajectory"][step_idx][key] = ""
                    elif args.mode == "thinking_pattern":
                        key = "thinking_pattern"
                        data[ep_idx]["trajectory"][step_idx][key] = ""
                    else:
                        key = "is_reasonable"
                        data[ep_idx]["trajectory"][step_idx][key] = False

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode == "thinking":
        print(f"avg_words_thinking_refine: {_avg_refined_thinking_words(data):.2f}")
    if args.mode == "thinking_pattern":
        total_steps = 0
        total_words = 0
        for ep in data:
            for step in ep.get("trajectory", []):
                total_steps += 1
                total_words += _word_count_by_space(str(step.get("thinking_pattern", "") or ""))
        avg_words_thinking_pattern = (total_words / total_steps) if total_steps > 0 else 0.0
        print(f"avg_words_thinking_pattern: {avg_words_thinking_pattern:.2f}")
    if args.mode == "eval":
        total_steps = 0
        false_count = 0
        true_count = 0
        for ep in data:
            for step in ep.get("trajectory", []):
                total_steps += 1
                if bool(step.get("is_reasonable", False)):
                    true_count += 1
                else:
                    false_count += 1
        print(f"is_reasonable_true: {true_count}")
        print(f"is_reasonable_false: {false_count}")
        print(f"is_reasonable_total: {total_steps}")
    print(f"[OK] saved: {output_path}")


if __name__ == "__main__":
    main()

