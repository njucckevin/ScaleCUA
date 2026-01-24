# import os
# import json
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import re
import os
import sys
import requests


# data_test = [
#     {"base_task_name": "AudioRecorderRecordAudio", "instruction": "Create a new contact person with the name cckevin and the phone number 18851134288.", "sample_id": "0"},
#     {"base_task_name": "ExpenseDeleteMultiple", "instruction": "Delete the Ride-Sharing expenses from pro expense app", "sample_id": "1"},
# ]
# json.dump(data_test, open("data_test.json", "w"), indent=4)


# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any


# def _read_json(path: Path) -> dict[str, Any] | None:
#     try:
#         with path.open("r", encoding="utf-8") as f:
#             obj = json.load(f)
#         return obj if isinstance(obj, dict) else None
#     except Exception:
#         return None


# def _last_step_has_enter_button(response: str) -> bool:
#     # Common patterns observed in trajectory "response" field.
#     needles = [
#         '"button": "Enter"',
#         '"button":"Enter"',
#         '"button": "enter"',
#         '"button":"enter"',
#         '"button": "Menu"',
#         '"button":"Menu"',
#         '"button": "menu"',
#         '"button":"menu"',
#     ]
#     return any(n in response for n in needles)


# def main() -> None:
#     # Use relative path so it works across machines.
#     base = Path(__file__).resolve().parent / "runs" / "gemini3pro_last3"
#     if not base.exists():
#         raise FileNotFoundError(f"Base path not found: {base}")

#     hit_tasks: list[str] = []
#     total_tasks = 0
#     total_with_result = 0

#     for task_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
#         total_tasks += 1
#         # Be tolerant to naming: some users may write "results.json" by mistake.
#         result_path = task_dir / "result.json"
#         if not result_path.exists():
#             continue

#         data = _read_json(result_path)
#         if not data:
#             continue
#         total_with_result += 1

#         traj = data.get("trajectory", [])
#         if not isinstance(traj, list) or not traj:
#             continue
#         last = traj[-1]
#         if not isinstance(last, dict):
#             continue
#         resp = last.get("response", "")
#         if not isinstance(resp, str):
#             resp = str(resp)

#         if _last_step_has_enter_button(resp):
#             hit_tasks.append(task_dir.name)

#     print(f"Base: {base}")
#     print(f"Task dirs total: {total_tasks}")
#     print(f"Task dirs with result.json: {total_with_result}")
#     print(f'Last step contains \\"button\\": \\"Enter\\" count: {len(hit_tasks)}')
#     if hit_tasks:
#         print("Hit tasks:")
#         for name in hit_tasks:
#             print(f"- {name}")


# if __name__ == "__main__":
#     main()


# ---------------------------
# Explore results statistics
# ---------------------------

# _EXPLORE_TRAJ_DIR = Path(__file__).resolve().parent / "explore_results" / "trajectories"


# def _summarize_explore_trajectories(traj_dir: Path = _EXPLORE_TRAJ_DIR) -> None:
#     """统计 explore_results/trajectories 下的轨迹数量与 step 分布。

#     - 轨迹条数：json 文件数量
#     - step 数：每个 json 文件内 list 的长度
#     - app 聚合：每个文件用第一个 step 的 "app" 字段归类（文件级别 app）
#     """
#     if not traj_dir.exists():
#         print(f"[STATS] Trajectory dir not found: {traj_dir}")
#         return

#     traj_files = sorted(traj_dir.glob("*.json"))
#     total_traj = len(traj_files)
#     total_steps = 0
#     steps_per_traj: list[int] = []
#     steps_by_app: Counter[str] = Counter()
#     bad_files: list[str] = []

#     for p in traj_files:
#         try:
#             obj = json.loads(p.read_text(encoding="utf-8"))
#         except Exception:
#             bad_files.append(p.name)
#             continue

#         if not isinstance(obj, list):
#             bad_files.append(p.name)
#             continue

#         n_steps = len(obj)
#         steps_per_traj.append(n_steps)
#         total_steps += n_steps

#         app_name = "UNKNOWN"
#         if n_steps > 0 and isinstance(obj[0], dict):
#             app_name = str(obj[0].get("app", "UNKNOWN"))
#         steps_by_app[app_name] += n_steps

#     print(f"[STATS] Trajectory dir: {traj_dir}")
#     print(f"[STATS] Trajectories: {total_traj}")
#     print(f"[STATS] Total steps: {total_steps}")
#     if steps_per_traj:
#         print(
#             "[STATS] Steps per trajectory: "
#             f"min={min(steps_per_traj)} max={max(steps_per_traj)} avg={total_steps/len(steps_per_traj):.2f}"
#         )

#     print("[STATS] Steps by app:")
#     for app, cnt in steps_by_app.most_common():
#         print(f"  - {app}: {cnt}")

#     if bad_files:
#         print(f"[STATS] Bad/unreadable trajectory files: {len(bad_files)}")
#         for name in bad_files[:20]:
#             print(f"  - {name}")
#         if len(bad_files) > 20:
#             print("  - ...")


# # 直接运行 temp.py 时打印统计（不需要 main 函数）
# if __name__ == "__main__":
#     _summarize_explore_trajectories()


# def _test_openai_api_hello() -> None:
#     """Minimal OpenAI API connectivity test (real API call).

#     Usage:
#       OPENAI_API_KEY=... python evaluation/AndroidWorld/temp.py
#       OPENAI_MODEL=gpt-4o-mini OPENAI_API_KEY=... python evaluation/AndroidWorld/temp.py
#     """
#     api_key = os.environ.get("OPENAI_API_KEY", "").strip()
#     if not api_key:
#         print("[SKIP] OPENAI_API_KEY 未设置，跳过 OpenAI API 联网测试。")
#         return

#     model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
#     print(f"[TEST] Calling OpenAI Chat Completions with model={model!r} ...")

#     headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
#     payload = {
#         "model": model,
#         "messages": [{"role": "user", "content": "Hello, who are you?"}],
#         "max_tokens": 16,
#         "temperature": 0.0,
#     }
#     resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
#     print("[TEST] status_code:", resp.status_code)
#     try:
#         data = resp.json()
#     except Exception:
#         print("[TEST] Non-JSON response:", resp.text[:500])
#         return

#     if resp.status_code != 200:
#         print("[TEST] error response:", json.dumps(data, ensure_ascii=False, indent=2)[:2000])
#         return

#     content = data["choices"][0]["message"]["content"]
#     print("[TEST] ok, content:", repr(content))


# if __name__ == "__main__":
#     _test_openai_api_hello()

# import pickle
# import os
# import json
# from tqdm import tqdm

# import sys
# # Use relative import path so it works across machines.
# sys.path.insert(0, str(Path(__file__).resolve().parent))

# params_dir = "./params"
# syn_tasks = json.load(open("./synthesized_tasks_0121_v1_eval_final_taskid_filter.json", "r"))
# for item in tqdm(syn_tasks):
#     task_id = item["task_id"]
#     params_path = os.path.join(params_dir, f"{task_id}_params.pkl")
#     if not os.path.exists(params_path):
#         print(f"Params not found: {params_path}")
#         continue
#     if task_id == 'd327e1da-93d6-42e8-8d5a-38c96b883e47':
#         params = pickle.load(open(params_path, "rb"))
#         print(params)
#         input()
#     params = pickle.load(open(params_path, "rb"))



# def _should_drop(base_task_name: str) -> bool:
#     # Prefix-based drops.
#     drop_prefixes = ("SportsTracker", "Tasks", "Notes", "Retro", "SimpleSmsReply")
#     if base_task_name.startswith(drop_prefixes):
#         return True
#     # Exact-name drops.
#     if base_task_name in ("FilesDeleteFile", "FilesMoveFile"):
#         return True
#     return False


# _DATE_PREFIX_RE = re.compile(r"\b20\d{2}[_-]\d{2}[_-]\d{2}\b")
# _UUID_RE = re.compile(
#     r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
# )
# # Random-looking 4-char token used by generate_modified_file_name(random_suffix), near extension.
# _RANDOM_4_NEAR_EXT_RE = re.compile(r"(?:^|[_-])[A-Za-z0-9]{4}(?=\.[A-Za-z0-9]{2,5}$)")
# # Random-looking short prefix sometimes used in generated names (e.g., 0bSM_...).
# _RANDOM_PREFIX_4_RE = re.compile(r"^[A-Za-z0-9]{4}[_-].+\.[A-Za-z0-9]{2,5}$")


# def _mentions_noise_filename(instruction: str) -> bool:
#     """Heuristic: instruction references a likely noise/variant filename from screenshots.

#     Examples: "2023_03_23_some_file..." or truncated names with '...' / '…'.
#     """
#     if not instruction:
#         return False

#     # Date-style prefix is a strong signal of generated variants.
#     if _DATE_PREFIX_RE.search(instruction):
#         return True

#     # UUID-like tokens are typically generated and hard to reproduce without exact params/seed.
#     if _UUID_RE.search(instruction):
#         return True

#     # Work on quoted segments to avoid false positives in general prose.
#     quoted = re.findall(r"['\"]([^'\"]+)['\"]", instruction)
#     for s in quoted:
#         s = s.strip()
#         if not s:
#             continue

#         # Truncated filenames are usually screenshot artifacts, not stable identifiers.
#         if ("..." in s) or ("…" in s):
#             return True

#         # If it looks like a filename (has an extension), check for common random-variant patterns.
#         if "." in s and re.search(r"\.[A-Za-z0-9]{2,5}$", s):
#             if _RANDOM_PREFIX_4_RE.search(s):
#                 return True
#             if _RANDOM_4_NEAR_EXT_RE.search(s):
#                 return True

#     return False


# def main() -> None:
#     base_dir = Path(__file__).resolve().parent
#     in_path = base_dir / "synthesized_tasks_0121_v1_eval_final_taskid.json"
#     out_path = base_dir / "synthesized_tasks_0121_v1_eval_final_taskid_filter.json"

#     data = json.loads(in_path.read_text(encoding="utf-8"))
#     if not isinstance(data, list):
#         raise ValueError("Input JSON must be a list.")

#     kept = []
#     dropped = 0
#     dropped_by_reason = Counter()
#     for i, item in enumerate(data):
#         if not isinstance(item, dict):
#             dropped += 1
#             dropped_by_reason["bad_item"] += 1
#             continue
#         base_task_name = item.get("base_task_name")
#         if not isinstance(base_task_name, str) or not base_task_name:
#             dropped += 1
#             dropped_by_reason["missing_base_task_name"] += 1
#             continue
#         instruction = item.get("instruction", "")
#         if not isinstance(instruction, str):
#             instruction = str(instruction)
#         if _should_drop(base_task_name):
#             dropped += 1
#             dropped_by_reason["base_task_rule"] += 1
#             continue
#         if _mentions_noise_filename(instruction):
#             dropped += 1
#             dropped_by_reason["noise_filename_in_instruction"] += 1
#             continue
#         kept.append(item)

#     out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

#     app_counts = Counter()
#     for item in kept:
#         app = item.get("app", "UNKNOWN")
#         app_counts[str(app)] += 1

#     print(f"Input: {in_path}")
#     print(f"Output: {out_path}")
#     print(f"Total: {len(data)} | Kept: {len(kept)} | Dropped: {dropped}")
#     if dropped_by_reason:
#         print("\nDropped by reason:")
#         for reason, cnt in dropped_by_reason.most_common():
#             print(f"- {reason}: {cnt}")
#     print("\nCounts by app:")
#     for app, cnt in app_counts.most_common():
#         print(f"- {app}: {cnt}")


# if __name__ == "__main__":
#     main()


import random

syn_data = json.load(open("/Users/cckevin/ScaleCUA/evaluation/AndroidWorld/synthesized_tasks_0123_v2_eval_final_taskid.json", "r"))
random.shuffle(syn_data)
print(len(syn_data))

syn_data_1 = syn_data[:300]
syn_data_2 = syn_data[300:600]
syn_data_3 = syn_data[600:]

json.dump(syn_data_1, open("/Users/cckevin/ScaleCUA/evaluation/AndroidWorld/synthesized_tasks_0123_v2_eval_final_taskid_split1.json", "w"), indent=4)
json.dump(syn_data_2, open("/Users/cckevin/ScaleCUA/evaluation/AndroidWorld/synthesized_tasks_0123_v2_eval_final_taskid_split2.json", "w"), indent=4)
json.dump(syn_data_3, open("/Users/cckevin/ScaleCUA/evaluation/AndroidWorld/synthesized_tasks_0123_v2_eval_final_taskid_split3.json", "w"), indent=4)