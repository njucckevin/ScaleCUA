# import os
# import json


# data_test = [
#     {"base_task_name": "AudioRecorderRecordAudio", "instruction": "Create a new contact person with the name cckevin and the phone number 18851134288.", "sample_id": "0"},
#     {"base_task_name": "ExpenseDeleteMultiple", "instruction": "Delete the Ride-Sharing expenses from pro expense app", "sample_id": "1"},
# ]
# json.dump(data_test, open("data_test.json", "w"), indent=4)


from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _last_step_has_enter_button(response: str) -> bool:
    # Common patterns observed in trajectory "response" field.
    needles = [
        '"button": "Enter"',
        '"button":"Enter"',
        '"button": "enter"',
        '"button":"enter"',
        '"button": "Menu"',
        '"button":"Menu"',
        '"button": "menu"',
        '"button":"menu"',
    ]
    return any(n in response for n in needles)


def main() -> None:
    base = Path(
        "/Users/chengkanzhi/Desktop/ScaleCUA/evaluation/AndroidWorld/runs/gemini3pro_last3"
    )
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    hit_tasks: list[str] = []
    total_tasks = 0
    total_with_result = 0

    for task_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        total_tasks += 1
        # Be tolerant to naming: some users may write "results.json" by mistake.
        result_path = task_dir / "result.json"
        if not result_path.exists():
            continue

        data = _read_json(result_path)
        if not data:
            continue
        total_with_result += 1

        traj = data.get("trajectory", [])
        if not isinstance(traj, list) or not traj:
            continue
        last = traj[-1]
        if not isinstance(last, dict):
            continue
        resp = last.get("response", "")
        if not isinstance(resp, str):
            resp = str(resp)

        if _last_step_has_enter_button(resp):
            hit_tasks.append(task_dir.name)

    print(f"Base: {base}")
    print(f"Task dirs total: {total_tasks}")
    print(f"Task dirs with result.json: {total_with_result}")
    print(f'Last step contains \\"button\\": \\"Enter\\" count: {len(hit_tasks)}')
    if hit_tasks:
        print("Hit tasks:")
        for name in hit_tasks:
            print(f"- {name}")


if __name__ == "__main__":
    main()