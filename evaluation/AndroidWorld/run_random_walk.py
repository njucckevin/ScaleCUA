import os
import time
from datetime import datetime
import sys
import shlex

def run_random_walk(i: int) -> int:
    os.makedirs("logs", exist_ok=True)
    log = f"logs/random_walk_{i:05d}.log"
    # Print to terminal AND save to log file.
    # Use pipefail so the exit status reflects the python process (not tee).
    py = shlex.quote(sys.executable)  # always use the currently running venv python
    log_q = shlex.quote(log)
    cmd = (
        "/bin/bash -c "
        + '"set -o pipefail; '
        + f"{py} random_walk_aw.py --perform_emulator_setup=true "
        + f"2>&1 | tee {log_q}"
        + '"'
    )
    status = os.system(cmd)  # 返回的是 wait status，不一定等于 exit code
    try:
        exit_code = os.waitstatus_to_exitcode(status)
    except Exception:
        exit_code = status
    print(f"[{datetime.now()}] run={i} exit_code={exit_code} log={log}")
    return exit_code

i = 0
while True:
    i += 1
    run_random_walk(i)
    time.sleep(20)