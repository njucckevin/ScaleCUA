#!/usr/bin/env bash
set -euo pipefail

# Nightly sequential runs for AndroidWorld Qwen3VL evaluation.
# - Runs are executed strictly one-by-one (the next starts only after the previous finishes).
# - Logs are written into each run's checkpoint_dir.
#
# Usage:
#   cd /Users/chengkanzhi/Desktop/ScaleCUA/evaluation/AndroidWorld
#   bash run_nightly_qwen3vl.sh

cd "$(dirname "$0")"

run_one() {
  local checkpoint_dir="$1"
  shift

  mkdir -p "${checkpoint_dir}"
  local log_file="${checkpoint_dir}/console.log"

  echo "==================================================" | tee -a "${log_file}"
  echo "[START] $(date '+%Y-%m-%d %H:%M:%S')  checkpoint_dir=${checkpoint_dir}" | tee -a "${log_file}"
  echo "[CMD] python run.py $*" | tee -a "${log_file}"
  echo "--------------------------------------------------" | tee -a "${log_file}"

  # Run synchronously; do not start next until this finishes.
  set +e
  python run.py "$@" 2>&1 | tee -a "${log_file}"
  local rc="${PIPESTATUS[0]}"
  set -e

  echo "--------------------------------------------------" | tee -a "${log_file}"
  echo "[END]   $(date '+%Y-%m-%d %H:%M:%S')  exit_code=${rc}" | tee -a "${log_file}"
  echo "==================================================" | tee -a "${log_file}"

  if [[ "${rc}" -ne 0 ]]; then
    echo "[ERROR] run failed (exit_code=${rc}). Stop remaining runs to avoid emulator crash." | tee -a "${log_file}"
    exit "${rc}"
  fi
}

# Sleep between runs to let emulator/services stabilize.
SLEEP_BETWEEN_RUNS_SECONDS=60

# -------------------------
# Run 1
# -------------------------
run_one "runs/qwen25vl_ours_0208_thinking_rl_pattern_50_common_try1" \
  --agent_name qwen25vl \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=true \
  --qwen3vl_model_base_url http://10.210.9.11:32011/v1 \
  --qwen3vl_model_name Qwen2.5-VL-7B-Instruct \
  --qwen3vl_model_api_key EMPTY \
  --checkpoint_dir runs/qwen25vl_ours_0208_thinking_rl_pattern_50_common_try1 \
  --task_random_seed 30

# -------------------------
# Cooldown
# -------------------------
echo "[SLEEP] ${SLEEP_BETWEEN_RUNS_SECONDS}s cooldown before next run..."
sleep "${SLEEP_BETWEEN_RUNS_SECONDS}"

# -------------------------
# Run 2 (starts only after Run 1 ends)
# -------------------------
run_one "runs/qwen25vl_ours_0208_thinking_rl_pattern_50_common_try2" \
  --agent_name qwen25vl \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=true \
  --qwen3vl_model_base_url http://10.210.9.11:32011/v1 \
  --qwen3vl_model_name Qwen2.5-VL-7B-Instruct \
  --qwen3vl_model_api_key EMPTY \
  --checkpoint_dir runs/qwen25vl_ours_0208_thinking_rl_pattern_50_common_try2 \
  --task_random_seed 40

# -------------------------
# Cooldown
# -------------------------
echo "[SLEEP] ${SLEEP_BETWEEN_RUNS_SECONDS}s cooldown before next run..."
sleep "${SLEEP_BETWEEN_RUNS_SECONDS}"

# -------------------------
# Run 3 (starts only after Run 2 ends)
# -------------------------
run_one "runs/qwen25vl_ours_0208_thinking_rl_pattern_50_common_try3" \
  --agent_name qwen25vl \
  --console_port 5554 \
  --grpc_port 8554 \
  --perform_emulator_setup=true \
  --qwen3vl_model_base_url http://10.210.9.11:32011/v1 \
  --qwen3vl_model_name Qwen2.5-VL-7B-Instruct \
  --qwen3vl_model_api_key EMPTY \
  --checkpoint_dir runs/qwen25vl_ours_0208_thinking_rl_pattern_50_common_try3 \
  --task_random_seed 50

echo "[OK] All runs finished successfully."

