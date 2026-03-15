#!/usr/bin/env bash
set -euo pipefail

# ====== Paths (container paths; should map to your Windows D:\design1\GANDCTAnalysis\...) ======
TRAIN_TF="/dct/database/gandct/for_prepare_2k_dct_log_scaled_normalized_train_tf/data.tfrecords"
VAL_TF="/dct/database/gandct/for_prepare_2k_dct_log_scaled_normalized_val_tf/data.tfrecords"
TEST_TF="/dct/database/gandct/for_prepare_2k_color_dct_log_scaled_normalized_test_tf/data.tfrecords"

# Output summary file (this maps to: D:\design1\GANDCTAnalysis\final_models\ouput.txt)
OUT_TXT="/dct/final_models/ouput.txt"

LOG_DIR="/dct/exp_logs"
mkdir -p "$LOG_DIR"
mkdir -p "/dct/final_models"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Parse "Saving model with accuracy ... to <PATH>/" from train log
extract_model_dir() {
  local train_log="$1"
  # Example line:
  # Saving model with accuracy - 91.27% - to ./final_models/log2_2026-.../
  local line
  line="$(grep -E "Saving model .* to " "$train_log" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return
  fi
  # Extract the part after " to "
  local dir
  dir="$(echo "$line" | sed -E 's/^.* to //')"
  # Make it absolute if it's relative like ./final_models/...
  if [[ "$dir" == ./* ]]; then
    dir="/dct/${dir#./}"
  fi
  echo "$dir"
}

# Parse the last Keras-style summary line from test output:
# 4687/4687 ... - 179s 38ms/step - loss: 0.6998 - acc: 0.9066
extract_test_metrics() {
  local test_log="$1"
  local line
  line="$(grep -E "[0-9]+/[0-9]+.*loss:.*acc:" "$test_log" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return
  fi
  echo "$line"
}

# Compute throughput images/s from test line + batch size
# Uses: steps_total (second number in "a/b"), seconds (e.g., 179s), batch
compute_ips() {
  local test_line="$1"
  local batch="$2"

  # 提取 steps_total：把 "4687/4687" 的后半截取出来
  local steps_token steps_total seconds
  steps_token="$(echo "$test_line" | awk '{print $1}')"
  steps_total="$(echo "$steps_token" | cut -d'/' -f2 | tr -d '\r\n')"

  # 提取 seconds：匹配 "- 173s"
  seconds="$(echo "$test_line" | sed -nE 's/^.*- ([0-9]+)s .*$/\1/p' | tr -d '\r\n')"

  if [[ -z "$steps_total" || -z "$seconds" || "$seconds" == "0" ]]; then
    echo "NA"
    return
  fi

  python -c "print(f'{(int($steps_total)*int($batch))/float($seconds):.2f}')"
}

append_summary() {
  local model="$1"
  local batch="$2"
  local epochs="$3"
  local model_dir="$4"
  local test_line="$5"
  local ips="$6"

  # Extract loss & acc from test_line
  local loss acc
  loss="$(echo "$test_line" | sed -nE 's/^.*loss: ([0-9.]+).*$/\1/p')"
  acc="$(echo "$test_line" | sed -nE 's/^.*acc: ([0-9.]+).*$/\1/p')"

  {
    echo "[$(timestamp)] model=${model} batch=${batch} epochs=${epochs}"
    echo "  model_dir=${model_dir}"
    echo "  test_line=${test_line}"
    echo "  test_loss=${loss} test_acc=${acc} throughput_images_per_s=${ips}"
    echo ""
  } >> "$OUT_TXT"
}

run_one() {
  local model="$1"
  local batch="$2"
  local epochs="$3"
  shift 3
  local extra_args=("$@")

  local train_log="${LOG_DIR}/train_${model}_b${batch}_e${epochs}.log"
  local test_log="${LOG_DIR}/test_${model}_b${batch}_e${epochs}.log"

  echo "=== [$(timestamp)] TRAIN $model (b=$batch, e=$epochs) ==="
  python /dct/classifier.py train "$model" \
    "$TRAIN_TF" "$VAL_TF" \
    --classes 2 -b "$batch" -e "$epochs" "${extra_args[@]}" \
    | tee "$train_log"

  local model_dir
  model_dir="$(extract_model_dir "$train_log")"
  if [[ -z "${model_dir}" ]]; then
    echo "ERROR: Could not detect saved model directory from train log: $train_log"
    echo "       Check the log for 'Saving model ... to ...'"
    exit 1
  fi

  echo "=== [$(timestamp)] TEST $model (b=$batch) ==="
  python /dct/classifier.py test \
    "$model_dir" "$TEST_TF" \
    -b "$batch" \
    | tee "$test_log"

  local test_line
  test_line="$(extract_test_metrics "$test_log")"
  if [[ -z "${test_line}" ]]; then
    echo "ERROR: Could not parse test metrics line from: $test_log"
    exit 1
  fi

  local ips
  ips="$(compute_ips "$test_line" "$batch")"

  append_summary "$model" "$batch" "$epochs" "$model_dir" "$test_line" "$ips"

  echo "=== [$(timestamp)] DONE $model | IPS=$ips images/s ==="
  echo
}

# ====== Header ======
{
  echo "============================================================"
  echo "GANDCTAnalysis auto-run summary (log -> cnn -> resnet)"
  echo "Output file: $OUT_TXT"
  echo "Started at: $(timestamp)"
  echo "============================================================"
  echo
} >> "$OUT_TXT"

# ====== Run models ======
run_one "log"   32 10
run_one "cnn"   32 15 --early_stopping 5
run_one "resnet" 16 10 --early_stopping 3 --l2 0.02

echo "All done. Summary appended to: $OUT_TXT"