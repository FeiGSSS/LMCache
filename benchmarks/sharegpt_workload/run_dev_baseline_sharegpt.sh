#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/fei/research/llm/KVCache/LMCache-Schedule}
BENCH_SCRIPT=${BENCH_SCRIPT:-$PROJECT_DIR/benchmarks/sharegpt_workload/sharegpt_multiturn_ttft_bench.py}
SCENARIO_FILE=${SCENARIO_FILE:-$PROJECT_DIR/benchmarks/sharegpt_workload/scenarios_multiturn_ttft.yaml}

MODEL_PATH=${MODEL_PATH:-/home/fei/research/models/Qwen3-8B}
SHAREGPT_PATH=${SHAREGPT_PATH:-/home/fei/research/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${API_KEY:-EMPTY}
REQUEST_TIMEOUT_SEC=${REQUEST_TIMEOUT_SEC:-120}
PROGRESS_INTERVAL_SEC=${PROGRESS_INTERVAL_SEC:-10}

SEEDS=${SEEDS:-"41 42 43"}
SCENARIOS=${SCENARIOS:-"functional_serial target_10_inflight_600s scan_arrival_low scan_arrival_mid scan_arrival_high scan_inflight_3 scan_inflight_5 scan_inflight_10 scan_inflight_20"}

OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_DIR/outputs/sharegpt_dev_baseline_$(date +%Y%m%d_%H%M%S)}

if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "Benchmark script not found: $BENCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SCENARIO_FILE" ]]; then
  echo "Scenario file not found: $SCENARIO_FILE" >&2
  exit 1
fi
if [[ ! -f "$SHAREGPT_PATH" ]]; then
  echo "ShareGPT file not found: $SHAREGPT_PATH" >&2
  exit 1
fi

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/.uvenv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.uvenv/bin/python"
else
  echo "No virtual env found under $PROJECT_DIR (.venv or .uvenv)." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [m for m in ["openai", "yaml"] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing dependencies: {missing}. Install with 'uv pip install openai pyyaml'.")
print("Dependency check passed: openai, pyyaml")
PY

mkdir -p "$OUTPUT_ROOT"

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "project_dir=$PROJECT_DIR"
  echo "model_path=$MODEL_PATH"
  echo "sharegpt_path=$SHAREGPT_PATH"
  echo "base_url=$BASE_URL"
  echo "progress_interval_sec=$PROGRESS_INTERVAL_SEC"
  echo "scenarios=$SCENARIOS"
  echo "seeds=$SEEDS"
  if command -v git >/dev/null 2>&1; then
    echo "branch=$(git -C "$PROJECT_DIR" branch --show-current || true)"
    echo "head=$(git -C "$PROJECT_DIR" rev-parse --short HEAD || true)"
    echo "origin_dev=$(git -C "$PROJECT_DIR" rev-parse --short origin/dev || true)"
    echo "upstream_dev=$(git -C "$PROJECT_DIR" rev-parse --short upstream/dev || true)"
    echo "ahead_behind_upstream=$(git -C "$PROJECT_DIR" rev-list --left-right --count upstream/dev...HEAD || true)"
  fi
} > "$OUTPUT_ROOT/run_meta.txt"

echo "Output root: $OUTPUT_ROOT"

for scenario in $SCENARIOS; do
  for seed in $SEEDS; do
    out_csv="$OUTPUT_ROOT/${scenario}_seed${seed}_requests.csv"
    out_json="$OUTPUT_ROOT/${scenario}_seed${seed}_summary.json"
    out_progress_json="$OUTPUT_ROOT/${scenario}_seed${seed}_progress_live.json"

    echo "[RUN] scenario=$scenario seed=$seed"
    "$PYTHON_BIN" "$BENCH_SCRIPT" \
      --sharegpt-path "$SHAREGPT_PATH" \
      --base-url "$BASE_URL" \
      --model "$MODEL_PATH" \
      --api-key "$API_KEY" \
      --scenario-file "$SCENARIO_FILE" \
      --scenario-name "$scenario" \
      --seed "$seed" \
      --request-timeout-sec "$REQUEST_TIMEOUT_SEC" \
      --progress-interval-sec "$PROGRESS_INTERVAL_SEC" \
      --progress-summary-json "$out_progress_json" \
      --output-csv "$out_csv" \
      --summary-json "$out_json"
  done
done

echo "All baseline runs completed."
echo "Summary files: $OUTPUT_ROOT"
