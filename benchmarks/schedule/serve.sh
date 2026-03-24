#!/bin/bash
set -euo pipefail
source .venv/bin/activate

usage() {
    cat <<'EOF'
Usage: benchmarks/schedule/serve.sh <scenario>

Scenarios:
  cpu               GPU 0, port 8001, CPU backend only
  cpu_disk          GPU 1, port 8002, CPU + Disk, tiering disabled
  cpu_disk_tiering  GPU 2, port 8003, CPU + Disk, tiering enabled
EOF
    exit 1
}

if [[ $# -ne 1 ]]; then
    usage
fi

SCENARIO="$1"
MODEL_PATH="/data/llm-models/Qwen3-8B"
MODEL_NAME="Qwen3-8B"
GPU_MEMORY_UTILIZATION="0.4"
DISK_PATH=""

case "$SCENARIO" in
    cpu)
        GPU=0
        PORT=8001
        ;;
    cpu_disk)
        GPU=1
        PORT=8002
        DISK_PATH="/tmp/lmcache_schedule_disk_cpu_disk/"
        ;;
    cpu_disk_tiering)
        GPU=2
        PORT=8003
        DISK_PATH="/tmp/lmcache_schedule_disk_cpu_disk_tiering/"
        ;;
    *)
        usage
        ;;
esac

if [[ -n "$DISK_PATH" ]]; then
    mkdir -p "$DISK_PATH"
fi

echo "Starting scenario=$SCENARIO gpu=$GPU port=$PORT"
if [[ -n "$DISK_PATH" ]]; then
    echo "Using disk path: $DISK_PATH"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_SERVER_DEV_MODE=1
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=30

if [[ -n "$DISK_PATH" ]]; then
    export LMCACHE_LOCAL_DISK="file://$DISK_PATH"
    export LMCACHE_MAX_LOCAL_DISK_SIZE=120
else
    unset LMCACHE_LOCAL_DISK || true
    unset LMCACHE_MAX_LOCAL_DISK_SIZE || true
fi

if [[ "$SCENARIO" == "cpu_disk_tiering" ]]; then
    export LMCACHE_ENABLE_TIERING=True
else
    unset LMCACHE_ENABLE_TIERING || true
fi

exec vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-prompt-tokens-details \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
