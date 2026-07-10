#!/usr/bin/env bash
set -euo pipefail

PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-https://llm.agaii.org/llm/v1}"

probe_json() {
  local name="$1" url="$2"
  local output
  if output="$(curl -fsS --max-time 10 "$url" 2>&1)"; then
    printf '%s\tOK\t%s\n' "$name" "$output"
  else
    printf '%s\tFAIL\t%s\n' "$name" "$output"
    return 1
  fi
}

failed=0
probe_json "e4b_local" "http://127.0.0.1:8888/v1/models" || failed=1
probe_json "e2b_local" "http://127.0.0.1:8889/v1/models" || failed=1
probe_json "router" "http://127.0.0.1:8890/healthz" || failed=1
probe_json "public" "$PUBLIC_BASE_URL/models" || failed=1

printf '\nGPU\n'
nvidia-smi \
  --query-gpu=index,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw \
  --format=csv,noheader

printf '\nPROCESSES\n'
ps -eo pid,ppid,etime,stat,args |
  grep -E '(gemma-4-E[24]B|ops.model_router:app|tunnel_tang_2557|cap-voice)' |
  grep -v grep || true

printf '\nLISTENERS\n'
ss -ltnp | grep -E ':(8888|8889|8890|24536) ' || true

exit "$failed"
