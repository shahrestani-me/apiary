#!/usr/bin/env bash
#
# Apply the Ollama tuning settings from SETUP.md to a server started by the
# Ollama **desktop app**.
#
# Why this script exists: `export OLLAMA_... ` in ~/.zshrc only reaches a server
# you start yourself from that shell. The desktop app is launched by launchd and
# never reads your shell config, so those exports silently do nothing. The
# symptom is a 10-30 s model reload stall on every node of the graph, which
# feels like the swarm is broken when it is only cold.
#
# `launchctl setenv` writes into the per-user launchd session, which GUI apps
# inherit at launch — hence the restart at the end.
#
# NOTE: this does not survive a reboot. Re-run it, or start the server from a
# shell instead (`brew services start ollama` with the vars in its plist, or
# plain `ollama serve` from a shell that has them exported).
#
# Verify afterwards with:  ./scripts/ollama-tuning.sh --check

set -euo pipefail

# Keep the model warm between graph nodes. Load-bearing: without it an 18 GB
# model unloads between calls.
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
# 36 GB fits exactly one big model.
MAX_LOADED="${OLLAMA_MAX_LOADED_MODELS:-1}"
# Careful: KV cache memory scales as NUM_PARALLEL x num_ctx. Don't raise this
# and SWARM_WORKER_CTX at the same time.
NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
# Required for the KV quantization below.
FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
# Halves KV cache memory. Measured perplexity cost is 0.002-0.05, i.e. noise.
KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"

check() {
  local pid
  pid="$(pgrep -f 'Ollama.app/Contents/Resources/ollama' | head -1 || true)"
  if [[ -z "$pid" ]]; then
    pid="$(pgrep -x ollama | head -1 || true)"
  fi
  if [[ -z "$pid" ]]; then
    echo "no ollama server process found — start Ollama first"
    return 1
  fi
  echo "ollama server pid $pid environment:"
  # `ps eww` prints the process environment; anything missing here is NOT in
  # effect, no matter what your ~/.zshrc says.
  ps eww "$pid" | tr ' ' '\n' | grep -E '^OLLAMA_' || echo "  (no OLLAMA_* vars set)"
}

if [[ "${1:-}" == "--check" ]]; then
  check
  exit 0
fi

echo "setting launchd environment..."
launchctl setenv OLLAMA_KEEP_ALIVE "$KEEP_ALIVE"
launchctl setenv OLLAMA_MAX_LOADED_MODELS "$MAX_LOADED"
launchctl setenv OLLAMA_NUM_PARALLEL "$NUM_PARALLEL"
launchctl setenv OLLAMA_FLASH_ATTENTION "$FLASH_ATTENTION"
launchctl setenv OLLAMA_KV_CACHE_TYPE "$KV_CACHE_TYPE"

echo "restarting Ollama so it picks them up..."
osascript -e 'quit app "Ollama"' 2>/dev/null || true
sleep 3
open -a Ollama
sleep 5

echo
check
