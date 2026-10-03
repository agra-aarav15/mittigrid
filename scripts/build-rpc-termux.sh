#!/bin/sh
# MittiGrid v0.4 — build llama.cpp with the RPC backend on Termux (Android).
# Run inside Termux, from any directory you want llama.cpp/ and build/ to live:
#   sh build-rpc-termux.sh
# When it finishes you have ./build/bin/rpc-server — start it and join the grid.

set -e

echo "== [1/5] install build tools (clang, cmake, git) =="
pkg install -y clang cmake git

echo "== [2/5] fetch llama.cpp source (shallow clone) =="
if [ -d llama.cpp ]; then
  echo "    llama.cpp/ already exists here - reusing it"
else
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

echo "== [3/5] configure (RPC on, native CPU optimizations on) =="
cmake -S llama.cpp -B build -DGGML_RPC=ON -DGGML_NATIVE=ON

echo "== [4/5] build (release, $(nproc) parallel jobs) =="
cmake --build build --config Release -j"$(nproc)"

echo "== [5/5] done =="
echo ""
echo "Start the RPC worker (leave it running in this session):"
echo "  ./build/bin/rpc-server --host 0.0.0.0 --port 50052"
echo ""
echo "Then join the grid from a second Termux session:"
echo "  node agent.js --coord http://<coordinator-ip>:7400"
echo ""
echo "The agent probes 127.0.0.1:50052 at boot (or set MITTI_RPC_PORT=50052)"
echo "and advertises this phone to the coordinator automatically."
