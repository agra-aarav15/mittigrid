# Real models (llama.cpp RPC) — the honest guide

MittiGrid v0.3 hands your laptop one command that turns the phones and PCs on
your LAN into one shared memory pool for llama.cpp. This document says exactly
what that buys you, what it does not, and how to set it up.

## What this gives you

- **Pooled RAM across devices.** A 16 GB laptop plus an 8 GB phone is roughly
  24 GB of memory for model weights and KV cache — quantized models that fit
  on no single device fit across the pool.
- **No cloud, no GPU rental, no API key.** llama.cpp runs locally on hardware
  you own; MittiGrid only discovers the workers and generates the command.
- **Battery-aware participation.** Phones at low battery drop out of the pool
  automatically and rejoin when charged. Nothing to configure.

## What this does NOT do (read this part)

- **Pipeline-parallel is capacity, not speed.** Layers are split across
  devices; each token flows through them one hop at a time over your network.
  You fit bigger models — you do not get faster tokens.
- **Expect roughly 1-10 tokens/s on Wi-Fi**, depending on model size, hop
  count, and radios. This is for "it runs at all", not for snappy chat.
- **rpc-server has NO authentication.** Anyone who can reach the port can use
  it and read what flows through it. Trusted LAN or Tailscale only — never
  expose it to the internet, never run it on public Wi-Fi.
- **Phones thermal-throttle.** A phone that is fast for two minutes can halve
  its speed at minute ten. Plugged in, cool, and screen-off is the good citizen.
- **The toy 12-layer model and the RPC pool are separate systems today.**
  MittiGrid does not load GGUF weights itself; it orchestrates `llama-server`
  via `--rpc`. The toy model proves the pipeline, the RPC pool runs real weights.

## Phone side (Termux, once)

1. Install Termux (the F-Droid build), then inside Termux: `pkg update`
2. Copy the repo (or just `scripts/build-rpc-termux.sh`) to the phone and run:
   ```
   sh scripts/build-rpc-termux.sh
   ```
   It installs clang/cmake/git, shallow-clones llama.cpp, and builds with
   `GGML_RPC=ON`. Budget 15-40 minutes on a mid-range phone.
3. Start the worker:
   ```
   ./build/bin/rpc-server --host 0.0.0.0 --port 50052
   ```
4. In a second Termux session, join the grid:
   ```
   node agent.js --coord http://<coordinator-ip>:7400
   ```
   The agent probes `127.0.0.1:50052` once at boot (or set `MITTI_RPC_PORT=50052`)
   and advertises the phone's LAN IP and RAM to the coordinator automatically.

## Laptop side (Windows, or any OS)

1. Get llama.cpp: download a Windows release build from the llama.cpp GitHub
   releases (CUDA build if you have an NVIDIA GPU, otherwise the CPU/AVX2
   build) — or build it yourself with `-DGGML_RPC=ON` so it can talk to the phones.
2. Open the dashboard `http://<coordinator-ip>:7400/` and copy the command in
   the REAL MODEL section, or fetch it:
   ```
   curl "http://<coordinator-ip>:7400/llama/command?model=Qwen2.5-7B-Instruct-Q4_K_M.gguf"
   ```
3. Run it. For a 16 GB laptop + 8 GB phone it looks like this:
   ```
   llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --rpc 192.168.1.20:50052,192.168.1.10:50052 --tensor-split 16,8 --host 0.0.0.0 --port 8080 -ngl 99
   ```
   `--rpc` lists workers biggest-RAM-first; `--tensor-split` weights each
   worker's share of the layers proportionally to its RAM.
4. Open `http://localhost:8080` — llama-server's own web UI — and chat.

## The battery rule

A device joins the RPC pool and holds model layers only while its battery is
above 30% or it is charging. Below that (discharging) the agent flips to
**standby**: the coordinator pulls its shards, drops it from generated
commands, and keeps it joined. It reappears in everything automatically once
it is charging again — no restart, no reconfiguration.

Termux reports the real battery. For tests on any machine, simulate it:

```
MITTI_FAKE_BATTERY='{"level":22,"charging":false}' node agent.js   # standby
MITTI_FAKE_BATTERY='{"level":80,"charging":true}'  node agent.js   # full member
```

## Safety checklist

- [ ] rpc-server only on a network you control (home LAN or Tailscale)
- [ ] coordinator reachable only from your devices (same rule)
- [ ] phones plugged in for long sessions — the battery rule will otherwise
      bench them mid-run, by design
