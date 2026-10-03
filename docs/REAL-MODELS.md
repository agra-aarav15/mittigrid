# Real models — the runbook (Gemma 3n on your own hardware)

MittiGrid does not run the model itself. It watches your devices, and hands
you one command that turns them into one shared memory pool for llama.cpp.
This document is the exact path from nothing to a real answer in grid chat,
with honest numbers at every step.

The model this runbook installs: **Gemma 3n E2B** (`gemma-3n-E2B-it-Q4_K_M.gguf`,
about 3 GB). It answers comfortably on a laptop CPU — roughly **5-10
tokens/s** on typical recent x64 laptops. The E4B variant is covered below.

## Step 1 — llama.cpp on the Windows laptop

llama.cpp ships ready-made Windows builds. No compiler, no git.

1. Open <https://github.com/ggml-org/llama.cpp/releases>
2. Download the CPU build for x64 Windows. The asset name looks like:
   `llama-b<build>-bin-win-cpu-x64.zip` (for example
   `llama-b6409-bin-win-cpu-x64.zip`). If you have an NVIDIA GPU there are
   `cuda` assets instead — this grid works with either, CPU is just slower.
3. Unzip it somewhere like `C:\llama\`. You need one file from it:
   `llama-server.exe` (plus the DLLs beside it — keep them together).

Shortcut for THIS repo: a working copy already lives in `tools/`
(`tools/llama-server.exe` + DLLs, build 11374, commit b92761a51). You can skip
the download and use `tools\llama-server.exe` directly.

Whichever binary you use, it must be a recent build — Gemma 3n support landed
in llama.cpp mid-2025, so anything from before that will refuse the model.

## Step 2 — the model file

Download the quantized Gemma 3n E2B (an instruction-tuned model, ~3 GB —
"Q4_K_M" is a 4-bit quantization that keeps quality high while fitting in
laptop RAM):

```
curl -L -o gemma-3n-E2B-it-Q4_K_M.gguf "https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q4_K_M.gguf"
```

Or open that URL in a browser. Put the file next to `llama-server.exe` (or
anywhere you like — the generated command just names it). In this repo it
lives in `models/`.

**E4B variant** (bigger, smarter, roughly half the speed on CPU — about 4.5 GB):

```
curl -L -o gemma-3n-E4B-it-Q4_K_M.gguf "https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q4_K_M.gguf"
```

The model is Google's Gemma 3n under the Gemma license; the GGUF conversions
are by the unsloth community team. Fine to run locally, not for reselling.

## Step 3 — start the coordinator and the workers

On the laptop:

```
node coordinator.js
```

Dashboard: <http://localhost:7400> — grid chat: <http://localhost:7400/chat>

On every Android phone (Termux), one command does the whole join:

```
sh scripts/onboard-phone.sh
```

It asks one question (the coordinator's address), installs Node.js, builds
llama.cpp's `rpc-server` (15-40 minutes once), starts it in the background,
and runs the agent. Full walk-through is inside the script — every step
explains itself.

On the laptop itself (or any other PC), the agent is just:

```
node agent.js
```

Watch the dashboard: every device should appear under **AGENTS**, and every
device running `rpc-server` under **REAL MODEL** with its RAM and a battery
chip. The command in the REAL MODEL section updates itself as devices join
and leave.

## Step 4 — the generated llama-server command

Ask the coordinator for the command (or copy it from the dashboard):

```
curl "http://localhost:7400/llama/command?model=gemma-3n-E2B-it-Q4_K_M.gguf"
```

With a 16 GB laptop and one 8 GB phone joined, the answer is:

```
llama-server -m gemma-3n-E2B-it-Q4_K_M.gguf --rpc 192.168.1.20:50052,192.168.1.10:50052 --tensor-split 16,8 --host 0.0.0.0 --port 8080 -ngl 99
```

How it is built (all from live agent heartbeats, never guessed):

- `--rpc` lists every eligible worker's rpc-server, **biggest RAM first**
  (the laptop leads the pipeline). Low-battery devices are excluded by the
  battery rule — see below.
- `--tensor-split` weights each device's share of the model layers
  proportionally to its RAM. Those same proportions are what grid chat
  reports as the device split.
- `--port 8080` is where llama-server listens. The chat page probes exactly
  this (override the probe with `MITTI_LLAMA_HOST` / `MITTI_LLAMA_PORT` env
  vars on the coordinator if llama-server runs elsewhere).

Run that command on the laptop, wait for llama-server to print that it is
listening, then open <http://localhost:7400/chat>.

## Step 5 — grid chat, with real numbers

Type a prompt. Every answer reports:

- **tokens/sec** — taken ONLY from llama-server's own `timings` field
  (`predicted_per_second`), the measurement llama-server itself made while
  generating. If timings are missing, the coordinator computes
  wall-clock tokens/elapsed and labels it **measured (wall clock)**. It
  never invents a number.
- **output tokens** and the round-trip time.
- **device split** — the share of the tensor-split each device holds
  (laptop 66.7% / 8 GB phone 33.3% in the example above).

## Honest speed table

| Setup | What to actually expect |
|---|---|
| E2B, laptop CPU only (measured: Ryzen-class 6-core) | **5-10 tok/s** — real chat, one word group at a time |
| E2B, laptop + phones (rpc pool) | About the same per-token speed — see the next paragraph |
| E4B, laptop CPU | Roughly half the E2B speed (more parameters per token) |
| E2B/E4B on one phone alone (llama-cli, no rpc) | Single-digit tok/s — usable for patience, not chat |

**Why phones do not make it faster:** llama.cpp's RPC mode is
pipeline-parallel. Each token flows through the layers device-by-device over
your network, one hop at a time. Adding phones adds **memory capacity** —
longer context, or a bigger model that no single device could hold — not raw
speed. The honest reason to join phones is "the model finally fits", not
"more tok/s". Phones also thermal-throttle: fast for two minutes, half speed
by minute ten. Plugged in, cool, screen-off is the good citizen.

The grid chat shows the source of every number for exactly this reason: if
an answer is slower than the table promised, the tok/s on screen is the
truth, not the table.

## The battery rule

A device joins the RPC pool and holds model layers only while its battery is
above 30% or it is charging. Below that (discharging) the agent flips to
**standby**: the coordinator pulls its shards, drops it from generated
commands, and keeps it joined. It reappears in everything automatically once
it is charging again — no restart, no reconfiguration. If a phone benches
itself mid-session, that is the rule working; plug it in.

Termux reports the real battery. For tests on any machine, simulate it:

```
MITTI_FAKE_BATTERY='{"level":22,"charging":false}' node agent.js   # standby
MITTI_FAKE_BATTERY='{"level":80,"charging":true}'  node agent.js   # full member
```

## Safety checklist (read before running rpc-server)

- [ ] **rpc-server has NO authentication.** Anyone who can reach the port can
      use it and read everything that flows through it. Trusted home LAN or
      Tailscale only — never a public/campus/cafe Wi-Fi, never port-forwarded,
      never exposed to the internet.
- [ ] Same rule for the coordinator: it accepts jobs and results from anyone
      who can reach port 7400.
- [ ] Phones plugged in for long sessions — the battery rule will otherwise
      bench them mid-run, by design.
- [ ] Everything stays on your hardware. No API keys, no cloud, no telemetry.

## If something does not work

- **/chat says "no llama-server running"** — it probed `127.0.0.1:8080/health`
  and got nothing. Start the generated command on the laptop; the page prints
  the exact command to run. When llama-server is up, reload the page.
- **"no rpc workers" and no command is generated** — no device is advertising
  rpc-server yet. Run `scripts/onboard-phone.sh` on a phone, or `node agent.js`
  with rpc-server running locally. `GET /llama/status` shows what the
  coordinator can see.
- **llama-server exits with a model error** — usually an old llama.cpp build
  without Gemma 3n support (needs a mid-2025 or newer build), or a truncated
  download (the file must be ~3 GB, re-check the size).
- **First prompt is slow** — that is the model loading into memory. It is
  cached after that.
