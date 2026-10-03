# MittiGrid — every device you own, one brain.

MittiGrid pools the idle compute of every device on your network into one grid. The phone in the drawer, the laptop that never sleeps, the old PC by the TV — together they run jobs no single one of them could.

## The vision

Every home already has a datacenter hiding in plain sight: processors sitting idle most of the day in devices you already own. MittiGrid's goal is simple to say and hard to do: **every device connected to one network — your LAN, or a Tailscale mesh — connects and pools its resources, so small devices together run large jobs.** No cloud bill, no e-waste, no rentals. Your hardware, your network, your data, your rules.

## What v0.1 actually does (honest)

v0.1 is a proof of concept, not the vision. Here is exactly what works:

- **One-command agent join** from any device on the network: `node agent.js --coord http://<pc-ip>:7400`
- **Live in-browser dashboard** — agents, jobs, and stats, auto-refreshing, plain dark design
- **Distributed CPU jobs proven across 2+ devices** — prime counting and word frequency, split into chunks and executed in parallel by whichever agents show up
- **`node demo.js`** — submits jobs, waits, prints per-agent result tables and totals

What v0.1 does **not** do: model inference, automatic device discovery, authentication, or persistence (coordinator state is in-memory). The agent sandbox is `node:vm` isolation, not a security boundary.

## What v0.2 adds (honest)

v0.2 keeps every v0.1 feature and adds the first real piece of the vision: **a model's layers, divided across your devices.**

- **Layer-sharded toy model** — a fixed 12-layer model (seeded deterministic weights, 8-dim activations, tanh per layer, 8-to-4 head, argmax token) is split into contiguous layer ranges balanced across shard-capable agents: 2 agents host 6+6 layers, 3 agents host 4+4+4. Rebalanced on every join and leave.
- **Activations hop device-to-device** — `POST /model/infer` starts the pass at the first shard-holder; each agent computes its layers and forwards the activations over HTTP **directly to the next agent** (next hop learned from the coordinator), until the holder of the last layer applies the head and returns the final vector. The coordinator records the total ms and a per-hop trace.
- **Each device keeps its normal work** — an agent hosting layers still heartbeats and still executes primes/wordcount jobs. The shard runs in the background on a tiny per-agent HTTP server (port 7410+ by default, `--port` to pin it).
- **Fault tolerance proven** — if a shard-holder is unreachable (connection error or 2s timeout), the coordinator reassigns its layers to a surviving agent in memory (`shard <range> moved <from> -> <to>`) and retries the pass once. `node demo.js --model` does this end to end: spawns 2 agents, runs a pass, kills one agent, runs again, and prints the new shard map — the final vector comes out identical, because the sharded math is deterministic.
- **Honest label** — the toy model is **untrained**. It proves shard assignment, the activation pipeline, and failover — not the quality. Real 700B-class models arrive via llama.cpp RPC / exo-style sharding (see Roadmap).

What v0.2 does **not** do yet: real model weights or GPU work, parallel inference (one forward pass at a time), authentication, or persistence (coordinator state is in-memory). The shard endpoint and the `node:vm` sandbox are isolation, not a security boundary.

## What v0.3 adds (honest)

v0.3 keeps every v0.1 + v0.2 feature and points the grid at real models: **llama.cpp RPC worker discovery, battery-aware participation, and a live pulse map.**

- **Battery-aware participation** — every heartbeat carries `{battery, standby, ramGB, rpc}`. A device reading below 30% while discharging flips to **standby**: the coordinator pulls its model layers (they move to survivors through the existing failover path) and drops it from the llama RPC pool, but keeps it joined — and it recovers automatically once it charges. On Termux the agent reads the real battery; anywhere else, simulate one: `MITTI_FAKE_BATTERY='{"level":22,"charging":false}'`.
- **RPC advertisement** — an agent that finds a local llama.cpp `rpc-server` (env `MITTI_RPC_PORT`, else a one-time 1s TCP probe of `127.0.0.1:50052` at boot) advertises `{rpc:{host,port}}` plus its RAM (`os.totalmem`, 1 decimal) to the coordinator.
- **One command for pooled RAM** — `GET /llama/command?model=<file.gguf>` returns a ready-to-run `llama-server` command using only live, charged workers (`--rpc` biggest-RAM-first, `--tensor-split` proportional to each worker's RAM). `GET /llama/status` shows the pool; 404 with a clean error when nobody has joined. The dashboard's REAL MODEL section shows ready state, workers with battery chips, and a copyable command block.
- **PULSE MAP** — the dashboard draws shard-holding devices as labeled nodes on a monochrome canvas. On every inference pass a bright pulse dot travels node-to-node in hop order with a fading trail; idle nodes breathe softly. The last pass's hops are serialized into the dashboard state, so the map survives the 3s meta-refresh cycle.
- **Unit tests** — `npm test` covers llama worker selection (ordering, tensor split, low-battery exclusion) and the battery rules (fake JSON variants, garbage → null).

What v0.3 does **not** do: load GGUF weights itself (it orchestrates `llama-server` via `--rpc`), make tokens faster (RPC pooling is capacity, not speed), authentication, or persistence.

## Real models (llama.cpp RPC)

With a llama.cpp `rpc-server` running on each device (build guide for Termux and laptops: [docs/REAL-MODELS.md](docs/REAL-MODELS.md)), the coordinator turns your devices into one shared memory pool for llama.cpp:

```
curl "http://<coordinator-ip>:7400/llama/command?model=Qwen2.5-7B-Instruct-Q4_K_M.gguf"
# -> llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --rpc 192.168.1.20:50052,192.168.1.10:50052 --tensor-split 16,8 --host 0.0.0.0 --port 8080 -ngl 99
```

Honest limits (full list in the guide):

- **Capacity, not speed** — you fit bigger models across devices; tokens still hop device-to-device, so expect roughly 1-10 tok/s on Wi-Fi.
- **rpc-server has NO authentication** — trusted LAN or Tailscale only. Never expose it to the internet.
- **Phones thermal-throttle** — plugged in, cool, and screen-off is the good citizen.

**Battery rule:** a device holds model shards and joins the RPC pool only while above 30% battery or charging. Below that it goes standby — kept joined, layers moved to survivors, excluded from generated commands — and reappears automatically once it charges.

## Architecture

```
 [agent phone]    [agent laptop]    [agent old PC]
       |                |                |
       +---- join / poll / result -------+
                        |
              [ coordinator :7400 ]
                        |
        [ dashboard  http://<pc>:7400/ ]

 v0.2 — one forward pass through the sharded toy model:

 [coordinator] --input--> [agent A  layers 0-6] --activations--> [agent B  layers 6-12]
      ^                                                                        |
      +----------------------------- final vector -----------------------------+
```

Agents heartbeat and pull work; the coordinator splits jobs into chunks and stitches results. Shard assignments ride the same heartbeat channel as control messages on poll (`{type:'shard', start, end}`) — no new endpoints on the agent side, no extra config. Zero npm dependencies — Node built-ins only.

## Run it

Four terminals (or four devices). Node >= 20, no install step:

```
node coordinator.js          # terminal 1 — dashboard at http://localhost:7400
node agent.js                # terminal 2 — joins as <hostname>:<pid>, hosts layers
node demo.js                 # terminal 3 — distributed primes + wordcount tables
node demo.js --model         # terminal 4 — self-contained sharded-inference + failover proof
```

Join from another device on the LAN:

```
node agent.js --coord http://<pc-ip>:7400 --name old-phone-1
```

Options: `agent.js --name <name> --coord <url> --port <shard-port>` · `demo.js --coord <url> --chunks <n>` · `demo.js --model` · coordinator port: `PORT=7400`.

Model endpoints: `POST /model/infer {"input":[8 numbers]}` runs one forward pass · `GET /model/status` shows the shard map and last trace · the dashboard carries a MODEL SHARDS table (agent, layers, last hop ms). Llama endpoints: `GET /llama/status` · `GET /llama/command?model=<file.gguf>` (RPC worker pool, see Real models below).

## Tailscale tip

Install [Tailscale](https://tailscale.com) on every device, then point agents at the coordinator's Tailscale IP:

```
node agent.js --coord http://100.x.y.z:7400
```

The mesh works across the internet and stays private — no ports forwarded, no cloud in the middle.

## Roadmap

- [ ] Tailscale auto-discovery — agents find the coordinator with zero flags
- [ ] LLM layer-sharding across agents — honest note: [exo](https://github.com/exo-explore/exo) pioneered this, and we respect it; MittiGrid aims to be the dead-simple, India-first version
- [ ] WebGPU devices — phones and laptops contributing their GPUs
- [ ] Sandbox hardening — beyond `node:vm`, real isolation per job
- [ ] Job retry when an agent disappears mid-task

## License

AGPL-3.0 — see [LICENSE](LICENSE).
