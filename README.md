# MittiGrid — every device you own, one brain.

MittiGrid pools the idle compute of every device on your network into one grid. The phone in the drawer, the laptop that never sleeps, the old PC by the TV — together they run jobs no single one of them could.

## The vision

Every home already has a datacenter hiding in plain sight: processors sitting idle most of the day in devices you already own. MittiGrid's goal is simple to say and hard to do: **every device connected to one network — your LAN, or a Tailscale mesh — connects and pools its resources, so small devices together run large jobs.** No cloud bill, no e-waste, no rentals. Your hardware, your network, your data, your rules.

## What v0.1 actually does (honest)

v0.1 is a proof of concept, not the vision. Here is exactly what works today:

- **One-command agent join** from any device on the network: `node agent.js --coord http://<pc-ip>:7400`
- **Live in-browser dashboard** — agents, jobs, and stats, auto-refreshing, plain dark design
- **Distributed CPU jobs proven across 2+ devices** — prime counting and word frequency, split into chunks and executed in parallel by whichever agents show up
- **`node demo.js`** — submits jobs, waits, prints per-agent result tables and totals

What v0.1 does **not** do yet: GPU or LLM work, automatic device discovery, job retry when an agent dies mid-task, authentication, or persistence (coordinator state is in-memory). The agent sandbox is `node:vm` isolation, not a security boundary.

## Architecture

```
 [agent phone]    [agent laptop]    [agent old PC]
       |                |                |
       +---- join / poll / result -------+
                        |
              [ coordinator :7400 ]
                        |
        [ dashboard  http://<pc>:7400/ ]
```

Agents heartbeat and pull work; the coordinator splits jobs into chunks and stitches results. Zero npm dependencies — Node built-ins only.

## Run it

Three terminals (or three devices). Node >= 20, no install step:

```
node coordinator.js          # terminal 1 — dashboard at http://localhost:7400
node agent.js                # terminal 2 — joins as <hostname>:<pid>
node demo.js                 # terminal 3 — submits jobs, prints tables
```

Join from another device on the LAN:

```
node agent.js --coord http://<pc-ip>:7400 --name old-phone-1
```

Options: `agent.js --name <name> --coord <url>` · `demo.js --coord <url> --chunks <n>` · coordinator port: `PORT=7400`.

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
