// MittiGrid v0.4 — llama.cpp RPC worker selection + command generation
// Zero dependencies, pure module: no I/O, no state, no clock. The coordinator
// feeds it its view of the agents; this file decides who is eligible, orders
// them and renders the llama-server command. Pure so node --test can cover it.
//
// A worker candidate mirrors one agent's heartbeat fields:
//   { agent, host, port, ramGB, battery:{level,charging,mocked,low}|null, standby }
//
// Eligibility: advertises an rpc host:port, has a ramGB to split by, and is
// NOT in low-battery standby. Ordering: biggest ramGB first (the laptop leads
// the pipeline), ties broken by agent id for determinism.

import { batteryLow } from './battery.js';

export function isEligibleWorker(w) {
  if (!w || typeof w !== 'object') return false;
  if (typeof w.agent !== 'string' || !w.agent) return false;
  if (typeof w.host !== 'string' || !w.host.trim()) return false;
  const port = Number(w.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) return false;
  const ram = Number(w.ramGB);
  if (!Number.isFinite(ram) || ram <= 0) return false;
  if (w.standby === true) return false;
  if (batteryLow(w.battery)) return false;
  return true;
}

export function sortWorkers(workers) {
  return [...(workers || [])].sort((a, b) =>
    (Number(b.ramGB) - Number(a.ramGB)) ||
    (a.agent < b.agent ? -1 : a.agent > b.agent ? 1 : 0));
}

// Filter + order in one call — the coordinator's single entry point.
export function selectWorkers(candidates) {
  return sortWorkers((candidates || []).filter(isEligibleWorker));
}

// The one command a laptop needs to pool every eligible device's RAM:
//   llama-server -m <model> --rpc h1:p1,h2:p2 --tensor-split <ramGB weights>
// --tensor-split takes raw weights (llama.cpp normalizes them), so the GB
// numbers themselves are the exact proportional split. Returns null when
// there is no model or no eligible worker (the endpoint turns that into 404).
export function buildLlamaCommand(model, workers) {
  const m = String(model || '').trim();
  const ws = selectWorkers(workers);
  if (!m || !ws.length) return null;
  const rpc = ws.map((w) => `${w.host}:${w.port}`).join(',');
  const split = ws.map((w) => Number(w.ramGB)).join(',');
  return `llama-server -m ${m} --rpc ${rpc} --tensor-split ${split} --host 0.0.0.0 --port 8080 -ngl 99`;
}

// v0.4 (grid chat): each device's share of the generated --tensor-split as a
// percentage, rounded to one decimal, in command order (biggest RAM first).
// A pure mirror of the split buildLlamaCommand emits: same filter, same
// order, same numbers — so the /chat page reports exactly what the command
// generator stored, never a guess. Empty array when nobody is eligible.
export function tensorSplitShares(workers) {
  const ws = selectWorkers(workers);
  if (!ws.length) return [];
  const total = ws.reduce((s, w) => s + Number(w.ramGB), 0);
  return ws.map((w) => ({
    agent: w.agent,
    ramGB: Number(w.ramGB),
    pct: Math.round((Number(w.ramGB) / total) * 1000) / 10,
  }));
}
