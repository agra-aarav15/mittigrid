// MittiGrid v0.3 — coordinator
// Zero dependencies, node built-ins only. Node >= 20.
//
// v0.1 routes (unchanged):
//   GET  /            -> single-file dark dashboard, auto-refresh every 3s
//   GET  /status.json -> { agents, jobs, stats:{online,queued,done}, model }
//   POST /join        -> { id, info, shardCapable, shardPort, shard }
//   GET  /poll?id=    -> { job } | { job: null } (+ control message, see below)
//   POST /result      -> { id, jobId, ok, result, error, ms }
//   POST /job         -> { type:'primes', start, end, chunks }
//                    or { type:'wordcount', text, chunks }
//
// v0.2 model routes:
//   POST /model/infer   { input: number[8] } -> ONE forward pass through the
//                       sharded pipeline: coordinator calls the FIRST
//                       shard-holder; agents hop activations to each other
//                       directly; the last holder returns the final vector.
//                       Returns { ok, vector, token, ms, attempts, trace }.
//   GET  /model/status  -> { layers, ready, shards:[{agent, range}], lastTrace }
//   GET  /model/next?from=N -> which agent holds layer N (agents use this to
//                       forward activations to the next hop)
//
// v0.3 llama.cpp routes (RPC worker pool for real models):
//   GET  /llama/status -> { workers:[{agent,host,port,ramGB,battery}], ready, command }
//   GET  /llama/command?model=<file.gguf> -> { ok, model, workers, command }
//                        (404 {error:'No RPC workers joined yet'} when empty)
//
// Battery awareness: /join heartbeats now carry {battery, standby, ramGB, rpc}.
// A standby (low-battery) agent stays joined but holds no shards and never
// appears in the llama pool — its layers move to survivors through the
// existing failover path, and it recovers automatically once it charges.
//
// Shard assignment travels to agents as a control message on /poll:
// { control: { type:'shard', start, end } }. Agents acknowledge it by sending
// their current range back on the next /join. (POST /shard/assign was the
// alternative; the poll channel needed no new endpoint and reuses heartbeats.)

import http from 'node:http';
import { MODEL_LAYERS, DIM, validInput } from './model.js';
import { batteryLow, normalizeBattery } from './lib/battery.js';
import { selectWorkers, buildLlamaCommand } from './lib/llama-command.js';

const PORT = Number(process.env.PORT) || 7400;
const OFFLINE_MS = 15000; // agent is dimmed/offline when lastSeen is older than this
const HOP_TIMEOUT_MS = 2000; // hard cap on one shard hop (coordinator -> agent)
const SWEEP_MS = 3000; // periodic rebalance: drops agents that went offline

// id -> { info:{platform,cpus,totalMem}, lastSeen, busy, jobsDone,
//         shardCapable, shardUrl, shardAcked,
//         v0.3: battery:{level,charging,mocked,low}|null, standby, ramGB,
//         rpc:{host,port}|null }
const agents = new Map();
// { id, batch, type, payload, status:'queued'|'running'|'done', agent, result, error, ms }
const jobs = [];
let jobSeq = 0;
let batchSeq = 0;

// ---- v0.2: model shard registry -----------------------------------------
// agentId -> [startLayer, endLayer). Contiguous ranges, balanced across
// shard-capable online agents: 2 agents -> 6+6, 3 -> 4+4+4, ...
const shards = new Map();
// agentId -> ts until which it may not hold shards: failover evicts dead
// holders so the periodic sweep cannot hand layers back to a corpse while its
// heartbeat still looks fresh. A live /join clears the eviction (revival).
const evicted = new Map();
let inferBusy = false; // an inference pass is in flight (route must not move)
let lastTrace = null; // per-hop trace of the last completed pass
let lastInferMs = null;
let passCount = 0; // completed inference passes since boot
const lastHopMs = new Map(); // agentId -> ms of its hop in the last pass

const now = () => Date.now();
const isOnline = (a) => now() - a.lastSeen < OFFLINE_MS;

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => {
      data += c;
      if (data.length > 10 * 1024 * 1024) {
        reject(new Error('body too large'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

const parseJSON = (s) => {
  try { return JSON.parse(s); } catch { return null; }
};

function sendJSON(res, obj, code = 200) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function fetchErrText(e) {
  const cause = e && e.cause;
  return (cause && (cause.code || cause.message)) || (e && e.message) || 'unknown';
}

// Split a submitted job into N deterministic subtasks sharing one batch id.
function splitJob(body) {
  const type = body.type;
  const batch = 'b' + (++batchSeq);
  const tasks = [];

  if (type === 'primes') {
    const start = Math.max(1, Math.floor(Number(body.start) || 0));
    const end = Math.floor(Number(body.end) || 0);
    const chunks = Math.min(512, Math.max(1, Math.floor(Number(body.chunks) || 1)));
    const size = Math.ceil((end - start) / chunks);
    for (let i = 0; i < chunks; i++) {
      const s = start + i * size;
      const e = Math.min(end, s + size);
      if (s >= e) break;
      tasks.push({ type, payload: { start: s, end: e } });
    }
  } else if (type === 'wordcount') {
    const text = String(body.text || '');
    const chunks = Math.min(512, Math.max(1, Math.floor(Number(body.chunks) || 1)));
    const words = text.split(/\s+/).filter(Boolean);
    const per = Math.ceil(words.length / chunks);
    for (let i = 0; i < chunks; i++) {
      const slice = words.slice(i * per, (i + 1) * per);
      if (!slice.length) continue;
      tasks.push({ type, payload: { text: slice.join(' ') } });
    }
  } else {
    return null;
  }

  return tasks.length ? { batch, tasks } : null;
}

function publicJob(j) {
  return {
    id: j.id,
    batch: j.batch,
    type: j.type,
    status: j.status,
    agent: j.agent,
    ms: j.ms,
    error: j.error,
    // keep status.json light: don't ship the full text back out
    payload: j.type === 'wordcount' ? { text: `${(j.payload.text || '').length} chars` } : j.payload,
    result: j.result,
  };
}

// ---- v0.2: shard map management ------------------------------------------

// The agent's shard server is reached at the IP it connected from (guaranteed
// routable if it can reach us) plus the port it declared on /join.
function connHost(req) {
  let ra = (req.socket && req.socket.remoteAddress) || '127.0.0.1';
  if (ra.startsWith('::ffff:')) ra = ra.slice(7);
  if (ra === '::1') ra = '127.0.0.1';
  return ra;
}

function shardUrlFor(host, port) {
  const h = host.includes(':') ? `[${host}]` : host;
  return `http://${h}:${port}`;
}

function mapSig(m) {
  return [...m.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([id, r]) => `${id}:${r[0]}-${r[1]}`)
    .join(',');
}

// Recompute balanced contiguous ranges across online shard-capable agents.
// Called on join and on the periodic sweep (which is how a leave is noticed).
// Skipped while an inference pass is in flight — never yank the route
// mid-pipeline; activations are hopping between devices.
function rebalance(reason) {
  if (inferBusy) return;
  for (const [id, until] of evicted) if (until <= now()) evicted.delete(id);
  const ids = [...agents.entries()]
    .filter(([id, a]) => isOnline(a) && a.shardCapable && a.shardUrl && !evicted.has(id) && !a.standby)
    .map(([id]) => id)
    .sort((a, b) => (a < b ? -1 : 1));
  const next = new Map();
  if (ids.length) {
    const per = Math.floor(MODEL_LAYERS / ids.length);
    let extra = MODEL_LAYERS % ids.length; // first `extra` agents get one more layer
    let start = 0;
    for (const id of ids) {
      const size = per + (extra > 0 ? 1 : 0);
      if (extra > 0) extra--;
      next.set(id, [start, start + size]);
      start += size;
    }
  }
  const cur = mapSig(shards);
  const sig = mapSig(next);
  if (cur === sig) return;
  shards.clear();
  for (const [id, r] of next) shards.set(id, r);
  console.log(
    `[shards] ${reason}: ` +
      (ids.length
        ? [...shards.entries()].map(([id, r]) => `${id} ${r[0]}-${r[1]}`).join(', ')
        : 'no shard-capable agents online — layers unassigned')
  );
}

// Fault tolerance: a holder is unreachable, so merge its layers into an
// adjacent survivor (keeps the map contiguous) and let the caller retry once.
function failover(fromId) {
  const range = shards.get(fromId);
  if (!range) return null;
  const ordered = [...shards.entries()].sort((a, b) => a[1][0] - b[1][0]);
  const idx = ordered.findIndex(([id]) => id === fromId);
  let toId = null;
  for (const cand of [ordered[idx - 1], ordered[idx + 1]]) {
    if (cand && cand[0] !== fromId) { toId = cand[0]; break; }
  }
  if (!toId) return null;
  const [s, e] = range;
  const [ts, te] = shards.get(toId);
  shards.set(toId, [Math.min(s, ts), Math.max(e, te)]);
  shards.delete(fromId);
  evicted.set(fromId, now() + OFFLINE_MS + 1000);
  console.log(`[shards] shard ${s}-${e} moved ${fromId} -> ${toId} (unreachable)`);
  return toId;
}

// Ordered pipeline route from the current shard map. ready = the map covers
// layers 0..MODEL_LAYERS contiguously.
function buildRoute() {
  const route = [...shards.entries()]
    .map(([agent, [start, end]]) => {
      const a = agents.get(agent);
      return a && a.shardUrl ? { agent, url: a.shardUrl, start, end } : null;
    })
    .filter(Boolean)
    .sort((x, y) => x.start - y.start);
  let ready = route.length > 0 && route[0].start === 0 && route[route.length - 1].end === MODEL_LAYERS;
  for (let i = 1; i < route.length; i++) if (route[i].start !== route[i - 1].end) ready = false;
  return { route, ready };
}

function modelStatus() {
  const { route, ready } = buildRoute();
  return {
    layers: MODEL_LAYERS,
    ready,
    shards: route.map((r) => ({
      agent: r.agent,
      range: `${r.start}-${r.end}`,
      lastHopMs: lastHopMs.has(r.agent) ? lastHopMs.get(r.agent) : null,
    })),
    lastInferenceMs: lastInferMs,
    lastTrace,
    passCount,
  };
}

// ---- v0.3: llama.cpp RPC pool ---------------------------------------------
// Workers = online agents advertising a local rpc-server, minus standby
// (low-battery) devices. lib/llama-command.js owns eligibility, ordering
// (biggest ramGB first) and the generated command, so it stays unit-testable.

const LLAMA_MODEL_PLACEHOLDER = 'model.gguf'; // dashboard's copyable command

function llamaCandidates() {
  const out = [];
  for (const [id, a] of agents) {
    if (!isOnline(a)) continue;
    out.push({
      agent: id,
      host: a.rpc ? a.rpc.host : null,
      port: a.rpc ? a.rpc.port : null,
      ramGB: a.ramGB,
      battery: a.battery,
      standby: a.standby === true,
    });
  }
  return out;
}

function llamaStatus(model) {
  const workers = selectWorkers(llamaCandidates());
  return {
    workers: workers.map((w) => ({
      agent: w.agent,
      host: w.host,
      port: w.port,
      ramGB: w.ramGB,
      battery: w.battery
        ? { level: w.battery.level, charging: w.battery.charging, low: batteryLow(w.battery) }
        : null,
    })),
    ready: workers.length > 0,
    command: buildLlamaCommand(model, workers),
  };
}

// ---- v0.2: pipelined inference -------------------------------------------

// One inference at a time: simple promise lock (a chained queue).
let inferChain = Promise.resolve();
function runInference(input) {
  const pass = inferChain.then(() => doInference(input));
  inferChain = pass.then(() => undefined, () => undefined);
  return pass;
}

async function callShard(entry, input) {
  try {
    const res = await fetch(entry.url + '/shard/run', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ input, startLayer: entry.start, endLayer: entry.end }),
      signal: AbortSignal.timeout(HOP_TIMEOUT_MS),
    });
    const data = await res.json();
    if (!data || data.ok !== true) {
      const err = new Error((data && data.error) || `shard ${entry.agent} failed`);
      err.failedAgent = (data && data.failedAgent) || entry.agent;
      throw err;
    }
    return data;
  } catch (e) {
    if (e && e.failedAgent != null) throw e; // downstream failure, already attributed
    const err = new Error(`shard holder ${entry.agent} unreachable (${fetchErrText(e)})`);
    err.failedAgent = entry.agent;
    throw err;
  }
}

async function doInference(input) {
  inferBusy = true;
  const wall0 = Date.now();
  try {
    for (let attempt = 1; attempt <= 2; attempt++) {
      const { route, ready } = buildRoute();
      if (!ready) {
        throw new Error(`model not ready: shard map must cover layers 0-${MODEL_LAYERS} contiguously (have ${route.length} shard(s))`);
      }
      try {
        const out = await callShard(route[0], input);
        const ms = Date.now() - wall0;
        lastTrace = out.hops || [];
        lastInferMs = ms;
        passCount++;
        for (const h of lastTrace) if (h && h.agent) lastHopMs.set(h.agent, h.ms);
        console.log(
          `[infer] ok in ${ms}ms (${lastTrace.length} hop(s): ` +
            lastTrace.map((h) => `${h.agent} ${h.layers} ${h.ms}ms`).join(' -> ') + ')'
        );
        return {
          ok: true,
          layers: MODEL_LAYERS,
          vector: out.vector,
          token: out.token ?? null,
          ms,
          attempts: attempt,
          trace: lastTrace,
        };
      } catch (e) {
        const failed = (e && e.failedAgent) || (route[0] && route[0].agent);
        if (attempt === 1 && failed && shards.has(failed)) {
          const to = failover(failed);
          if (to) continue; // retry the pass once with the repaired map
        }
        throw new Error(`inference failed: ${(e && e.message) || e}`);
      }
    }
    throw new Error('inference failed: retries exhausted');
  } finally {
    inferBusy = false;
  }
}

function statusPayload() {
  const agentList = [...agents.entries()].map(([id, a]) => ({
    id,
    info: a.info || {},
    busy: a.busy,
    jobsDone: a.jobsDone,
    lastSeenAgo: Math.round((now() - a.lastSeen) / 1000),
    online: isOnline(a),
    // v0.3 heartbeat fields, for /status.json consumers and the dashboard
    battery: a.battery || null,
    standby: a.standby === true,
    ramGB: a.ramGB ?? null,
    rpc: a.rpc || null,
  }));
  const stats = {
    online: agentList.filter((a) => a.online).length,
    queued: jobs.filter((j) => j.status === 'queued').length,
    done: jobs.filter((j) => j.status === 'done').length,
  };
  return { agents: agentList, jobs: jobs.map(publicJob), stats, model: modelStatus() };
}

function dashboardHTML() {
  const st = statusPayload();
  const model = st.model;
  // copyable command uses a placeholder model name; /llama/command?model=... is authoritative
  const llama = llamaStatus(LLAMA_MODEL_PLACEHOLDER);

  const agentRows = st.agents.map((a) => {
    const mem = a.info && a.info.totalMem ? (a.info.totalMem / 1073741824).toFixed(1) : '?';
    const cpus = a.info && a.info.cpus != null ? a.info.cpus : '?';
    const plat = a.info && a.info.platform ? a.info.platform : '?';
    const seen = a.online ? '<span class="dot"></span>online' : `${a.lastSeenAgo}s ago`;
    return `      <tr${a.online ? '' : ' class="dim"'}>
        <td>${esc(a.id)}</td>
        <td>${esc(plat)}</td>
        <td class="num">${esc(cpus)}</td>
        <td class="num">${esc(mem)}</td>
        <td>${a.busy ? '<span class="run">busy</span>' : '<span class="mut">idle</span>'}</td>
        <td class="num">${a.jobsDone}</td>
        <td class="num">${seen}</td>
      </tr>`;
  }).join('\n') || '      <tr><td colspan="7" class="dim">no agents yet — run: node agent.js</td></tr>';

  const shardRows = model.shards.map((s) => `      <tr>
        <td>${esc(s.agent)}</td>
        <td class="num">${esc(s.range)}</td>
        <td class="num">${s.lastHopMs != null ? s.lastHopMs : '—'}</td>
      </tr>`).join('\n') || '      <tr><td colspan="3" class="dim">no shard-capable agents yet — layers unassigned</td></tr>';

  const statusCls = (s) => (s === 'done' ? 'ok' : s === 'running' ? 'run' : s === 'queued' ? 'mut' : 'err');
  const jobRows = st.jobs.map((j) => `      <tr>
        <td>${esc(j.id)}</td>
        <td>${esc(j.batch)}</td>
        <td>${esc(j.type)}</td>
        <td class="${statusCls(j.status)}"${j.error ? ` title="${esc(j.error)}"` : ''}>${esc(j.status)}</td>
        <td>${esc(j.agent || '—')}</td>
        <td class="num">${j.ms != null ? j.ms : '—'}</td>
      </tr>`).join('\n') || '      <tr><td colspan="6" class="dim">no jobs yet — run: node demo.js</td></tr>';

  // ---- v0.3: REAL MODEL section (llama.cpp RPC pool) ----
  const batteryChip = (b) => b == null
    ? '<span class="mut">no battery data</span>'
    : b.low
      ? `<span class="chip warn" title="standby until charged">low ${esc(b.level)}%</span>`
      : `<span class="chip" title="${b.mocked ? 'mocked reading' : 'live reading'}">${b.charging ? 'charging' : 'ok'} ${esc(b.level)}%</span>`;
  const llamaRows = llama.workers.map((w) => `      <tr>
        <td>${esc(w.agent)}</td>
        <td>${esc(w.host)}:${esc(w.port)}</td>
        <td class="num">${esc(w.ramGB)}</td>
        <td>${batteryChip(w.battery)}</td>
      </tr>`).join('\n') || '      <tr><td colspan="4" class="dim">no rpc workers yet — run rpc-server on a device (docs/REAL-MODELS.md)</td></tr>';
  const standbyAgents = st.agents.filter((a) => a.online && a.standby);
  const standbyNote = standbyAgents.length
    ? ` &middot; <span class="dim">standby: ${standbyAgents.map((a) => esc(a.id)).join(', ')} (low battery — recovers on charge)</span>`
    : '';
  const cmdBlock = llama.command
    ? `  <pre class="cmd">${esc(llama.command)}</pre>
  <p class="stats">copy to the laptop that runs llama-server &middot; replace model.gguf with your quantized model &middot; guide: <b>docs/REAL-MODELS.md</b></p>`
    : '  <p class="stats dim">waiting for rpc workers — see docs/REAL-MODELS.md</p>';

  // ---- v0.3: PULSE MAP state (must survive the meta-refresh cycle) ----
  const pulse = {
    nodes: model.shards.map((s) => ({ agent: s.agent, range: s.range })),
    pass: model.lastTrace ? { ms: model.lastInferenceMs, hops: model.lastTrace } : null,
  };
  const pulseJSON = JSON.stringify(pulse).replace(/</g, '\\u003c');
  const pulseNote = model.lastTrace
    ? `last pass <b>${esc(model.lastInferenceMs)}ms</b> &middot; <b>${model.lastTrace.length}</b> hop(s) &middot; the pulse replays on each refresh`
    : 'no inference yet — run <b>node demo.js --model</b>';

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="3">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MittiGrid</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { background:#0a0a0a; color:#fafafa; margin:32px auto; max-width:900px; padding:0 16px;
         font:14px/1.5 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
  h1 { font-size:22px; font-weight:600; letter-spacing:-0.01em; margin:0; }
  h1 span { color:#a1a1a1; font-weight:normal; font-size:13px; letter-spacing:0; margin-left:8px; }
  .stats { color:#a1a1a1; margin:6px 0 8px; }
  .stats b { color:#fafafa; font-weight:600; }
  h2 { color:#a1a1a1; font-size:11px; font-weight:600; letter-spacing:.08em; margin:26px 0 8px; text-transform:uppercase; }
  table { border-collapse:collapse; width:100%; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid rgba(255,255,255,0.12); white-space:nowrap; }
  th { color:#a1a1a1; font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }
  tbody tr:hover { background:rgba(255,255,255,0.04); }
  td.num, th.num { text-align:right; }
  .dim { opacity:.4; }
  .mut { color:#a1a1a1; }
  .dot { background:#fafafa; border-radius:50%; display:inline-block; height:8px; margin-right:6px; vertical-align:-1px; width:8px; }
  .ok, .run, .err { color:#fafafa; }
  .run, .err { font-weight:600; }
  .chip { border:1px solid rgba(255,255,255,0.3); border-radius:3px; font-size:11px; padding:0 6px; white-space:nowrap; }
  .chip.warn { font-weight:700; }
  .cmd { background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.14); font:12px/1.6 ui-monospace, "Cascadia Mono", Menlo, Consolas, monospace;
         margin:10px 0 4px; overflow-x:auto; padding:10px 12px; white-space:pre-wrap; word-break:break-all; }
  #pulse { background:#0a0a0a; border:1px solid rgba(255,255,255,0.12); display:block; height:auto; max-width:100%; width:100%; }
  footer { border-top:1px solid rgba(255,255,255,0.12); color:#a1a1a1; font-size:12px; margin-top:32px; padding-top:12px; }
  footer div + div { margin-top:4px; }
</style>
</head>
<body>
  <h1>MittiGrid <span>v0.3 &middot; every device you own, one brain</span></h1>
  <p class="stats">agents online <b>${st.stats.online}</b> &middot; queued <b>${st.stats.queued}</b> &middot; done <b>${st.stats.done}</b></p>
  <h2>agents</h2>
  <table>
    <thead><tr><th>name</th><th>platform</th><th class="num">cpus</th><th class="num">ram gb</th><th>busy</th><th class="num">done</th><th class="num">last seen</th></tr></thead>
    <tbody>
${agentRows}
    </tbody>
  </table>
  <h2>MODEL SHARDS</h2>
  <p class="stats"><b>${model.layers}</b> layers &middot; <b>${model.shards.length}</b> shard(s) &middot; ${model.ready ? 'ready' : 'waiting for shard agents'}${model.lastInferenceMs != null ? ` &middot; last pass <b>${model.lastInferenceMs}ms</b>` : ''}${model.passCount ? ` &middot; <b>${model.passCount}</b> pass(es) served` : ''}</p>
  <table>
    <thead><tr><th>agent</th><th class="num">layers</th><th class="num">last hop ms</th></tr></thead>
    <tbody>
${shardRows}
    </tbody>
  </table>
  <p class="stats">run inference: <b>node demo.js --model</b> &middot; or POST /model/infer {"input":[8 numbers]}</p>
  <h2>REAL MODEL</h2>
  <p class="stats"><b>${llama.workers.length}</b> rpc worker(s) &middot; ${llama.ready ? 'ready' : 'waiting for rpc agents'}${standbyNote}</p>
  <table>
    <thead><tr><th>agent</th><th>rpc</th><th class="num">ram gb</th><th>battery</th></tr></thead>
    <tbody>
${llamaRows}
    </tbody>
  </table>
${cmdBlock}
  <h2>PULSE MAP</h2>
  <canvas id="pulse" width="860" height="150"></canvas>
  <p class="stats">${pulseNote}</p>
  <script>
    const MITTI_STATE = ${pulseJSON};
    (() => {
      const cv = document.getElementById('pulse');
      if (!cv) return;
      const ctx = cv.getContext('2d');
      const W = cv.width, H = cv.height, CY = 70, PAD = 70;
      const nodes = (MITTI_STATE && MITTI_STATE.nodes) || [];
      const pass = (MITTI_STATE && MITTI_STATE.pass) || null;
      const index = new Map(nodes.map((n, i) => [n.agent, i]));
      const px = (i) => (nodes.length <= 1 ? W / 2 : PAD + (i * (W - 2 * PAD)) / (nodes.length - 1));
      // hop path as node indexes: skip agents that left the map, collapse repeats
      const hops = [];
      const trace = (pass && pass.hops) || [];
      for (let k = 0; k < trace.length; k++) {
        const i = index.get(trace[k].agent);
        if (i == null) continue;
        if (hops.length && hops[hops.length - 1] === i) continue;
        hops.push(i);
      }
      const SEG_MS = 550; // pulse travel time per hop
      const LEAD_MS = 500; // let the idle glow establish first
      const t0 = performance.now();
      const rgba = (a) => 'rgba(255,255,255,' + a + ')';
      // x position at segment-fraction c (0..hops.length-1)
      const segX = (c) => {
        const last = hops.length - 1;
        const v = Math.max(0, Math.min(c, last));
        const i0 = hops[Math.floor(v)];
        const i1 = hops[Math.min(Math.floor(v) + 1, last)];
        const f = v - Math.floor(v);
        return px(i0) + (px(i1) - px(i0)) * f;
      };
      function draw(nowMs) {
        const t = nowMs - t0;
        ctx.clearRect(0, 0, W, H);
        if (!nodes.length) {
          ctx.fillStyle = '#737373';
          ctx.font = '12px monospace';
          ctx.textAlign = 'center';
          ctx.fillText('no shard agents yet - the map lights up when devices join', W / 2, CY);
          return; // nothing to animate; the next refresh carries new state
        }
        // link line
        ctx.strokeStyle = 'rgba(255,255,255,0.14)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px(0), CY);
        for (let i = 1; i < nodes.length; i++) ctx.lineTo(px(i), CY);
        ctx.stroke();
        // soft idle breathing glow per node
        ctx.textAlign = 'center';
        for (let i = 0; i < nodes.length; i++) {
          const x = px(i);
          const breathe = Math.sin(t / 900 + i * 0.9);
          ctx.beginPath();
          ctx.arc(x, CY, 11 + breathe * 1.5, 0, Math.PI * 2);
          ctx.fillStyle = rgba(0.07 + (0.05 * (breathe + 1)) / 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(x, CY, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(250,250,250,0.9)';
          ctx.fill();
          ctx.font = '10px monospace';
          ctx.fillStyle = '#a1a1a1';
          ctx.fillText(nodes[i].agent, x, i % 2 ? CY + 44 : CY + 28);
          ctx.fillStyle = 'rgba(161,161,161,0.6)';
          ctx.fillText(nodes[i].range, x, CY - 20);
        }
        // bright pulse dot, node-to-node in hop order, with a fading trail
        if (hops.length) {
          const p = (t - LEAD_MS) / SEG_MS;
          if (p >= 0) {
            for (let k = 10; k >= 1; k--) {
              const tx = segX(p - k * 0.07);
              ctx.beginPath();
              ctx.arc(tx, CY, Math.max(1, 4 - k * 0.32), 0, Math.PI * 2);
              ctx.fillStyle = rgba(0.28 * (1 - k / 11));
              ctx.fill();
            }
            ctx.beginPath();
            ctx.arc(segX(p), CY, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            const last = hops.length - 1;
            if (p > last) { // arrival flash on the final node
              const a = Math.max(0, 1 - (p - last) * 0.9);
              ctx.beginPath();
              ctx.arc(px(hops[last]), CY, 10 + (1 - a) * 14, 0, Math.PI * 2);
              ctx.strokeStyle = rgba(a);
              ctx.stroke();
            }
          }
        }
        requestAnimationFrame(draw);
      }
      requestAnimationFrame(draw);
    })();
  </script>
  <h2>jobs</h2>
  <table>
    <thead><tr><th>id</th><th>batch</th><th>type</th><th>status</th><th>agent</th><th class="num">ms</th></tr></thead>
    <tbody>
${jobRows}
    </tbody>
  </table>
  <footer>
    <div>MittiGrid v0.3 &middot; coordinator :${PORT} &middot; refreshes every 3s</div>
    <div>MittiGrid &middot; free &amp; open source &middot; pool every device you own</div>
  </footer>
</body>
</html>`;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const path = url.pathname;
  try {
    if (req.method === 'GET' && (path === '/' || path === '/index.html')) {
      const html = dashboardHTML();
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'content-length': Buffer.byteLength(html) });
      return res.end(html);
    }

    if (req.method === 'GET' && path === '/status.json') {
      return sendJSON(res, statusPayload());
    }

    if (req.method === 'POST' && path === '/join') {
      const body = parseJSON(await readBody(req));
      if (!body || typeof body.id !== 'string' || !body.id) {
        return sendJSON(res, { ok: false, error: 'id required' }, 400);
      }
      const prev = agents.get(body.id);
      if (!prev) {
        console.log(`[join] ${body.id} (${(body.info && body.info.platform) || '?'}, ${(body.info && body.info.cpus) || '?'} cpus)`);
      }
      const shardPort = Math.floor(Number(body.shardPort)) || 0;
      const capable = body.shardCapable === true && shardPort > 0;
      evicted.delete(body.id); // a join is proof of life: clear any failover eviction

      // v0.3 heartbeat fields: battery, standby, ramGB, rpc. `low` is
      // recomputed here — the coordinator never trusts a client-computed flag.
      const batt = body.battery && typeof body.battery === 'object'
        ? normalizeBattery(body.battery, body.battery.mocked === true)
        : null;
      const standby = body.standby === true || (batt != null && batt.low);
      const ramGBn = Number(body.ramGB);
      const ramGB = Number.isFinite(ramGBn) && ramGBn > 0
        ? Math.round(ramGBn * 10) / 10
        : (body.info && body.info.totalMem ? Math.round((body.info.totalMem / 1e9) * 10) / 10 : null);
      const rpcPortN = Math.floor(Number(body.rpc && body.rpc.port));
      const rpcHost = body.rpc && typeof body.rpc.host === 'string' ? body.rpc.host.trim() : '';
      const rpc = rpcHost && Number.isInteger(rpcPortN) && rpcPortN > 0 && rpcPortN < 65536
        ? { host: rpcHost, port: rpcPortN }
        : null;

      const a = {
        info: body.info || (prev && prev.info) || {},
        lastSeen: now(),
        busy: prev ? prev.busy : false,
        jobsDone: prev ? prev.jobsDone : 0,
        // v0.2: this agent hosts model layers in the background
        shardCapable: capable,
        shardUrl: capable ? shardUrlFor(connHost(req), shardPort) : null,
        shardAcked: null,
        // v0.3: battery-aware participation + llama.cpp RPC advertisement
        battery: batt,
        standby,
        ramGB,
        rpc,
      };
      const desired = shards.get(body.id);
      if (capable && desired && Array.isArray(body.shard) &&
          Number(body.shard[0]) === desired[0] && Number(body.shard[1]) === desired[1]) {
        a.shardAcked = [desired[0], desired[1]];
      }
      agents.set(body.id, a);
      // v0.3: a device that just went low-battery gives up its layers right
      // now via the existing failover path (rebalance below keeps it out of
      // future maps). Recovery is automatic: the next charged join clears the
      // eviction above, and the sweep re-shards normally.
      if (standby && !(prev && prev.standby)) {
        if (shards.has(body.id)) failover(body.id);
        else console.log(`[battery] ${body.id} low -> standby (recovers when charging)`);
      }
      if (capable) rebalance('join');
      return sendJSON(res, { ok: true });
    }

    if (req.method === 'GET' && path === '/poll') {
      const id = url.searchParams.get('id');
      if (!id) return sendJSON(res, { job: null });
      let a = agents.get(id);
      if (!a) {
        a = { info: {}, lastSeen: now(), busy: false, jobsDone: 0, shardCapable: false, shardUrl: null, shardAcked: null };
        agents.set(id, a);
      }
      a.lastSeen = now();
      // v0.3: standby agents hold no shards — never hand them layer control
      const desired = a.standby ? null : shards.get(id);
      const control = desired && !(a.shardAcked && a.shardAcked[0] === desired[0] && a.shardAcked[1] === desired[1])
        ? { type: 'shard', start: desired[0], end: desired[1] }
        : null;
      const job = jobs.find((j) => j.status === 'queued');
      if (job) {
        job.status = 'running';
        job.agent = id;
        a.busy = true;
        return sendJSON(res, control ? { job: { id: job.id, type: job.type, payload: job.payload }, control } : { job: { id: job.id, type: job.type, payload: job.payload } });
      }
      return sendJSON(res, control ? { job: null, control } : { job: null });
    }

    if (req.method === 'POST' && path === '/result') {
      const body = parseJSON(await readBody(req));
      if (!body || !body.id || !body.jobId) {
        return sendJSON(res, { ok: false, error: 'id and jobId required' }, 400);
      }
      const job = jobs.find((j) => j.id === body.jobId);
      if (job) {
        job.status = 'done';
        job.agent = body.id;
        job.result = body.ok ? body.result ?? null : null;
        job.error = body.ok ? null : String(body.error || 'unknown error');
        job.ms = Number.isFinite(Number(body.ms)) ? Number(body.ms) : null;
        if (job.error) console.log(`[fail] ${job.id} by ${body.id}: ${job.error}`);
      }
      const a = agents.get(body.id);
      if (a) {
        a.busy = false;
        if (body.ok) a.jobsDone++;
      }
      return sendJSON(res, { ok: true });
    }

    if (req.method === 'POST' && path === '/job') {
      const body = parseJSON(await readBody(req));
      if (!body) return sendJSON(res, { ok: false, error: 'invalid json' }, 400);
      const split = splitJob(body);
      if (!split) return sendJSON(res, { ok: false, error: 'unknown job type or empty work' }, 400);
      for (const t of split.tasks) {
        jobs.push({
          id: 'j' + (++jobSeq),
          batch: split.batch,
          type: t.type,
          payload: t.payload,
          status: 'queued',
          agent: null,
          result: null,
          error: null,
          ms: null,
        });
      }
      console.log(`[job] batch ${split.batch}: ${split.tasks.length} ${body.type} task(s) queued`);
      return sendJSON(res, { ok: true, batch: split.batch, tasks: split.tasks.length });
    }

    // ---- v0.2 model routes ----

    if (req.method === 'POST' && path === '/model/infer') {
      const body = parseJSON(await readBody(req));
      if (!body || !validInput(body.input)) {
        return sendJSON(res, { ok: false, error: `input must be an array of ${DIM} finite numbers` }, 400);
      }
      try {
        const out = await runInference(body.input.map(Number));
        return sendJSON(res, out);
      } catch (e) {
        return sendJSON(res, { ok: false, error: String((e && e.message) || e) }, 503);
      }
    }

    if (req.method === 'GET' && path === '/model/status') {
      return sendJSON(res, modelStatus());
    }

    if (req.method === 'GET' && path === '/model/next') {
      const from = Number(url.searchParams.get('from'));
      if (!Number.isInteger(from) || from < 0) {
        return sendJSON(res, { ok: false, error: 'from (layer index) required' }, 400);
      }
      if (from >= MODEL_LAYERS) return sendJSON(res, { ok: true, next: null });
      const { route, ready } = buildRoute();
      if (!ready) return sendJSON(res, { ok: false, error: 'shards not ready' }, 503);
      const holder = route.find((r) => from >= r.start && from < r.end);
      if (!holder) return sendJSON(res, { ok: false, error: `no shard covers layer ${from}` }, 503);
      return sendJSON(res, { ok: true, next: { agent: holder.agent, url: holder.url, start: holder.start, end: holder.end } });
    }

    // ---- v0.3 llama.cpp routes ----

    if (req.method === 'GET' && path === '/llama/status') {
      const model = url.searchParams.get('model') || LLAMA_MODEL_PLACEHOLDER;
      return sendJSON(res, llamaStatus(model));
    }

    if (req.method === 'GET' && path === '/llama/command') {
      const workers = selectWorkers(llamaCandidates());
      if (!workers.length) {
        return sendJSON(res, { error: 'No RPC workers joined yet' }, 404);
      }
      const model = String(url.searchParams.get('model') || '').trim();
      if (!model) {
        return sendJSON(res, { ok: false, error: 'model query param required, e.g. /llama/command?model=Qwen2.5-7B-Instruct-Q4_K_M.gguf' }, 400);
      }
      console.log(`[llama] command for ${model} across ${workers.length} worker(s)`);
      return sendJSON(res, { ok: true, model, workers: workers.length, command: buildLlamaCommand(model, workers) });
    }

    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('not found');
  } catch (e) {
    if (!res.headersSent) sendJSON(res, { ok: false, error: String((e && e.message) || e) }, 500);
    else res.end();
  }
});

server.listen(PORT, () => {
  console.log(`[mittigrid] v0.3 coordinator listening on http://localhost:${PORT}`);
  console.log(`[mittigrid] dashboard: http://localhost:${PORT}/  |  status: http://localhost:${PORT}/status.json`);
  console.log(`[mittigrid] model: ${MODEL_LAYERS} layers, sharded across agents — POST /model/infer to run a pass`);
  console.log(`[mittigrid] llama: GET /llama/status | GET /llama/command?model=<file.gguf> (needs rpc-server on a device)`);
});

// Periodic rebalance: drops shard entries for agents whose heartbeats went
// stale (a "leave"), redistributes layers, and picks up any rebalance that
// was skipped while an inference pass was in flight.
setInterval(() => rebalance('sweep'), SWEEP_MS);
