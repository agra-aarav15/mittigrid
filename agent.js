// MittiGrid v0.4 — agent
// Zero dependencies, node built-ins only. Node >= 20.
//
// Joins a MittiGrid coordinator, heartbeats every 3s, polls for jobs,
// executes them inside a node:vm sandbox (10s timeout), posts results back.
//
// v0.2: the agent also hosts its shard of the toy model in the background.
// A tiny HTTP server (port 7410+ by default, --port to pin) exposes
// POST /shard/run {input, startLayer, endLayer}: it computes its layers,
// then forwards the activations DIRECTLY to the next shard-holder (address
// learned from the coordinator), until the holder of the last layer applies
// the head and returns the final vector. Heartbeats and normal jobs never
// stop while it hosts layers.
//
// v0.3: every heartbeat (the /join call) carries battery + rpc fields:
//   battery — {level, charging, mocked, low} | null (lib/battery.js: env
//             MITTI_FAKE_BATTERY wins when set, else termux-battery-status)
//   standby — true while the battery is low (discharging < 30%): the
//             coordinator keeps us joined but pulls our layers and drops us
//             from the llama RPC pool; we recover automatically on charge
//   ramGB   — os.totalmem()/1e9, one decimal (drives --tensor-split)
//   rpc     — {host, port} of a local llama.cpp rpc-server | null (env
//             MITTI_RPC_PORT pins it; otherwise a one-time 1s probe of
//             127.0.0.1:50052 at boot — if something answers, we advertise)
//
//   node agent.js [--name my-device] [--coord http://<coordinator-ip>:7400] [--port 7410]

import os from 'node:os';
import vm from 'node:vm';
import net from 'node:net';
import http from 'node:http';
import { MODEL_LAYERS, DIM, runLayers, head, validInput } from './model.js';
import { readBattery } from './lib/battery.js';

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  if (i !== -1 && i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
    return process.argv[i + 1];
  }
  return def;
}

function strHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

const NAME = arg('name', `${os.hostname()}:${process.pid}`);
const COORD = String(arg('coord', 'http://localhost:7400')).replace(/\/+$/, '');
const PORT_ARG = parseInt(arg('port', ''), 10);
const SHARD_PORT = Number.isFinite(PORT_ARG) && PORT_ARG > 0
  ? PORT_ARG
  : 7410 + (strHash(NAME) % 100); // deterministic per-device default
const HEARTBEAT_MS = 3000; // idle loop cadence: join + poll
const VM_TIMEOUT_MS = 10000; // hard cap on any single job execution
const HOP_TIMEOUT_MS = 2000; // hard cap on one shard hop (agent -> agent / coordinator)

let myShard = null; // [start, end) — the layers this device hosts in the background
let lastServed = null; // last [start, end) actually served via /shard/run

const info = {
  platform: os.platform(),
  cpus: os.cpus().length,
  totalMem: os.totalmem(),
};

// ---- v0.3: battery + rpc advertisement ------------------------------------

const BATTERY_REFRESH_MS = 10000; // readings are cheap, but not free
const RPC_PROBE_HOST = '127.0.0.1';
const RPC_PROBE_PORT = 50052; // llama.cpp rpc-server default
const RPC_PROBE_TIMEOUT_MS = 1000;

let battery = null; // last known reading {level,charging,mocked,low} | null
let batteryReadAt = 0;

async function refreshBattery() {
  if (Date.now() - batteryReadAt < BATTERY_REFRESH_MS) return;
  batteryReadAt = Date.now();
  try {
    battery = await readBattery();
  } catch {
    battery = null;
  }
}

const standbyNow = () => !!(battery && battery.low);

function lanIp() {
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const i of ifaces[name] || []) {
      if (i.family === 'IPv4' && !i.internal) return i.address;
    }
  }
  return '127.0.0.1';
}

// One-shot TCP connect: is something listening on host:port? Errors (refused,
// timeout, unreachable) all resolve false — probing must never crash the agent.
function probeTcp(host, port, timeoutMs) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      s.destroy();
      resolve(ok);
    };
    const s = net.connect({ host, port });
    s.setTimeout(timeoutMs, () => finish(false));
    s.once('connect', () => finish(true));
    s.on('error', () => finish(false)); // stays attached: absorbs late errors
  });
}

async function post(path, body, timeoutMs = 8000) {
  const res = await fetch(COORD + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  return res.json();
}

async function get(path, timeoutMs = 8000) {
  const res = await fetch(COORD + path, { signal: AbortSignal.timeout(timeoutMs) });
  return res.json();
}

// raw-URL variant for agent-to-agent activation hops
async function postJSON(url, body, timeoutMs) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  return res.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

function fetchErrText(e) {
  const cause = e && e.cause;
  return (cause && (cause.code || cause.message)) || (e && e.message) || 'unknown';
}

// Job execution — inside node:vm so arbitrary payloads stay in a fresh context
// with a hard timeout. NOTE: node:vm is isolation, not a security boundary.
function execute(type, payload) {
  const t0 = Date.now();
  try {
    let result;
    if (type === 'primes') {
      // count primes in [start, end) by trial division
      result = vm.runInNewContext(
        `(function () {
          function isPrime(n) {
            if (n < 2) return false;
            if (n < 4) return true;
            if (n % 2 === 0) return false;
            for (let d = 3; d * d <= n; d += 2) if (n % d === 0) return false;
            return true;
          }
          let count = 0;
          for (let n = Math.max(2, Math.floor(start)); n < end; n++) if (isPrime(n)) count++;
          return { count: count };
        })()`,
        { start: payload.start, end: payload.end },
        { timeout: VM_TIMEOUT_MS }
      );
    } else if (type === 'wordcount') {
      // lowercase alphanumeric tokens; top 8 [word, count] pairs
      result = vm.runInNewContext(
        `(function () {
          const tokens = String(text).toLowerCase().match(/[a-z0-9]+/g) || [];
          const freq = Object.create(null);
          for (const w of tokens) freq[w] = (freq[w] || 0) + 1;
          const top = Object.keys(freq)
            .map((w) => [w, freq[w]])
            .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
            .slice(0, 8);
          return { words: tokens.length, top: top };
        })()`,
        { text: payload.text },
        { timeout: VM_TIMEOUT_MS }
      );
    } else {
      throw new Error('unknown job type: ' + type);
    }
    return { ok: true, result, ms: Date.now() - t0 };
  } catch (e) {
    return { ok: false, error: String((e && e.message) || e), ms: Date.now() - t0 };
  }
}

// ---- v0.2: background shard server ---------------------------------------
// Serves this device's layers of the toy model. The request carries the exact
// [startLayer, endLayer) to compute — the coordinator's shard map is the
// authority — so a merged range can be served immediately after failover,
// before the new assignment has propagated through the next poll. Weights are
// local (every device builds the same deterministic toy model), so the device
// simply computes the range it is asked for and says so when it is wider than
// the range it last saw.
const shardServer = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    if (req.method === 'POST' && url.pathname === '/shard/run') {
      const body = parseJSON(await readBody(req));
      if (!body || !validInput(body.input)) {
        return sendJSON(res, { ok: false, error: `input must be an array of ${DIM} finite numbers` }, 400);
      }
      const startLayer = Math.floor(Number(body.startLayer));
      const endLayer = Math.floor(Number(body.endLayer));
      if (!Number.isInteger(startLayer) || !Number.isInteger(endLayer) ||
          startLayer < 0 || endLayer > MODEL_LAYERS || startLayer >= endLayer) {
        return sendJSON(res, { ok: false, error: `need 0 <= startLayer < endLayer <= ${MODEL_LAYERS}` }, 400);
      }
      if (!lastServed || lastServed[0] !== startLayer || lastServed[1] !== endLayer) {
        lastServed = [startLayer, endLayer];
        if (myShard && (startLayer < myShard[0] || endLayer > myShard[1])) {
          console.log(`shard: serving merged range [${startLayer}, ${endLayer}) (assigned [${myShard[0]}, ${myShard[1]})) — failover in progress`);
        }
      }
      const t0 = performance.now();
      let vec;
      try {
        vec = runLayers(body.input, startLayer, endLayer);
      } catch (e) {
        return sendJSON(res, { ok: false, error: String((e && e.message) || e) }, 500);
      }
      let payload;
      if (endLayer >= MODEL_LAYERS) {
        // this device holds the last layer: apply the head, finish the pass
        const h = head(vec);
        payload = { ok: true, vector: vec, token: h.token, hops: [] };
      } else {
        // learn the next shard-holder from the coordinator, hop activations there
        let next;
        try {
          next = await get(`/model/next?from=${endLayer}`, HOP_TIMEOUT_MS);
        } catch (e) {
          return sendJSON(res, { ok: false, error: `coordinator unreachable for next-hop lookup: ${fetchErrText(e)}`, failedAgent: NAME }, 500);
        }
        if (!next || !next.ok || !next.next) {
          return sendJSON(res, { ok: false, error: (next && next.error) || `no shard holder for layer ${endLayer}`, failedAgent: null }, 500);
        }
        let downstream;
        try {
          downstream = await postJSON(next.next.url + '/shard/run', { input: vec, startLayer: next.next.start, endLayer: next.next.end }, HOP_TIMEOUT_MS);
        } catch (e) {
          return sendJSON(res, { ok: false, error: `next shard ${next.next.agent} unreachable (${fetchErrText(e)})`, failedAgent: next.next.agent }, 500);
        }
        if (!downstream || downstream.ok !== true) {
          return sendJSON(res, { ok: false, error: (downstream && downstream.error) || 'downstream shard failed', failedAgent: (downstream && downstream.failedAgent) || next.next.agent }, 500);
        }
        payload = downstream; // bubble the final vector back up the chain
      }
      // this hop's ms = time spent at this device (compute + downstream wait)
      const ms = Math.round((performance.now() - t0) * 10) / 10;
      payload.hops = [{ agent: NAME, layers: `${startLayer}-${endLayer}`, ms }, ...(payload.hops || [])];
      return sendJSON(res, payload);
    }

    if (req.method === 'GET' && url.pathname === '/shard/status') {
      return sendJSON(res, { ok: true, agent: NAME, coord: COORD, port: shardPort, range: myShard, layers: MODEL_LAYERS });
    }

    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('not found');
  } catch (e) {
    if (!res.headersSent) sendJSON(res, { ok: false, error: String((e && e.message) || e) }, 500);
    else res.end();
  }
});

function listenOnce(server, port) {
  return new Promise((resolve, reject) => {
    const onError = (e) => { cleanup(); reject(e); };
    const onListening = () => { cleanup(); resolve(); };
    const cleanup = () => {
      server.removeListener('error', onError);
      server.removeListener('listening', onListening);
    };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(port);
  });
}

async function startShardServer() {
  let port = SHARD_PORT;
  for (let attempt = 0; attempt < 10; attempt++) {
    port = SHARD_PORT + attempt;
    try {
      await listenOnce(shardServer, port);
      shardServer.on('error', (e) => console.log(`shard server error: ${(e && e.message) || e}`));
      return port;
    } catch {
      if (attempt < 9) console.log(`[mittigrid] shard port ${port} busy, trying ${port + 1}`);
    }
  }
  throw new Error(`no free port for shard server (tried ${SHARD_PORT}..${SHARD_PORT + 9})`);
}

console.log(
  `[mittigrid] agent ${NAME} -> ${COORD} ` +
    `(platform=${info.platform} cpus=${info.cpus} mem=${(info.totalMem / 1073741824).toFixed(1)}GB)`
);

// Main loop: heartbeat every 3s while idle; grab the next job immediately
// after finishing one so busy agents are never throttled by the cadence.
async function main() {
  // v0.3: read the battery before the first heartbeat so the very first join
  // already carries {battery, standby, ramGB, rpc}.
  await refreshBattery();

  // v0.3: advertise a local llama.cpp rpc-server. MITTI_RPC_PORT pins it;
  // otherwise probe the default port once — if something answers, advertise.
  const envRpcPort = parseInt(process.env.MITTI_RPC_PORT, 10);
  const rpcPort = Number.isFinite(envRpcPort) && envRpcPort > 0
    ? envRpcPort
    : (await probeTcp(RPC_PROBE_HOST, RPC_PROBE_PORT, RPC_PROBE_TIMEOUT_MS) ? RPC_PROBE_PORT : null);
  const rpc = rpcPort ? { host: lanIp(), port: rpcPort } : null;
  const ramGB = Math.round((info.totalMem / 1e9) * 10) / 10;

  if (rpc) console.log(`[mittigrid] rpc-server advertised at ${rpc.host}:${rpc.port}`);
  if (battery) {
    console.log(
      `[mittigrid] battery ${battery.level}%${battery.charging ? ' charging' : ''}` +
        `${standbyNow() ? ' — low, joining in standby' : ''}${battery.mocked ? ' (mocked)' : ''}`
    );
  }

  // v0.2: bring up the shard server before the first join, so the very first
  // heartbeat can already advertise a live shard port.
  const shardPort = await startShardServer();
  console.log(`[mittigrid] shard server on :${shardPort} — hosting toy-model layers in the background`);

  let wasStandby = standbyNow();
  for (;;) {
    try {
      await refreshBattery();
      const standby = standbyNow();
      if (standby !== wasStandby) {
        console.log(standby
          ? 'battery low — joining in standby (layers pulled, rpc pool paused until charged)'
          : 'battery recovered — leaving standby');
        wasStandby = standby;
      }
      await post('/join', {
        id: NAME,
        info,
        shardCapable: true,
        shardPort,
        shard: myShard,
        battery, // v0.3: {level,charging,mocked,low} | null
        standby, // v0.3: low battery -> coordinator pulls our layers
        ramGB, // v0.3: drives llama --tensor-split
        rpc, // v0.3: {host,port} of a local rpc-server | null
      });
      const r = await get(`/poll?id=${encodeURIComponent(NAME)}`);
      // v0.2 control channel: the coordinator hands us our layer range here
      if (r && r.control && r.control.type === 'shard') {
        const s = Math.floor(Number(r.control.start));
        const e = Math.floor(Number(r.control.end));
        if (!myShard || myShard[0] !== s || myShard[1] !== e) {
          myShard = [s, e];
          console.log(`shard assigned: layers [${s}, ${e}) of ${MODEL_LAYERS}`);
        }
      }
      const job = r && r.job;
      if (job) {
        const out = execute(job.type, job.payload);
        await post('/result', {
          id: NAME,
          jobId: job.id,
          ok: out.ok,
          result: out.result ?? null,
          error: out.error ?? null,
          ms: out.ms,
        });
        if (out.ok) console.log(`done ${job.type} ${out.ms}ms`);
        else console.log(`failed ${job.type} (${out.ms}ms): ${out.error}`);
        continue;
      }
    } catch (e) {
      console.log(`coordinator unreachable: ${(e && e.message) || e}`);
    }
    await sleep(HEARTBEAT_MS);
  }
}

main().catch((e) => {
  console.error(`agent crashed: ${(e && e.message) || e}`);
  process.exit(1);
});
