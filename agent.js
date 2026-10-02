// MittiGrid v0.1 — agent
// Zero dependencies, node built-ins only. Node >= 20.
//
// Joins a MittiGrid coordinator, heartbeats every 3s, polls for jobs,
// executes them inside a node:vm sandbox (10s timeout), posts results back.
//
//   node agent.js [--name my-device] [--coord http://<coordinator-ip>:7400]

import os from 'node:os';
import vm from 'node:vm';

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  if (i !== -1 && i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
    return process.argv[i + 1];
  }
  return def;
}

const NAME = arg('name', `${os.hostname()}:${process.pid}`);
const COORD = String(arg('coord', 'http://localhost:7400')).replace(/\/+$/, '');
const HEARTBEAT_MS = 3000; // idle loop cadence: join + poll
const VM_TIMEOUT_MS = 10000; // hard cap on any single job execution

const info = {
  platform: os.platform(),
  cpus: os.cpus().length,
  totalMem: os.totalmem(),
};

async function post(path, body) {
  const res = await fetch(COORD + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(8000),
  });
  return res.json();
}

async function get(path) {
  const res = await fetch(COORD + path, { signal: AbortSignal.timeout(8000) });
  return res.json();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

console.log(
  `[mittigrid] agent ${NAME} -> ${COORD} ` +
    `(platform=${info.platform} cpus=${info.cpus} mem=${(info.totalMem / 1073741824).toFixed(1)}GB)`
);

// Main loop: heartbeat every 3s while idle; grab the next job immediately
// after finishing one so busy agents are never throttled by the cadence.
for (;;) {
  try {
    await post('/join', { id: NAME, info });
    const r = await get(`/poll?id=${encodeURIComponent(NAME)}`);
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
