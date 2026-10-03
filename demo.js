// MittiGrid v0.2 — demo driver
// Zero dependencies. Two modes:
//
//   node demo.js          — v0.1: submits distributed primes + wordcount jobs
//                           to a running grid, waits, prints per-agent tables
//   node demo.js --model  — v0.2: spawns 2 agents itself, waits for the toy
//                           model's 12 layers to be sharded, runs one forward
//                           pass printing the per-hop trace, KILLS one agent,
//                           runs again proving shard failover, cleans up
//
//   node demo.js [--coord http://<coordinator-ip>:7400] [--chunks 8]

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  if (i !== -1 && i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
    return process.argv[i + 1];
  }
  return def;
}

function hasFlag(name) {
  return process.argv.includes('--' + name);
}

const COORD = String(arg('coord', 'http://localhost:7400')).replace(/\/+$/, '');
const CHUNKS = Math.max(1, parseInt(arg('chunks', '8'), 10) || 8);
const TIMEOUT_MS = 60000; // per stage

const AGENT_SCRIPT = fileURLToPath(new URL('./agent.js', import.meta.url));
const MODEL_INPUT = [0.12, -0.5, 0.9, 0.33, -0.77, 0.05, 0.6, -0.21];
const READY_TIMEOUT_MS = 45000;
const STABILITY_WAIT_MS = 17000; // > coordinator OFFLINE_MS: let stale agents age out
const FAULT_WAIT_MS = 3000; // let the grid notice the killed agent

// ~300-word text about free software and pocket clouds, with deliberately
// repeated phrases so the word-frequency counts are interesting.
const TEXT = `Free software means the users have the freedom to run, copy, distribute, study, change, and improve the software. A pocket cloud is the same idea pointed at hardware instead of code: every device you own already carries a processor that sits idle most of the day. The old phone in the drawer, the laptop that never sleeps, the mini PC beside the television, each one is a small brain waiting for work. MittiGrid pools those idle processors into one network, so small devices run big jobs together.

The free software movement taught a generation that source code wants to be shared, studied, and improved in the open. The pocket cloud teaches the same lesson about compute. Proprietary clouds rent you back the power of machines you already own. A pocket cloud refuses that trade: your devices, your network, your data, your rules. Free software gave us the right to run our own code; a pocket cloud gives us the right to run our own compute.

Watch this demo and notice what is happening. One coordinator counts the work. Many agents count the primes. No single device counted a million primes alone; the mesh counted them together, the way free software is written together. Word counts behave the same way: a long paragraph about free software and the pocket cloud arrives as one text and leaves as shared arithmetic.

That is the whole promise. Every device you own joins one network and pools its resources, so small devices run big jobs together. Free software made publishing cheap. The pocket cloud makes computing cheap. The old phone in the drawer is not e-waste yet; it is an agent waiting to join. The laptop that never sleeps is not just a laptop; it is a node. Every device you own is a member, and the grid is patient.`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function post(path, body) {
  const res = await fetch(COORD + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(8000),
  });
  return res.json();
}

async function getStatus() {
  const res = await fetch(COORD + '/status.json', { signal: AbortSignal.timeout(8000) });
  return res.json();
}

async function getModelStatus() {
  const res = await fetch(COORD + '/model/status', { signal: AbortSignal.timeout(8000) });
  return res.json();
}

// Submit a job, then poll status.json every 1s until every task in the batch
// is done (or timeout). Throws on submission failure, task failure, or timeout.
async function runBatch(submitBody) {
  const t0 = Date.now();
  const submit = await post('/job', submitBody);
  if (!submit.ok) throw new Error('coordinator rejected job: ' + (submit.error || 'unknown'));
  const batch = submit.batch;
  const deadline = t0 + TIMEOUT_MS;

  for (;;) {
    const st = await getStatus();
    const mine = st.jobs.filter((j) => j.batch === batch);
    if (mine.length && mine.every((j) => j.status === 'done')) {
      const failed = mine.find((j) => j.error);
      if (failed) throw new Error(`task ${failed.id} failed on ${failed.agent}: ${failed.error}`);
      return { batch, jobs: mine, wall: Date.now() - t0, tasks: submit.tasks };
    }
    if (Date.now() > deadline) {
      throw new Error(`timeout after ${TIMEOUT_MS / 1000}s waiting for batch ${batch}`);
    }
    await sleep(1000);
  }
}

const pad = (s, n) => String(s).padEnd(n);
const padl = (s, n) => String(s).padStart(n);

// ---- v0.2: model demo -----------------------------------------------------

function printShardMap(st, title) {
  console.log(`\n${title} — ${st.layers} layers, ${st.ready ? 'ready' : 'NOT ready'}`);
  console.log('  ' + pad('agent', 20) + pad('layers', 10));
  for (const s of st.shards) console.log('  ' + pad(s.agent, 20) + pad(s.range, 10));
  if (!st.shards.length) console.log('  (no shard-capable agents)');
}

function printInference(label, r) {
  console.log(`\n${label}: ${r.layers} layers, ${r.trace.length} hop(s), total ${r.ms}ms, token ${r.token}`);
  console.log('  ' + pad('agent', 20) + pad('layers', 10) + padl('ms', 8));
  for (const h of r.trace) console.log('  ' + pad(h.agent, 20) + pad(h.layers, 10) + padl(h.ms, 8));
  console.log('  final vector: [' + r.vector.map((x) => Number(x).toFixed(4)).join(', ') + ']');
}

// Wait until the model is ready and every agent we spawned holds a shard.
// If extra shard-holders are present (e.g. a just-killed agent from a previous
// run that the coordinator still counts as online), give them OFFLINE_MS to
// age out before proceeding with whatever else is genuinely on the grid.
async function waitShardsReady(names) {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  let stableSince = null;
  for (;;) {
    const st = await getModelStatus();
    const present = st.shards.map((s) => s.agent);
    if (st.ready && names.every((n) => present.includes(n))) {
      if (present.length === names.length) return st;
      if (stableSince == null) stableSince = Date.now();
      if (Date.now() - stableSince > STABILITY_WAIT_MS) return st;
    }
    if (Date.now() > deadline) {
      throw new Error(`shards not ready after ${READY_TIMEOUT_MS / 1000}s (shard agents: ${present.join(', ') || 'none'})`);
    }
    await sleep(400);
  }
}

async function runModelDemo() {
  const children = [];
  try {
    console.log(`[mittigrid] model demo -> ${COORD}`);

    // (0) spawn 2 agents as child processes, distinct shard ports, names
    // unique to this run so a previous run's ghosts can't be confused with us
    const run = String(process.pid);
    const specs = [
      { name: `demo-shard-a-${run}`, port: 7410 },
      { name: `demo-shard-b-${run}`, port: 7411 },
    ];
    for (const s of specs) {
      const c = spawn(process.execPath, [AGENT_SCRIPT, '--name', s.name, '--port', String(s.port), '--coord', COORD], { stdio: 'inherit' });
      children.push(c);
      console.log(`[spawn] ${s.name} (shard port ${s.port}, pid ${c.pid})`);
    }

    // (1) wait until the 12 layers are sharded across both
    const st0 = await waitShardsReady(specs.map((s) => s.name));
    printShardMap(st0, 'shard map (balanced across 2 agents)');

    // (2) one forward pass: activations hop agent-to-agent
    const r1 = await post('/model/infer', { input: MODEL_INPUT });
    if (!r1.ok) throw new Error('inference failed: ' + (r1.error || 'unknown'));
    printInference('pass 1 — both agents alive', r1);

    // (3) kill one agent; its layers must move to the survivor
    const victimName = specs[0].name;
    console.log(`\n[fault] killing ${victimName} (pid ${children[0].pid}) — its layers must be reassigned`);
    children[0].kill();
    await sleep(FAULT_WAIT_MS);

    // (4) second pass: coordinator failover happens inside /model/infer
    const r2 = await post('/model/infer', { input: MODEL_INPUT });
    if (!r2.ok) throw new Error('inference after failover failed: ' + (r2.error || 'unknown'));
    printInference(
      `pass 2 — ${victimName} dead` + (r2.attempts > 1 ? ' (shard reassigned, pass retried once)' : ' (no retry needed)'),
      r2
    );
    if (r2.attempts < 2) {
      console.log('  note: failover was not exercised this pass — the grid found another route');
    }

    const st2 = await getModelStatus();
    printShardMap(st2, 'shard map after failover');

    // (5) determinism: the sharded math must survive the topology change
    const same = JSON.stringify(r1.vector) === JSON.stringify(r2.vector) && r1.token === r2.token;
    console.log('');
    console.log(`determinism: final vector identical across failover: ${same ? 'yes' : 'NO'}`);
    if (!same) throw new Error('final vector changed after failover — pipeline is not deterministic');

    console.log('');
    console.log('model demo complete: layers sharded, activations hopped device-to-device, dead shard reassigned.');
  } finally {
    for (const c of children) {
      try { c.kill(); } catch { /* already gone */ }
    }
  }
}

async function main() {
  if (hasFlag('model')) {
    await runModelDemo();
    return;
  }

  console.log(`[mittigrid] demo -> ${COORD}`);

  const primes = await runBatch({ type: 'primes', start: 1, end: 1600000, chunks: CHUNKS });
  printPrimesReport(primes);

  const wordcount = await runBatch({ type: 'wordcount', text: TEXT, chunks: 4 });
  printWordcountReport(wordcount);

  console.log('');
  console.log('demo complete: one network, many devices, one brain.');
}

function printPrimesReport(r) {
  const jobsList = r.jobs.slice().sort((a, b) => a.payload.start - b.payload.start);
  const wAgent = Math.max(6, ...jobsList.map((j) => (j.agent || '?').length));
  console.log('');
  console.log(`stage A: primes 1..1,600,000 split into ${jobsList.length} chunks (batch ${r.batch}, wall ${(r.wall / 1000).toFixed(1)}s)`);
  console.log('');
  console.log('  ' + pad('agent', wAgent) + '  ' + pad('range', 21) + padl('primes', 8) + padl('ms', 7));
  let total = 0;
  for (const j of jobsList) {
    total += j.result.count;
    console.log(
      '  ' + pad(j.agent || '?', wAgent) + '  ' +
      pad(`${j.payload.start} - ${j.payload.end}`, 21) +
      padl(j.result.count, 8) + padl(j.ms, 7)
    );
  }
  const used = [...new Set(jobsList.map((j) => j.agent))];
  console.log('');
  console.log(`TOTAL primes below 1,600,000: ${total}  (${used.length} agent(s): ${used.join(', ')})`);
}

function printWordcountReport(r) {
  const jobsList = r.jobs.slice().sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
  const wAgent = Math.max(6, ...jobsList.map((j) => (j.agent || '?').length));
  console.log('');
  console.log(`stage B: wordcount, ${TEXT.split(/\s+/).filter(Boolean).length} words split into ${jobsList.length} chunks (batch ${r.batch}, wall ${(r.wall / 1000).toFixed(1)}s)`);
  console.log('');
  console.log('  ' + pad('agent', wAgent) + '  ' + pad('chunk', 6) + padl('words', 7) + padl('ms', 7));
  const totals = new Map();
  let wordsTotal = 0;
  for (const j of jobsList) {
    wordsTotal += j.result.words;
    console.log('  ' + pad(j.agent || '?', wAgent) + '  ' + pad(j.id, 6) + padl(j.result.words, 7) + padl(j.ms, 7));
    for (const [w, c] of j.result.top) totals.set(w, (totals.get(w) || 0) + c);
  }
  const top = [...totals.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, 10);
  const used = [...new Set(jobsList.map((j) => j.agent))];
  console.log('');
  console.log(`words counted: ${wordsTotal} across ${used.length} agent(s): ${used.join(', ')}`);
  console.log('top words: ' + top.map(([w, c]) => `${w} (${c})`).join(', '));
}

main().catch((e) => {
  console.error('demo failed: ' + ((e && e.message) || e));
  process.exit(1);
});
