// MittiGrid v0.1 — demo driver
// Zero dependencies. Submits two distributed jobs to the coordinator, waits for
// every chunk to finish, and prints per-agent result tables.
//
//   node demo.js [--coord http://<coordinator-ip>:7400] [--chunks 8]

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  if (i !== -1 && i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
    return process.argv[i + 1];
  }
  return def;
}

const COORD = String(arg('coord', 'http://localhost:7400')).replace(/\/+$/, '');
const CHUNKS = Math.max(1, parseInt(arg('chunks', '8'), 10) || 8);
const TIMEOUT_MS = 60000; // per stage

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

async function main() {
  console.log(`[mittigrid] demo -> ${COORD}`);

  const primes = await runBatch({ type: 'primes', start: 1, end: 1600000, chunks: CHUNKS });
  printPrimesReport(primes);

  const wordcount = await runBatch({ type: 'wordcount', text: TEXT, chunks: 4 });
  printWordcountReport(wordcount);

  console.log('');
  console.log('demo complete: one network, many devices, one brain.');
}

main().catch((e) => {
  console.error('demo failed: ' + ((e && e.message) || e));
  process.exit(1);
});
