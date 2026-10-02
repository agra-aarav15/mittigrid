// MittiGrid v0.1 — coordinator
// Zero dependencies, node built-ins only. Node >= 20.
//
// Routes:
//   GET  /            -> single-file dark dashboard, auto-refresh every 3s
//   GET  /status.json -> { agents, jobs, stats:{online,queued,done} }
//   POST /join        -> { id, info }                     register / heartbeat agent
//   GET  /poll?id=    -> heartbeat + job assignment       { job } | { job: null }
//   POST /result      -> { id, jobId, ok, result, error, ms }
//   POST /job         -> { type:'primes', start, end, chunks }
//                    or { type:'wordcount', text, chunks }

import http from 'node:http';

const PORT = Number(process.env.PORT) || 7400;
const OFFLINE_MS = 15000; // agent is dimmed/offline when lastSeen is older than this

// id -> { info:{platform,cpus,totalMem}, lastSeen, busy, jobsDone }
const agents = new Map();
// { id, batch, type, payload, status:'queued'|'running'|'done', agent, result, error, ms }
const jobs = [];
let jobSeq = 0;
let batchSeq = 0;

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

function statusPayload() {
  const agentList = [...agents.entries()].map(([id, a]) => ({
    id,
    info: a.info || {},
    busy: a.busy,
    jobsDone: a.jobsDone,
    lastSeenAgo: Math.round((now() - a.lastSeen) / 1000),
    online: isOnline(a),
  }));
  const stats = {
    online: agentList.filter((a) => a.online).length,
    queued: jobs.filter((j) => j.status === 'queued').length,
    done: jobs.filter((j) => j.status === 'done').length,
  };
  return { agents: agentList, jobs: jobs.map(publicJob), stats };
}

function dashboardHTML() {
  const st = statusPayload();

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

  const statusCls = (s) => (s === 'done' ? 'ok' : s === 'running' ? 'run' : s === 'queued' ? 'mut' : 'err');
  const jobRows = st.jobs.map((j) => `      <tr>
        <td>${esc(j.id)}</td>
        <td>${esc(j.batch)}</td>
        <td>${esc(j.type)}</td>
        <td class="${statusCls(j.status)}"${j.error ? ` title="${esc(j.error)}"` : ''}>${esc(j.status)}</td>
        <td>${esc(j.agent || '—')}</td>
        <td class="num">${j.ms != null ? j.ms : '—'}</td>
      </tr>`).join('\n') || '      <tr><td colspan="6" class="dim">no jobs yet — run: node demo.js</td></tr>';

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
  footer { border-top:1px solid rgba(255,255,255,0.12); color:#a1a1a1; font-size:12px; margin-top:32px; padding-top:12px; }
  footer div + div { margin-top:4px; }
</style>
</head>
<body>
  <h1>MittiGrid <span>every device you own, one brain</span></h1>
  <p class="stats">agents online <b>${st.stats.online}</b> &middot; queued <b>${st.stats.queued}</b> &middot; done <b>${st.stats.done}</b></p>
  <h2>agents</h2>
  <table>
    <thead><tr><th>name</th><th>platform</th><th class="num">cpus</th><th class="num">ram gb</th><th>busy</th><th class="num">done</th><th class="num">last seen</th></tr></thead>
    <tbody>
${agentRows}
    </tbody>
  </table>
  <h2>jobs</h2>
  <table>
    <thead><tr><th>id</th><th>batch</th><th>type</th><th>status</th><th>agent</th><th class="num">ms</th></tr></thead>
    <tbody>
${jobRows}
    </tbody>
  </table>
  <footer>
    <div>MittiGrid v0.1 &middot; coordinator :${PORT} &middot; refreshes every 3s</div>
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
      agents.set(body.id, {
        info: body.info || (prev && prev.info) || {},
        lastSeen: now(),
        busy: prev ? prev.busy : false,
        jobsDone: prev ? prev.jobsDone : 0,
      });
      return sendJSON(res, { ok: true });
    }

    if (req.method === 'GET' && path === '/poll') {
      const id = url.searchParams.get('id');
      if (!id) return sendJSON(res, { job: null });
      let a = agents.get(id);
      if (!a) {
        a = { info: {}, lastSeen: now(), busy: false, jobsDone: 0 };
        agents.set(id, a);
      }
      a.lastSeen = now();
      const job = jobs.find((j) => j.status === 'queued');
      if (job) {
        job.status = 'running';
        job.agent = id;
        a.busy = true;
        return sendJSON(res, { job: { id: job.id, type: job.type, payload: job.payload } });
      }
      return sendJSON(res, { job: null });
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

    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('not found');
  } catch (e) {
    if (!res.headersSent) sendJSON(res, { ok: false, error: String((e && e.message) || e) }, 500);
    else res.end();
  }
});

server.listen(PORT, () => {
  console.log(`[mittigrid] coordinator listening on http://localhost:${PORT}`);
  console.log(`[mittigrid] dashboard: http://localhost:${PORT}/  |  status: http://localhost:${PORT}/status.json`);
});
