// MittiGrid v0.4 — unit tests for lib/llama-command.js (pure module)
// Run: node --test test/

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildLlamaCommand, selectWorkers, sortWorkers, isEligibleWorker, tensorSplitShares } from '../lib/llama-command.js';

const worker = (over = {}) => ({
  agent: 'dev',
  host: '192.168.1.10',
  port: 50052,
  ramGB: 8,
  battery: null,
  standby: false,
  ...over,
});

test('two healthy rpc workers: ordered by ramGB, split proportional to ramGB', () => {
  const workers = [
    worker({ agent: 'phone-8gb', host: '192.168.1.10', ramGB: 8 }),
    worker({ agent: 'laptop-16gb', host: '192.168.1.20', ramGB: 16 }),
  ];
  const picked = selectWorkers(workers);
  assert.deepEqual(picked.map((w) => w.agent), ['laptop-16gb', 'phone-8gb']);
  const cmd = buildLlamaCommand('Qwen2.5-7B-Instruct-Q4_K_M.gguf', workers);
  assert.equal(
    cmd,
    'llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf' +
      ' --rpc 192.168.1.20:50052,192.168.1.10:50052' +
      ' --tensor-split 16,8' +
      ' --host 0.0.0.0 --port 8080 -ngl 99'
  );
});

test('low-battery worker is excluded from selection and the command', () => {
  const workers = [
    worker({ agent: 'phone-8gb', host: '192.168.1.10', ramGB: 8, battery: { level: 22, charging: false, mocked: true, low: true }, standby: true }),
    worker({ agent: 'laptop-16gb', host: '192.168.1.20', ramGB: 16, battery: { level: 90, charging: true, mocked: false, low: false } }),
  ];
  const picked = selectWorkers(workers);
  assert.deepEqual(picked.map((w) => w.agent), ['laptop-16gb']);
  const cmd = buildLlamaCommand('m.gguf', workers);
  assert.match(cmd, /--rpc 192\.168\.1\.20:50052/);
  assert.ok(!cmd.includes('192.168.1.10'), 'low-battery worker host must not appear');
  assert.ok(!cmd.includes('phone-8gb'));
});

test('standby flag alone excludes a worker even with a full battery', () => {
  const workers = [
    worker({ agent: 'a', ramGB: 4, standby: true, battery: { level: 100, charging: false, mocked: false, low: false } }),
    worker({ agent: 'b', ramGB: 2 }),
  ];
  assert.deepEqual(selectWorkers(workers).map((w) => w.agent), ['b']);
});

test('eligibility requires rpc host:port and a usable ramGB', () => {
  assert.equal(isEligibleWorker(worker()), true);
  assert.equal(isEligibleWorker(worker({ host: null })), false);
  assert.equal(isEligibleWorker(worker({ host: '  ' })), false);
  assert.equal(isEligibleWorker(worker({ port: 0 })), false);
  assert.equal(isEligibleWorker(worker({ port: 70000 })), false);
  assert.equal(isEligibleWorker(worker({ port: 50052.5 })), false);
  assert.equal(isEligibleWorker(worker({ ramGB: null })), false);
  assert.equal(isEligibleWorker(worker({ ramGB: 0 })), false);
  assert.equal(isEligibleWorker(null), false);
});

test('no workers -> clean null (the endpoint turns it into 404)', () => {
  assert.deepEqual(selectWorkers([]), []);
  assert.deepEqual(selectWorkers(undefined), []);
  assert.equal(buildLlamaCommand('m.gguf', []), null);
  assert.equal(buildLlamaCommand('m.gguf', [worker({ host: null })]), null);
});

test('missing or empty model -> null', () => {
  const workers = [worker(), worker({ agent: 'b', host: '192.168.1.20', ramGB: 16 })];
  assert.equal(buildLlamaCommand('', workers), null);
  assert.equal(buildLlamaCommand(null, workers), null);
  assert.equal(buildLlamaCommand('   ', workers), null);
});

test('sortWorkers is stable on ties and does not mutate its input', () => {
  const workers = [worker({ agent: 'b', ramGB: 8 }), worker({ agent: 'a', ramGB: 8 })];
  const sorted = sortWorkers(workers);
  assert.deepEqual(sorted.map((w) => w.agent), ['a', 'b']);
  assert.deepEqual(workers.map((w) => w.agent), ['b', 'a']);
});

test('fractional ramGB values survive into --tensor-split', () => {
  const cmd = buildLlamaCommand('m.gguf', [worker({ agent: 'x', host: '10.0.0.2', ramGB: 7.8 })]);
  assert.match(cmd, /--tensor-split 7\.8/);
});

// ---- v0.4: tensorSplitShares (the grid chat's device split) ----------------

test('tensorSplitShares: percentages mirror the command order, one decimal', () => {
  const shares = tensorSplitShares([
    worker({ agent: 'phone-8gb', host: '192.168.1.10', ramGB: 8 }),
    worker({ agent: 'laptop-16gb', host: '192.168.1.20', ramGB: 16 }),
  ]);
  assert.deepEqual(shares, [
    { agent: 'laptop-16gb', ramGB: 16, pct: 66.7 },
    { agent: 'phone-8gb', ramGB: 8, pct: 33.3 },
  ]);
});

test('tensorSplitShares: excludes standby/low-battery workers, same as the command', () => {
  const shares = tensorSplitShares([
    worker({ agent: 'sleepy-phone', ramGB: 8, standby: true }),
    worker({ agent: 'laptop-16gb', ramGB: 16 }),
  ]);
  assert.equal(shares.length, 1);
  assert.equal(shares[0].agent, 'laptop-16gb');
  assert.equal(shares[0].pct, 100);
});

test('tensorSplitShares: empty or all-ineligible -> empty array (never fake numbers)', () => {
  assert.deepEqual(tensorSplitShares([]), []);
  assert.deepEqual(tensorSplitShares(undefined), []);
  assert.deepEqual(tensorSplitShares([worker({ host: null })]), []);
});

test('tensorSplitShares: rounding stays faithful (shares sum to ~100)', () => {
  const shares = tensorSplitShares([
    worker({ agent: 'a', ramGB: 7.8 }),
    worker({ agent: 'b', ramGB: 8 }),
  ]);
  const sum = shares.reduce((s, w) => s + w.pct, 0);
  assert.ok(Math.abs(sum - 100) < 0.2, `expected ~100, got ${sum}`);
  assert.deepEqual(shares, [
    { agent: 'b', ramGB: 8, pct: 50.6 },
    { agent: 'a', ramGB: 7.8, pct: 49.4 },
  ]);
});

test('tensorSplitShares: single worker reports 100%', () => {
  const shares = tensorSplitShares([worker({ agent: 'laptop', ramGB: 16 })]);
  assert.deepEqual(shares, [{ agent: 'laptop', ramGB: 16, pct: 100 }]);
});
