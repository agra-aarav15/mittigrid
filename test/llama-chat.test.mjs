// MittiGrid v0.4 — unit tests for lib/llama-chat.js (pure module)
// Covers: llama-server response parsing (timings first, labeled wall-clock
// fallback, never an invented number), chat message validation, and the
// /api/llama/chat-config state shape.
// Run: node --test test/

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chatResultFromUpstream, validChatMessages, chatConfig, DEFAULT_CHAT_MODEL } from '../lib/llama-chat.js';

// ---- chatResultFromUpstream -------------------------------------------------

const upstreamWithTimings = {
  choices: [{ message: { role: 'assistant', content: 'Hello from the grid.' } }],
  usage: { prompt_tokens: 9, completion_tokens: 42 },
  timings: { prompt_n: 9, prompt_ms: 120.5, predicted_n: 42, predicted_ms: 1789.2, predicted_per_second: 23.4721 },
};

test('timings present: tokensPerSecond comes ONLY from predicted_per_second', () => {
  const r = chatResultFromUpstream(upstreamWithTimings, 99999);
  assert.equal(r.ok, true);
  assert.equal(r.content, 'Hello from the grid.');
  assert.equal(r.tokens, 42);
  assert.equal(r.tokensPerSecond, 23.47);
  assert.equal(r.tokensPerSecondSource, 'llama-server timings');
});

test('timings present but predicted_n missing: tokens fall back to usage', () => {
  const r = chatResultFromUpstream({
    choices: [{ message: { content: 'hi' } }],
    usage: { completion_tokens: 7 },
    timings: { predicted_per_second: 5.5 },
  }, 1000);
  assert.equal(r.ok, true);
  assert.equal(r.tokens, 7);
  assert.equal(r.tokensPerSecond, 5.5);
});

test('timings missing: wall-clock tokens/elapsed, labeled "measured"', () => {
  const body = {
    choices: [{ message: { content: 'measured answer' } }],
    usage: { completion_tokens: 24 },
  };
  const r = chatResultFromUpstream(body, 3000); // 24 tokens in 3s
  assert.equal(r.ok, true);
  assert.equal(r.tokens, 24);
  assert.equal(r.tokensPerSecond, 8);
  assert.match(r.tokensPerSecondSource, /^measured/);
  assert.match(r.tokensPerSecondSource, /wall clock/);
});

test('no timings and no token count: tokensPerSecond null, source says why', () => {
  const r = chatResultFromUpstream({ choices: [{ message: { content: 'no numbers here' } }] }, 1500);
  assert.equal(r.ok, true);
  assert.equal(r.content, 'no numbers here');
  assert.equal(r.tokens, null);
  assert.equal(r.tokensPerSecond, null);
  assert.match(r.tokensPerSecondSource, /unavailable/);
});

test('timings present but useless (zero / NaN / not an object) -> measured fallback, not zero', () => {
  for (const timings of [{ predicted_per_second: 0 }, { predicted_per_second: 'fast' }, 'oops']) {
    const r = chatResultFromUpstream({
      choices: [{ message: { content: 'x' } }],
      usage: { completion_tokens: 10 },
      timings,
    }, 2000);
    assert.equal(r.ok, true);
    assert.equal(r.tokensPerSecond, 5);
    assert.match(r.tokensPerSecondSource, /^measured/);
  }
});

test('upstream error shapes surface as ok:false with the model server message', () => {
  const obj = chatResultFromUpstream({ error: { message: 'model failed to load' } }, 100);
  assert.equal(obj.ok, false);
  assert.match(obj.error, /model failed to load/);
  const str = chatResultFromUpstream({ error: 'slot unavailable' }, 100);
  assert.equal(str.ok, false);
  assert.match(str.error, /slot unavailable/);
});

test('garbage upstream bodies -> ok:false, never a throw', () => {
  assert.equal(chatResultFromUpstream(null, 10).ok, false);
  assert.equal(chatResultFromUpstream('nope', 10).ok, false);
  assert.equal(chatResultFromUpstream([], 10).ok, false);
  assert.equal(chatResultFromUpstream({}, 10).ok, false, 'no choices');
  assert.equal(chatResultFromUpstream({ choices: [] }, 10).ok, false, 'empty choices');
  assert.equal(chatResultFromUpstream({ choices: [{ message: {} }] }, 10).ok, false, 'no content');
  assert.equal(chatResultFromUpstream(42, 10).ok, false);
});

test('negative or zero wall clock cannot produce a measured rate', () => {
  const body = { choices: [{ message: { content: 'x' } }], usage: { completion_tokens: 10 } };
  const r = chatResultFromUpstream(body, 0);
  assert.equal(r.ok, true);
  assert.equal(r.tokensPerSecond, null);
});

// ---- validChatMessages ------------------------------------------------------

test('validChatMessages: clean input is normalized to role+content only', () => {
  const out = validChatMessages([
    { role: 'system', content: 'You are MittiGrid.', extra: 'stripped' },
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'hi' },
  ]);
  assert.deepEqual(out, [
    { role: 'system', content: 'You are MittiGrid.' },
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'hi' },
  ]);
});

test('validChatMessages: rejects wrong shapes (never throws)', () => {
  assert.equal(validChatMessages(null), null);
  assert.equal(validChatMessages(undefined), null);
  assert.equal(validChatMessages('hi'), null);
  assert.equal(validChatMessages([]), null, 'must have at least one message');
  assert.equal(validChatMessages([{ role: 'user' }]), null, 'no content');
  assert.equal(validChatMessages([{ role: 'user', content: '   ' }]), null, 'blank content');
  assert.equal(validChatMessages([{ role: 'tool', content: 'x' }]), null, 'unknown role');
  assert.equal(validChatMessages([{ content: 'no role' }]), null);
  assert.equal(validChatMessages([null]), null);
  assert.equal(validChatMessages([{ role: 'user', content: 5 }]), null, 'non-string content');
  assert.equal(validChatMessages([{ role: 'user', content: 'x'.repeat(100001) }]), null, 'over the size cap');
  const tooMany = Array.from({ length: 201 }, () => ({ role: 'user', content: 'x' }));
  assert.equal(validChatMessages(tooMany), null, 'over the count cap');
});

// ---- chatConfig -------------------------------------------------------------

test('chatConfig: llama-server up -> llamaUrl points at the stored port, model from stored command', () => {
  const cfg = chatConfig({
    lastCommand: { model: 'gemma-3n-E2B-it-Q4_K_M.gguf', port: 8080, command: 'llama-server -m ...', workers: [], at: 1 },
    llamaUp: true,
    workers: 2,
    command: 'llama-server -m gemma-3n-E2B-it-Q4_K_M.gguf --rpc 10.0.0.2:50052',
  });
  assert.equal(cfg.ok, true);
  assert.equal(cfg.ready, true);
  assert.equal(cfg.llamaUrl, 'http://127.0.0.1:8080/v1');
  assert.equal(cfg.model, 'gemma-3n-E2B-it-Q4_K_M.gguf');
  assert.equal(cfg.workers, 2);
  assert.match(cfg.command, /--rpc 10\.0\.0\.2:50052/);
});

test('chatConfig: llama-server down -> llamaUrl EMPTY (never a guessed URL), ready false', () => {
  const cfg = chatConfig({
    lastCommand: { model: 'm.gguf', port: 8080, command: 'llama-server ...', workers: [], at: 1 },
    llamaUp: false,
    workers: 2,
    command: 'llama-server ...',
  });
  assert.equal(cfg.ready, false);
  assert.equal(cfg.llamaUrl, '');
  assert.equal(cfg.port, 8080);
});

test('chatConfig: no stored command yet -> documented default model and port 8080', () => {
  const cfg = chatConfig({ lastCommand: null, llamaUp: false, workers: 0, command: null });
  assert.equal(cfg.model, DEFAULT_CHAT_MODEL);
  assert.equal(cfg.model, 'gemma-3n-E2B-it-Q4_K_M.gguf');
  assert.equal(cfg.port, 8080);
  assert.equal(cfg.llamaUrl, '');
  assert.equal(cfg.workers, 0);
  assert.equal(cfg.command, null);
});

test('chatConfig: an explicit host override flows into llamaUrl', () => {
  const cfg = chatConfig({
    lastCommand: { model: 'm.gguf', port: 8090 },
    llamaUp: true,
    workers: 1,
    command: null,
    host: '192.168.1.5',
  });
  assert.equal(cfg.llamaUrl, 'http://192.168.1.5:8090/v1');
});

test('chatConfig: an explicit port (env override) beats the stored command port', () => {
  const cfg = chatConfig({
    lastCommand: { model: 'm.gguf', port: 8080 },
    llamaUp: true,
    workers: 1,
    command: null,
    port: 8123,
  });
  assert.equal(cfg.port, 8123);
  assert.equal(cfg.llamaUrl, 'http://127.0.0.1:8123/v1');
});
