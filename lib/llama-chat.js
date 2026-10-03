// MittiGrid v0.4 — grid chat parsing + config (pure module)
// Zero dependencies: no I/O, no state, no clock. The coordinator feeds it
// llama-server's reply and its own stored command state; this file decides
// what the /chat page sees. Pure so node --test can cover it.
//
// LAW OF THIS FILE: no number is ever invented. Tokens/sec comes ONLY from
// llama-server's own `timings.predicted_per_second` field. When timings are
// missing we fall back to wall-clock tokens/elapsed and label it "measured".
// When neither source can produce a number, tokensPerSecond is null and the
// source string says why.

// The model this grid is documented around (docs/REAL-MODELS.md). Used when
// no llama command has been generated yet — never a fabricated answer, just
// the default the runbook tells the owner to download.
export const DEFAULT_CHAT_MODEL = 'gemma-3n-E2B-it-Q4_K_M.gguf';

const CHAT_ROLES = ['system', 'user', 'assistant'];
const MAX_MESSAGES = 200;
const MAX_CONTENT_CHARS = 100000;

// Count from an upstream number: positive finite -> integer, else null.
function countOrNull(n) {
  const x = Number(n);
  return Number.isFinite(x) && x > 0 ? Math.round(x) : null;
}

function round2(x) {
  return Math.round(x * 100) / 100;
}

// Parse llama-server's OpenAI-compatible /v1/chat/completions response.
//   json   — the parsed upstream body
//   wallMs — wall-clock ms the proxied request took (for the measured fallback)
// Returns { ok, content, tokens, tokensPerSecond, tokensPerSecondSource } or
// { ok: false, error }.
export function chatResultFromUpstream(json, wallMs) {
  if (!json || typeof json !== 'object' || Array.isArray(json)) {
    return { ok: false, error: 'llama-server returned an unexpected response (not an object)' };
  }
  if (json.error) {
    const msg = typeof json.error === 'string'
      ? json.error
      : (json.error && json.error.message) || JSON.stringify(json.error);
    return { ok: false, error: `llama-server error: ${msg}` };
  }
  const choice = Array.isArray(json.choices) && json.choices.length ? json.choices[0] : null;
  const content = choice && choice.message && typeof choice.message.content === 'string'
    ? choice.message.content
    : null;
  if (content == null) {
    return { ok: false, error: 'llama-server returned no message content' };
  }

  const usageTokens = countOrNull(json.usage && json.usage.completion_tokens);
  const timings = json.timings && typeof json.timings === 'object' ? json.timings : null;
  const perSecond = timings ? Number(timings.predicted_per_second) : NaN;

  // Primary source: llama-server measured its own generation speed.
  if (Number.isFinite(perSecond) && perSecond > 0) {
    return {
      ok: true,
      content,
      tokens: countOrNull(timings.predicted_n) ?? usageTokens,
      tokensPerSecond: round2(perSecond),
      tokensPerSecondSource: 'llama-server timings',
    };
  }

  // Fallback: wall clock over the whole proxied request. Honest caveat: this
  // includes prompt processing and network time, so it is a lower bound.
  const seconds = Number(wallMs) / 1000;
  if (usageTokens && Number.isFinite(seconds) && seconds > 0) {
    return {
      ok: true,
      content,
      tokens: usageTokens,
      tokensPerSecond: round2(usageTokens / seconds),
      tokensPerSecondSource: 'measured (wall clock, includes prompt processing)',
    };
  }

  return {
    ok: true,
    content,
    tokens: usageTokens,
    tokensPerSecond: null,
    tokensPerSecondSource: 'unavailable (no timings and no token count from llama-server)',
  };
}

// Validate + normalize the {messages} array the /chat page posts. Returns a
// clean [{role, content}, ...] or null. Never throws.
export function validChatMessages(messages) {
  if (!Array.isArray(messages) || !messages.length || messages.length > MAX_MESSAGES) return null;
  const out = [];
  for (const m of messages) {
    if (!m || typeof m !== 'object') return null;
    if (!CHAT_ROLES.includes(m.role)) return null;
    if (typeof m.content !== 'string' || !m.content.trim()) return null;
    if (m.content.length > MAX_CONTENT_CHARS) return null;
    out.push({ role: m.role, content: m.content });
  }
  return out;
}

// Pure: the GET /api/llama/chat-config payload.
//   lastCommand — coordinator state: the most recent llama-server command its
//                 command generator produced ({model, port, ...}) | null
//   llamaUp     — result of the live /health probe
//   workers     — eligible rpc workers in the pool right now (count)
//   command     — the freshly generated llama-server command | null
//   host, port  — where the coordinator actually probed (env overrides applied)
// llamaUrl is EMPTY when no llama-server answered — the /chat page turns that
// into the honest "run this command" message. It never guesses a URL.
export function chatConfig({ lastCommand, llamaUp, workers, command, host = '127.0.0.1', port }) {
  let p = countOrNull(port) ? Math.floor(Number(port)) : 0;
  if (!p && lastCommand) p = countOrNull(lastCommand.port) ? Math.floor(Number(lastCommand.port)) : 0;
  if (!p) p = 8080;
  const model = lastCommand && typeof lastCommand.model === 'string' && lastCommand.model.trim()
    ? lastCommand.model
    : DEFAULT_CHAT_MODEL;
  const h = typeof host === 'string' && host.trim() ? host.trim() : '127.0.0.1';
  return {
    ok: true,
    ready: llamaUp === true,
    llamaUrl: llamaUp === true ? `http://${h}:${p}/v1` : '',
    model,
    port: p,
    workers: Number(workers) || 0,
    command: typeof command === 'string' && command ? command : null,
  };
}
