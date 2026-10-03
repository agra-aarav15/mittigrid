// MittiGrid v0.2 — toy model, shared by coordinator and agents
// Zero dependencies, node built-ins only. Deterministic: weights come from a
// fixed-seed PRNG, so every device computes byte-identical weights and a
// forward pass is reproducible bit-for-bit.
//
// HONEST LABEL: untrained toy model — it proves the pipeline (shard
// assignment, activations hopping device-to-device, failover), not the
// quality. Real 700B-class models arrive via llama.cpp RPC / exo-style
// sharding (see README roadmap).
//
// Shape: input vector of 8 numbers
//        -> 12 layers, each an 8x8 matvec + tanh (activations stay 8-dim)
//        -> head: 8 -> 4 logits, argmax -> token id (0..3), applied by
//          whoever holds the last layer.

export const MODEL_LAYERS = 12;
export const DIM = 8;
export const HEAD_DIM = 4;

// mulberry32 — tiny deterministic PRNG
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildWeights() {
  const rand = mulberry32(0x4d495454); // "MITT"
  const layers = [];
  for (let i = 0; i < MODEL_LAYERS; i++) {
    const W = [];
    for (let r = 0; r < DIM; r++) {
      const row = [];
      for (let c = 0; c < DIM; c++) row.push(rand() * 2 - 1);
      W.push(row);
    }
    layers.push(W);
  }
  const headW = [];
  for (let r = 0; r < HEAD_DIM; r++) {
    const row = [];
    for (let c = 0; c < DIM; c++) row.push(rand() * 2 - 1);
    headW.push(row);
  }
  return { layers, headW };
}

const W = buildWeights();

function matvec(Wm, v) {
  const out = new Array(Wm.length);
  for (let r = 0; r < Wm.length; r++) {
    const row = Wm[r];
    let s = 0;
    for (let c = 0; c < DIM; c++) s += row[c] * v[c];
    out[r] = s;
  }
  return out;
}

export function validInput(x) {
  return Array.isArray(x) && x.length === DIM && x.every((n) => Number.isFinite(Number(n)));
}

// Run layers [startLayer, endLayer) on `vec` — the unit one device executes
// when it hosts a shard. Same input, same range, same result bits, no matter
// which device ran it or how the layers are split across hops.
export function runLayers(vec, startLayer, endLayer) {
  if (!Number.isInteger(startLayer) || !Number.isInteger(endLayer) ||
      startLayer < 0 || endLayer > MODEL_LAYERS || startLayer >= endLayer) {
    throw new Error(`invalid layer range [${startLayer}, ${endLayer})`);
  }
  let v = vec.slice();
  for (let i = startLayer; i < endLayer; i++) {
    v = matvec(W.layers[i], v).map(Math.tanh);
  }
  return v;
}

// Final head, applied by whoever holds the last layer: 8 -> 4, argmax.
export function head(vec) {
  const logits = matvec(W.headW, vec);
  let token = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[token]) token = i;
  return { logits, token };
}
