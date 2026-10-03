// MittiGrid v0.4 — battery awareness (shared by agent and coordinator)
// Zero dependencies, node built-ins only.
//
// A reading is { level, charging, mocked, low }:
//   level    0..100 (integer)
//   charging true when plugged in / full
//   mocked   true when the value came from MITTI_FAKE_BATTERY (test rigs)
//   low      discharging and below LOW_BATTERY_PCT — the standby trigger
//
// Sources, in order: env MITTI_FAKE_BATTERY (a set env is authoritative, even
// when it is garbage — deterministic test rigs beat surprises), then
// `termux-battery-status` on Android/Termux (3s cap), else null. Devices with
// no battery reading simply never go standby.

import { execFile } from 'node:child_process';

export const LOW_BATTERY_PCT = 30; // discharging below this -> standby
const TERMUX_TIMEOUT_MS = 3000;

// low = discharging and level < LOW_BATTERY_PCT. Unknown battery -> never low.
export function batteryLow(b) {
  if (!b || typeof b !== 'object') return false;
  const level = Number(b.level);
  return b.charging === false && Number.isFinite(level) && level < LOW_BATTERY_PCT;
}

// Accepts the fake-battery shape ({level, charging, ...}) and the Termux
// shape ({percentage, status: 'CHARGING'|'FULL'|'DISCHARGING'|...}). Returns
// a normalized reading, or null when the input carries no usable battery info.
export function normalizeBattery(raw, mocked = false) {
  if (!raw || typeof raw !== 'object') return null;
  let level = raw.level != null ? Number(raw.level) : Number(raw.percentage);
  if (!Number.isFinite(level)) return null;
  level = Math.max(0, Math.min(100, Math.round(level)));
  let charging;
  if (typeof raw.charging === 'boolean') charging = raw.charging;
  else if (raw.status === 'CHARGING' || raw.status === 'FULL') charging = true;
  else if (typeof raw.status === 'string' && raw.status) charging = false;
  else return null;
  return { level, charging, mocked: mocked === true, low: batteryLow({ level, charging }) };
}

// Parse a MITTI_FAKE_BATTERY string. Garbage -> null (never throws).
export function fakeBattery(json) {
  if (typeof json !== 'string' || !json.trim()) return null;
  let parsed;
  try {
    parsed = JSON.parse(json);
  } catch {
    return null;
  }
  return normalizeBattery(parsed, true);
}

// termux-battery-status, capped at TERMUX_TIMEOUT_MS. Anything unexpected
// (missing binary, non-JSON output, timeout) resolves null — never throws.
export function termuxBattery() {
  return new Promise((resolve) => {
    let settled = false;
    const done = (v) => {
      if (!settled) {
        settled = true;
        resolve(v);
      }
    };
    try {
      execFile('termux-battery-status', { timeout: TERMUX_TIMEOUT_MS }, (err, stdout) => {
        if (err) return done(null);
        let parsed;
        try {
          parsed = JSON.parse(String(stdout));
        } catch {
          return done(null);
        }
        done(normalizeBattery(parsed, false));
      });
    } catch {
      done(null);
    }
  });
}

// The agent's entry point: env override first (authoritative when set), then
// the Termux probe. `env` may be passed for tests; defaults to process.env.
export async function readBattery(env = process.env) {
  const v = env && env.MITTI_FAKE_BATTERY;
  if (v != null && v !== '') return fakeBattery(v);
  return termuxBattery();
}
