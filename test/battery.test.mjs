// MittiGrid v0.4 — unit tests for lib/battery.js (fake-battery variants,
// normalization, the low-battery rule, env priority)
// Run: node --test test/

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { batteryLow, normalizeBattery, fakeBattery, readBattery, LOW_BATTERY_PCT } from '../lib/battery.js';

test('LOW_BATTERY_PCT is the documented 30% standby threshold', () => {
  assert.equal(LOW_BATTERY_PCT, 30);
});

test('fake JSON: discharging below threshold -> low true (standby)', () => {
  const b = fakeBattery('{"level":22,"charging":false}');
  assert.ok(b);
  assert.equal(b.level, 22);
  assert.equal(b.charging, false);
  assert.equal(b.low, true);
  assert.equal(b.mocked, true, 'env-sourced readings are always mocked');
});

test('fake JSON: charging -> never low, even at 1%', () => {
  const b = fakeBattery('{"level":1,"charging":true}');
  assert.ok(b);
  assert.equal(b.low, false);
});

test('fake JSON: discharging above threshold -> low false', () => {
  const b = fakeBattery('{"level":80,"charging":false,"mocked":false}');
  assert.ok(b);
  assert.equal(b.low, false);
  assert.equal(b.mocked, true, 'env wins over a mocked:false in the payload');
});

test('fake JSON: garbage -> null (never throws)', () => {
  assert.equal(fakeBattery('not json'), null);
  assert.equal(fakeBattery('{"level":'), null);
  assert.equal(fakeBattery(''), null);
  assert.equal(fakeBattery('   '), null);
});

test('fake JSON: wrong shapes -> null', () => {
  assert.equal(fakeBattery('42'), null);
  assert.equal(fakeBattery('"hi"'), null);
  assert.equal(fakeBattery('null'), null);
  assert.equal(fakeBattery('[]'), null);
  assert.equal(fakeBattery('{}'), null, 'no level, no charging/status');
  assert.equal(fakeBattery('{"charging":false}'), null, 'charging without a level');
  assert.equal(fakeBattery('{"level":"abc","charging":false}'), null, 'non-numeric level');
  assert.equal(fakeBattery(null), null);
  assert.equal(fakeBattery(undefined), null);
  assert.equal(fakeBattery(5), null);
});

test('fake JSON: levels are clamped to 0..100 and rounded', () => {
  assert.equal(fakeBattery('{"level":150,"charging":true}').level, 100);
  assert.equal(fakeBattery('{"level":-5,"charging":true}').level, 0);
  assert.equal(fakeBattery('{"level":22.6,"charging":true}').level, 23);
});

test('batteryLow rule: discharging and strictly below threshold', () => {
  assert.equal(batteryLow({ level: 29, charging: false }), true);
  assert.equal(batteryLow({ level: 30, charging: false }), false, '30% is not low');
  assert.equal(batteryLow({ level: 5, charging: true }), false, 'charging is never low');
  assert.equal(batteryLow({ level: 5, charging: false }), true);
  assert.equal(batteryLow(null), false);
  assert.equal(batteryLow(undefined), false);
  assert.equal(batteryLow('nope'), false);
  assert.equal(batteryLow({ charging: false }), false, 'no level -> unknown -> not low');
});

test('normalizeBattery accepts the Termux shape', () => {
  const charging = normalizeBattery({ percentage: 55, status: 'CHARGING' });
  assert.deepEqual(charging, { level: 55, charging: true, mocked: false, low: false });
  const full = normalizeBattery({ percentage: 100, status: 'FULL' });
  assert.equal(full.charging, true);
  const discharging = normalizeBattery({ percentage: 10, status: 'DISCHARGING' });
  assert.equal(discharging.charging, false);
  assert.equal(discharging.low, true);
});

test('normalizeBattery rejects unusable input', () => {
  assert.equal(normalizeBattery(null), null);
  assert.equal(normalizeBattery({ percentage: 50 }), null, 'level but no charging/status signal');
  assert.equal(normalizeBattery({ charging: true }), null);
});

test('readBattery: a set MITTI_FAKE_BATTERY is authoritative (even garbage -> null)', async () => {
  const good = await readBattery({ MITTI_FAKE_BATTERY: '{"level":90,"charging":true}' });
  assert.equal(good.level, 90);
  assert.equal(good.mocked, true);
  const garbage = await readBattery({ MITTI_FAKE_BATTERY: 'garbage' });
  assert.equal(garbage, null, 'garbage env beats falling back to the Termux probe');
  const empty = await readBattery({ MITTI_FAKE_BATTERY: '' });
  assert.ok(empty === null || empty === undefined || typeof empty === 'object',
    'empty env falls through to the probe (null on machines without termux-battery-status)');
});
