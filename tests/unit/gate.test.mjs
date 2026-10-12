import { test } from 'node:test'
import assert from 'node:assert/strict'
import { resolveThresholds, evaluateAccuracyGate, evaluatePerfGate, evaluateGates, nearLimitTokenFor, GATE_FIELDS } from '../../src/gate.mjs'
import { ERROR_CODES } from '../../src/errors.mjs'
import { makeSandbox } from '../helpers/sandbox.mjs'
import { createRuntime } from '../../src/config.mjs'

async function rt(gates = null) {
  const sb = await makeSandbox()
  const runtime = await createRuntime({ root: sb.root, options: sb.options })
  if (gates) runtime.policy.gates = gates
  return { sb, runtime }
}

test('门禁: 阈值必须由调用方传入；不传且策略文件没设 → 拒绝执行（不猜默认值）', async () => {
  const { sb, runtime } = await rt(null)
  try {
    assert.throws(() => resolveThresholds(runtime, {}), (err) => {
      assert.equal(err.code, ERROR_CODES.E_THRESHOLD_NOT_PROVIDED)
      assert.match(err.message, /max_accuracy_drop_pct/)
      assert.match(err.message, /min_perf_gain_pct/)
      assert.match(err.remediation, /不会.*猜一个默认阈值/)
      return true
    })
  } finally { await sb.cleanup() }
})

test('门禁: 只传一个也不够（两个都必须有来源）', async () => {
  const { sb, runtime } = await rt(null)
  try {
    assert.throws(() => resolveThresholds(runtime, { maxAccuracyDropPct: 5 }), (err) => {
      assert.equal(err.code, ERROR_CODES.E_THRESHOLD_NOT_PROVIDED)
      assert.deepEqual(err.details.missing, ['min_perf_gain_pct（性能收益目标）'])
      return true
    })
  } finally { await sb.cleanup() }
})

test('门禁: 调用方传入优先于 gates.yaml', async () => {
  const { sb, runtime } = await rt({ max_accuracy_drop_pct: 9, min_perf_gain_pct: 3 })
  try {
    const a = resolveThresholds(runtime, { maxAccuracyDropPct: 2, minPerfGainPct: 50 })
    assert.equal(a.maxAccuracyDropPct, 2)
    assert.equal(a.source.maxAccuracyDropPct, 'caller')
    const b = resolveThresholds(runtime, {})
    assert.equal(b.maxAccuracyDropPct, 9)
    assert.equal(b.source.maxAccuracyDropPct, 'gates.yaml')
  } finally { await sb.cleanup() }
})

test('门禁: 负的精度容忍被拒绝', async () => {
  const { sb, runtime } = await rt({ min_perf_gain_pct: 10 })
  try {
    assert.throws(() => resolveThresholds(runtime, { maxAccuracyDropPct: -1 }), (err) => {
      assert.equal(err.code, ERROR_CODES.E_THRESHOLD_NOT_PROVIDED)
      assert.match(err.message, /不能为负/)
      return true
    })
  } finally { await sb.cleanup() }
})

test('门禁: 精度判定 —— 通过 / 不通过 / 接近上限', () => {
  assert.equal(evaluateAccuracyGate({ dropPct: 1, thresholdPct: 5 }).passed, true)
  assert.equal(evaluateAccuracyGate({ dropPct: 5, thresholdPct: 5 }).passed, true, '等于阈值算通过（≤）')
  assert.equal(evaluateAccuracyGate({ dropPct: 5.1, thresholdPct: 5 }).passed, false)
  const near = evaluateAccuracyGate({ dropPct: 4.6, thresholdPct: 5 })
  assert.equal(near.passed, true)
  assert.equal(near.nearLimit, true, `4.6/5=92% 应判接近上限`)
  assert.equal(evaluateAccuracyGate({ dropPct: 3, thresholdPct: 5 }).nearLimit, false)
})

test('门禁: 未测得的数值判为 undetermined，不能算通过', () => {
  const a = evaluateAccuracyGate({ dropPct: null, thresholdPct: 5 })
  assert.equal(a.passed, false)
  assert.equal(a.undetermined, true)
  const p = evaluatePerfGate({ gainPct: null, thresholdPct: 10 })
  assert.equal(p.passed, false)
  assert.equal(p.undetermined, true)
})

test('门禁: 性能收益为负（变慢）必须判不通过', () => {
  assert.equal(evaluatePerfGate({ gainPct: -5, thresholdPct: 0 }).passed, false)
  assert.equal(evaluatePerfGate({ gainPct: 0, thresholdPct: 0 }).passed, true)
  assert.equal(evaluatePerfGate({ gainPct: 25, thresholdPct: 20 }).passed, true)
})

test('门禁: 双门禁合并 —— 只有都通过才是 sign；精度通过+性能不过 = degrade', () => {
  const th = { maxAccuracyDropPct: 5, minPerfGainPct: 20 }
  const ok = evaluateGates({ accuracy: { dropPct: 1 }, perf: { gainPct: 30 }, thresholds: th })
  assert.equal(ok.passed, true)
  assert.equal(ok.decision, 'sign')
  assert.equal(ok.signable, true)

  const perfFail = evaluateGates({ accuracy: { dropPct: 1 }, perf: { gainPct: 5 }, thresholds: th })
  assert.equal(perfFail.passed, false)
  assert.equal(perfFail.decision, 'degrade')

  const accFail = evaluateGates({ accuracy: { dropPct: 9 }, perf: { gainPct: 50 }, thresholds: th })
  assert.equal(accFail.decision, 'degrade')
  assert.match(accFail.reasons.join(' '), /精度/)

  const near = evaluateGates({ accuracy: { dropPct: 4.8 }, perf: { gainPct: 30 }, thresholds: th })
  assert.equal(near.passed, true)
  assert.equal(near.decision, 'confirm-near-limit')
  assert.equal(near.needsConfirmation, true)
  assert.equal(near.signable, false, '接近上限时不能直接签收')
})

test('门禁: 接近上限的确认令牌与"模型+方法+精度值"绑定（不同目标不复用）', () => {
  const a = nearLimitTokenFor({ sourceModel: '/m/a', method: 'awq-w4a16', dropPct: 4.8 })
  const b = nearLimitTokenFor({ sourceModel: '/m/b', method: 'awq-w4a16', dropPct: 4.8 })
  const c = nearLimitTokenFor({ sourceModel: '/m/a', method: 'w8a16', dropPct: 4.8 })
  assert.match(a, /^nlq_/)
  assert.notEqual(a, b)
  assert.notEqual(a, c)
  assert.equal(a, nearLimitTokenFor({ sourceModel: '/m/a', method: 'awq-w4a16', dropPct: 4.8 }))
})

test('门禁: GATE_FIELDS 与 gates.yaml 的键名一致（契约不漂移）', () => {
  assert.deepEqual(GATE_FIELDS, ['max_accuracy_drop_pct', 'min_perf_gain_pct'])
})
