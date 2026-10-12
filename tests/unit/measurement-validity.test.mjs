/**
 * 测量成立性门禁 —— 单元测试。
 *
 * 用户裁决（2026-09-24）：门禁不仅要判"数值是否达标"，还要判"测量是否成立"。
 *
 * 本组测试的第二条用例直接**复刻真机事故的原始数字**：
 *   meanNll = 11.93121445743442, perplexity = 151935.96944645362, vocab_size = 151936
 *   （ln(151936) = 11.9312146585，差 2.0e-7 —— 落在 float32 累积误差内）
 * 这个数在数学上没错，但它是"输出均匀分布"的签名，此时 dropPct = 0 毫无意义。
 * 旧门禁据此签收了"精度零损失"。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { assertMeasurementValid, evaluateGates } from '../../src/gate.mjs'

/** 造一个"健康"的测量侧。 */
function healthy({ meanNll = 2.556, vocabSize = 151936, n = 50, jitter = 0.2, scoredSamples = n } = {}) {
  const itemMeanNlls = Array.from({ length: n }, (_, i) => meanNll + jitter * Math.sin(i * 1.7))
  return {
    meanNll,
    perplexity: Math.exp(meanNll),
    vocabSize,
    scoredSamples,
    validity: { valid: true, reasons: [] },
    weightHealth: { available: true, zeroTensorCount: 0 },
    itemMeanNlls,
  }
}

test('测量成立性: 健康测量 → valid', () => {
  const r = assertMeasurementValid({ baseline: healthy(), candidate: healthy({ meanNll: 2.6 }) })
  assert.equal(r.valid, true, JSON.stringify(r.reasons))
  assert.equal(r.reasons.length, 0)
  assert.equal(r.checks.bitIdenticalMeanNll, false)
  // 均匀分布签名距离很远
  assert.ok(r.checks.baselineUniformGap > 9, `gap 应很大，实际 ${r.checks.baselineUniformGap}`)
})

test('测量成立性: 复刻真机事故数值 → 判"均匀输出"，测量不成立', () => {
  const incident = {
    meanNll: 11.93121445743442,
    perplexity: 151935.96944645362,
    vocabSize: 151936,
    scoredSamples: 400,
    itemMeanNlls: Array.from({ length: 400 }, () => 11.93121445743442),
  }
  const r = assertMeasurementValid({ baseline: incident, candidate: incident })
  assert.equal(r.valid, false, '真机事故的数值必须被判为"测量不成立"')
  const codes = r.reasons.map((x) => x.code)
  assert.ok(codes.includes('uniform-output'), `应含 uniform-output，实际 ${JSON.stringify(codes)}`)
  assert.ok(codes.includes('indiscriminate'), `应含 indiscriminate，实际 ${JSON.stringify(codes)}`)
  assert.ok(codes.includes('no-item-variance'), `应含 no-item-variance，实际 ${JSON.stringify(codes)}`)
  // 证据要能复核：把 ln(vocab) 的差写出来
  assert.ok(r.checks.baselineUniformGap <= 1e-4)
  assert.equal(r.checks.baselineVocabSize, 151936)
})

test('测量成立性: 基线与候选逐位相同 → indiscriminate（与权重无关的数不构成测量）', () => {
  const b = healthy({ meanNll: 2.5 })
  const c = healthy({ meanNll: 2.5 })
  const r = assertMeasurementValid({ baseline: b, candidate: c })
  assert.equal(r.valid, false)
  assert.deepEqual(r.reasons.map((x) => x.code), ['indiscriminate'])
  assert.equal(r.checks.bitIdenticalMeanNll, true)
})

test('测量成立性: 逐条 meanNll 极差为 0 → no-item-variance（用极差而非标准差，不需要容差魔数）', () => {
  const flat = healthy({ n: 30, jitter: 0 })
  const r = assertMeasurementValid({ baseline: flat, candidate: healthy({ meanNll: 2.9 }) })
  assert.equal(r.valid, false)
  assert.deepEqual(r.reasons.map((x) => x.code), ['no-item-variance'])
  assert.equal(r.checks.baselineItemMeanNllRange, 0)
  // 极差为 0 的判定不依赖任何浮点容差：标准差算出来可能是 ~1e-15
  assert.equal(r.checks.baselineItemMeanNllMin, r.checks.baselineItemMeanNllMax)
})

test('测量成立性: 单条样本不适用方差判定（极差无意义），但仍可能被均匀签名拦下', () => {
  const one = { ...healthy({ n: 1 }), itemMeanNlls: [2.5] }
  const r = assertMeasurementValid({ baseline: one, candidate: healthy({ meanNll: 2.6 }) })
  assert.equal(r.valid, true, JSON.stringify(r.reasons))
})

test('测量成立性: sidecar 自报无效 → 原样上抛，保留原因', () => {
  const bad = { ...healthy(), validity: { valid: false, reasons: [{ code: 'zero-weight-tensors', detail: '3 个大张量整块为零' }] } }
  const r = assertMeasurementValid({ baseline: bad, candidate: healthy({ meanNll: 2.6 }) })
  assert.equal(r.valid, false)
  assert.equal(r.reasons[0].code, 'sidecar:zero-weight-tensors')
  assert.match(r.reasons[0].message, /3 个大张量整块为零/)
})

test('测量成立性: 一侧没有任何样本评分成功 → no-scored-samples', () => {
  const r = assertMeasurementValid({ baseline: healthy({ scoredSamples: 0 }), candidate: healthy({ meanNll: 2.6 }) })
  assert.equal(r.valid, false)
  assert.ok(r.reasons.some((x) => x.code === 'no-scored-samples'))
})

test('测量成立性: 缺一侧结果 → missing-side（不静默当通过）', () => {
  const r = assertMeasurementValid({ baseline: healthy(), candidate: null })
  assert.equal(r.valid, false)
  assert.ok(r.reasons.some((x) => x.code === 'missing-side'))
})

test('测量成立性: 上游没给 vocabSize 时不做均匀判定（不猜词表大小）', () => {
  const b = { ...healthy(), vocabSize: null }
  const c = { ...healthy({ meanNll: 2.6 }), vocabSize: null }
  const r = assertMeasurementValid({ baseline: b, candidate: c })
  assert.equal(r.valid, true)
  assert.equal(r.checks.baselineUniformGap, null)
})

// ---------------------------------------------------------------- 与门禁联动

test('门禁: 测量不成立 → decision=undetermined、不可签收（哪怕数值全部达标）', () => {
  const incident = {
    meanNll: 11.93121445743442, perplexity: 151935.96944645362, vocabSize: 151936,
    scoredSamples: 400, itemMeanNlls: Array.from({ length: 400 }, () => 11.93121445743442),
  }
  const measurement = assertMeasurementValid({ baseline: incident, candidate: incident })
  const g = evaluateGates({
    accuracy: { dropPct: 0 },            // 数值"完美达标"
    perf: { gainPct: 500 },              // 性能也"远超目标"
    thresholds: { maxAccuracyDropPct: 50, minPerfGainPct: 0 },
    measurement,
  })
  assert.equal(g.decision, 'undetermined')
  assert.equal(g.passed, false, '数值再好，测量不成立也不得判通过')
  assert.equal(g.signable, false)
  assert.equal(g.measurementValid, false)
  assert.ok(g.reasons.some((x) => x.startsWith('测量：')), JSON.stringify(g.reasons))
})

test('门禁: 测量成立 + 数值达标 → 行为与旧版一致（sign）', () => {
  const measurement = assertMeasurementValid({ baseline: healthy(), candidate: healthy({ meanNll: 2.6 }) })
  const g = evaluateGates({
    accuracy: { dropPct: 5 }, perf: { gainPct: 30 },
    thresholds: { maxAccuracyDropPct: 10, minPerfGainPct: 20 }, measurement,
  })
  assert.equal(g.decision, 'sign')
  assert.equal(g.passed, true)
  assert.equal(g.signable, true)
  assert.equal(g.measurementValid, true)
})

test('门禁: 不传 measurement 时保持旧行为（纯数值判定仍可被独立测试）', () => {
  const g = evaluateGates({
    accuracy: { dropPct: 5 }, perf: { gainPct: 30 },
    thresholds: { maxAccuracyDropPct: 10, minPerfGainPct: 20 },
  })
  assert.equal(g.decision, 'sign')
  assert.equal(g.measurementValid, null)
})

test('门禁: 测量不成立优先于"降级" —— 不能靠换档位绕过', () => {
  const measurement = assertMeasurementValid({
    baseline: healthy({ meanNll: 11.93121445743442, vocabSize: 151936, jitter: 0 }),
    candidate: healthy({ meanNll: 12.5, vocabSize: 151936 }),
  })
  const g = evaluateGates({
    accuracy: { dropPct: 999 }, perf: { gainPct: -50 },
    thresholds: { maxAccuracyDropPct: 10, minPerfGainPct: 20 }, measurement,
  })
  assert.equal(g.decision, 'undetermined', '应是"测量不成立"，而不是"精度/性能不达标→降级"')
  assert.equal(g.passed, false)
})
