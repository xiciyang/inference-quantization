/**
 * validate-existing（M2 的"验收已有 checkpoint"入口）—— **可穷举的那一部分**。
 *
 * 这个测试刻意只测**纯函数**与**形状**，不测 parityCheck 的真机路径（那需要容器 + GPU）：
 * 真机路径由 `qat-verification-contract.md` §7 的 A7 条件单独要求，不在无 GPU 的 CI 里假装通过。
 *
 * 核心断言：`accuracyOnlyGate` 与既有的 `gate.evaluateGates` 对**精度维**的处置逐点一致
 * （同一个 `evaluateAccuracyGate`、同一个 nearLimitFactor、同一条决策阶梯）。
 * 这是"复用既有判据"这句话的**可执行形式** —— 不是"看着一样"。
 */
import test from 'node:test'
import assert from 'node:assert/strict'

import { accuracyOnlyGate } from '../../src/validate-existing.mjs'
import { evaluateGates, evaluateAccuracyGate, assertMeasurementValid } from '../../src/gate.mjs'

const TH = { maxAccuracyDropPct: 2, minPerfGainPct: 10 }
const VALID = { valid: true, reasons: [], checks: {} }

test('validate-existing: 精度维门禁与 evaluateGates 的精度处置逐点一致', () => {
  const drops = [-1, 0, 0.5, 1.799, 1.8, 1.99, 2, 2.0001, 3, 10, 100]
  for (const dropPct of drops) {
    const mine = accuracyOnlyGate({ dropPct, thresholds: TH, measurement: VALID })
    // evaluateGates 的精度维：给一个**远超阈值**的性能收益，使决策完全由精度维决定
    const ref = evaluateGates({ accuracy: { dropPct }, perf: { gainPct: 999 }, thresholds: TH, measurement: VALID })
    assert.equal(mine.accuracy.passed, ref.accuracy.passed, `dropPct=${dropPct} 精度通过性一致`)
    assert.equal(mine.accuracy.nearLimit, ref.accuracy.nearLimit, `dropPct=${dropPct} nearLimit 一致`)
    assert.equal(mine.accuracy.ratioOfThreshold, ref.accuracy.ratioOfThreshold, `dropPct=${dropPct} ratio 一致`)
    assert.equal(mine.pass, ref.passed, `dropPct=${dropPct} 总体 pass 一致`)
    assert.equal(mine.decision, ref.decision, `dropPct=${dropPct} 决策一致`)
    assert.equal(mine.accuracy.reason, ref.accuracy.reason, `dropPct=${dropPct} 理由文案一致`)
  }
})

test('validate-existing: 决策阶梯的四个取值都能到达', () => {
  const seen = new Set()
  for (const dropPct of [-1, 0, 1.81, 3]) seen.add(accuracyOnlyGate({ dropPct, thresholds: TH, measurement: VALID }).decision)
  seen.add(accuracyOnlyGate({ dropPct: null, thresholds: TH, measurement: VALID }).decision)
  seen.add(accuracyOnlyGate({ dropPct: 0.1, thresholds: TH, measurement: { valid: false, reasons: [{ code: 'x', message: 'm' }] } }).decision)
  assert.deepEqual([...seen].sort(), ['confirm-near-limit', 'degrade', 'sign', 'undetermined'])
})

test('validate-existing: 测量不成立 ⇒ 不可签收（数值再好也不签）', () => {
  const bad = { valid: false, reasons: [{ code: 'uniform-output', message: '均匀输出' }] }
  const g = accuracyOnlyGate({ dropPct: 0, thresholds: TH, measurement: bad })
  assert.equal(g.pass, false, 'dropPct=0 也不签收')
  assert.equal(g.decision, 'undetermined')
  assert.ok(g.reasons.some((r) => /测量：/.test(r)), '理由要能区分"测量不成立"与"数值未达标"')
})

test('validate-existing: 近上限只要求确认、但不等于签收', () => {
  const g = accuracyOnlyGate({ dropPct: 1.9, thresholds: TH, measurement: VALID })
  assert.equal(g.decision, 'confirm-near-limit')
  assert.equal(g.needsConfirmation, true)
  assert.equal(g.signable, false)
})

test('validate-existing: dropPct 未测得 ⇒ undetermined 且理由点明 E_ACCURACY_GATE_MISSING', () => {
  const g = accuracyOnlyGate({ dropPct: null, thresholds: TH, measurement: VALID })
  assert.equal(g.decision, 'undetermined')
  assert.equal(g.pass, false)
  assert.ok(g.reasons.some((r) => /E_ACCURACY_GATE_MISSING/.test(r)))
})

test('validate-existing: 阈值必须来自调用方（本入口不造默认值）', () => {
  // 这条是**源码级**断言：纯度函数不含任何数值默认阈值；未给阈值时 evaluateAccuracyGate 判不通过。
  const g = accuracyOnlyGate({ dropPct: 0.5, thresholds: { maxAccuracyDropPct: undefined }, measurement: VALID })
  assert.equal(g.accuracy.passed, false, '阈值 undefined ⇒ 不通过（不猜）')
  assert.equal(g.accuracy.thresholdPct, undefined)
})

test('validate-existing: 与 assertMeasurementValid 的组合语义一致（先判测量、再判数值）', () => {
  const sides = {
    baseline: { meanNll: 3.0, perplexity: 20, vocabSize: 1000, scoredSamples: 10, itemMeanNlls: [2.9, 3.1, 3.0] },
    candidate: { meanNll: 3.05, perplexity: 21, vocabSize: 1000, scoredSamples: 10, itemMeanNlls: [2.95, 3.15, 3.05] },
  }
  const m = assertMeasurementValid(sides)
  assert.equal(m.valid, true, '有区分力的测量成立')
  const g = accuracyOnlyGate({ dropPct: 5, thresholds: TH, measurement: m })
  assert.equal(g.decision, 'degrade', '测量成立且数值未达标 ⇒ degrade（两类失败可区分）')

  const identical = assertMeasurementValid({
    baseline: sides.baseline,
    candidate: { ...sides.candidate, meanNll: 3.0 },
  })
  assert.equal(identical.valid, false, '两侧 meanNll 逐位相同 ⇒ 测量不成立')
  const g2 = accuracyOnlyGate({ dropPct: 0, thresholds: TH, measurement: identical })
  assert.equal(g2.decision, 'undetermined', '测量不成立优先于数值')
  assert.notEqual(evaluateAccuracyGate({ dropPct: 0, thresholdPct: 2 }).passed, undefined)
})
