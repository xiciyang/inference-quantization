import { test } from 'node:test'
import assert from 'node:assert/strict'
import { planNextMethod, isDegradable, TERMINAL_FAILURES, describeChain, chainMethods } from '../../src/degrade.mjs'
import { allMethods } from '../../src/backends/index.mjs'
import { DEGRADE_CHAIN_IDS } from '../../src/backends/llmcompressor.mjs'

const methods = allMethods()

test('降级链: 档位次序与题面一致 —— INT4→INT8→FP8→W8A16', () => {
  assert.deepEqual(methods.map((m) => m.id), DEGRADE_CHAIN_IDS)
  assert.deepEqual(methods.map((m) => m.level), [1, 2, 3, 4])
  assert.deepEqual(methods.map((m) => m.bits), [4, 8, 8, 8])
})

test('降级链: 逐档推进，用尽后进入第 5 档 QAT 计划（不执行）', () => {
  let tried = []
  for (let i = 0; i < 4; i += 1) {
    const p = planNextMethod({ failureCode: 'E_ACCURACY_GATE_FAILED', triedLevels: tried, methods, maxSteps: 5 })
    assert.ok(p.method, `第 ${i + 1} 次应给出方法`)
    assert.equal(p.level, i + 1)
    tried = [...tried, p.level]
  }
  const p5 = planNextMethod({ failureCode: 'E_ACCURACY_GATE_FAILED', triedLevels: tried, methods, maxSteps: 5 })
  assert.equal(p5.qatPlan, true, '4 档用尽后应进入 QAT 计划档')
  assert.equal(p5.exhausted, false)
  assert.match(p5.reason, /QAT 计划/)
})

test('降级链: 有界 —— 试满 5 档后耗尽，不会无限重试', () => {
  const p = planNextMethod({ failureCode: 'E_ACCURACY_GATE_FAILED', triedLevels: [1, 2, 3, 4, 5], methods, maxSteps: 5 })
  assert.equal(p.exhausted, true)
  assert.equal(p.qatPlan, false)
  assert.match(p.reason, /最大降级档位 5/)
})

test('降级链: 不可降级的失败立即中止（换档位没用）', () => {
  for (const code of ['E_CALIB_LEAKAGE_DETECTED', 'E_SOURCE_MODEL_MODIFIED', 'E_THRESHOLD_NOT_PROVIDED',
    'E_SERVING_UNAVAILABLE', 'E_IMAGE_NOT_VERIFIED', 'E_SANDBOX_UNAVAILABLE', 'E_DATASET_UNAVAILABLE',
    'E_OUTPUT_DIR_UNSAFE', 'E_CAPABILITY_DENIED']) {
    assert.equal(isDegradable(code), false, `${code} 应不可降级`)
    const p = planNextMethod({ failureCode: code, triedLevels: [], methods, maxSteps: 5 })
    assert.equal(p.exhausted, true)
    assert.equal(p.terminal, true)
  }
  assert.ok(TERMINAL_FAILURES.has('E_QAT_REQUIRES_APPROVAL') === false, 'QAT 需批准不是"不可降级失败"（它是终态之一）')
})

test('降级链: 可降级的失败会给出下一档', () => {
  for (const code of ['E_ACCURACY_GATE_FAILED', 'E_PERF_GATE_FAILED', 'E_QUANTIZE_FAILED', 'E_QUANTIZE_TIMEOUT']) {
    assert.equal(isDegradable(code), true, `${code} 应可降级`)
    const p = planNextMethod({ failureCode: code, triedLevels: [1], methods, maxSteps: 5 })
    assert.equal(p.level, 2)
  }
})

test('降级链: describeChain 产出可读轨迹', () => {
  const s = describeChain([{ level: 1, method: 'awq-w4a16', failureCode: 'E_ACCURACY_GATE_FAILED' }, { level: 2, method: 'int8-w8a8' }])
  assert.equal(s, 'L1:awq-w4a16(E_ACCURACY_GATE_FAILED) → L2:int8-w8a8')
})

test('降级链: 方法表不可用时退回兜底次序（不崩）', () => {
  const c = chainMethods([])
  assert.equal(c.length, 4)
  assert.equal(c[0].id, 'awq-w4a16')
})
