/**
 * F-5 回归：失效判定必须**落盘且机读**。
 *
 * 背景（集成测试 S3-11/12/12b/12c 实测）：`qr_25b26474a21b` 在矩阵里长期是
 * `status: verified` / `accuracy.passed: true` / `ppl = 151935.97`（= vocab_size，
 * 均匀输出签名），而它产生于一个 223/338 张量全零的残缺模型。
 * 作废只写在报告散文里 —— 下游读状态文件，读不到的不存在。
 *
 * 本文件把"散文里的作废"变成"状态里的字段"，并锁住几个关键语义：
 *   - 失效标记**不改 status**（状态是当时的事实，失效是事后的判定 —— 两个不同的事实）
 *   - `summarizeRun` / `list` 必须把标记带出去（否则列表会把作废记录显示得像正常记录）
 *   - `isEvidenceGrade` 是下游唯一该用的判据
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { makeSandbox } from '../helpers/sandbox.mjs'
import { createRuntime } from '../../src/config.mjs'
import { withMatrix, findRun, markRunInvalid, isRunInvalid, isEvidenceGrade, summarizeRun } from '../../src/state/matrix.mjs'
import { KNOWN_INVALID_RUNS, retroMark } from '../../scripts/retro-mark-invalid.mjs'

test('F-5: 失效标记落盘 + 进 history，且**不改 status**', async () => {
  const sb = await makeSandbox()
  const runtime = await createRuntime({ root: sb.root, options: {} })
  try {
    await withMatrix(runtime, (m) => {
      m.runs.push({ id: 'qr_test', runKey: 'k', sourceModel: '/m', status: 'verified', accuracy: { dropPct: 0, passed: true } })
      return null
    })
    await withMatrix(runtime, (m) => {
      const r = findRun(m, 'qr_test')
      markRunInvalid(r, { code: 'E_MEASUREMENT_INVALID', reason: '均匀输出', evidence: { x: 1 } })
      return null
    })
    await withMatrix(runtime, (m) => {
      const r = findRun(m, 'qr_test')
      assert.equal(r.status, 'verified', '失效标记不得改写 status —— 那是当时发生的事实')
      assert.equal(r.invalid.code, 'E_MEASUREMENT_INVALID')
      assert.ok(r.invalid.at)
      assert.equal(r.invalid.evidence.x, 1)
      assert.ok(r.history.some((h) => h.event === 'marked-invalid'), '失效必须进 history')
      assert.equal(isRunInvalid(r), true)
      assert.equal(isEvidenceGrade(r), false, '带失效标记的 run 不得作为验收证据')
      // 列表视图必须带上标记
      assert.equal(summarizeRun(r).invalid.code, 'E_MEASUREMENT_INVALID')
      return null
    })
  } finally { await sb.cleanup() }
})

test('F-5: 无标记的 verified run 才算可作证据', () => {
  assert.equal(isEvidenceGrade({ status: 'verified' }), true)
  assert.equal(isEvidenceGrade({ status: 'failed' }), false)
  assert.equal(isEvidenceGrade({ status: 'verified', invalid: { code: 'X' } }), false)
  assert.equal(isEvidenceGrade(null), false)
})

test('F-5: 历史失效清单是幂等的，且每条都带可复核依据', async () => {
  const sb = await makeSandbox()
  const runtime = await createRuntime({ root: sb.root, options: {} })
  try {
    // 造一条与历史同 id 的记录
    const spec = KNOWN_INVALID_RUNS[0]
    await withMatrix(runtime, (m) => {
      m.runs.push({ id: spec.id, runKey: 'k', sourceModel: spec.evidence.sourceModel, status: 'verified', accuracy: { dropPct: 0, passed: true } })
      return null
    })
    const first = await retroMark({ root: sb.root, apply: true })
    assert.deepEqual(first.marked.map((x) => x.id), [spec.id])
    const second = await retroMark({ root: sb.root, apply: true })
    assert.deepEqual(second.marked, [], '第二次不得重复写')
    assert.deepEqual(second.alreadyMarked, [spec.id])

    // 每条历史判定都必须写明依据与来源，否则就是"凭感觉作废"
    for (const s of KNOWN_INVALID_RUNS) {
      assert.ok(s.code && s.reason && s.evidence, `${s.id} 缺依据`)
      assert.ok(s.evidence.replacedBy || s.evidence.detectedBy, `${s.id} 必须指向替代证据或检出机制`)
    }
  } finally { await sb.cleanup() }
})
