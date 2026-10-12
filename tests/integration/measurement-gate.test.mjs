/**
 * 集成测试：**"测量是否成立"真的挡在签收之前**。
 *
 * 这一组对应真机事故（2026-09-23）的闭环复现：
 *   一次"精度零损失"被签收，事后证明源模型残缺、模型输出均匀分布、
 *   基线与候选逐位相同 —— 数值没错，测量毫无意义。
 *
 * 因此这里不只测纯函数，而是**跑完整闭环**，断言：
 *   ① 退化测量 → 运行以 E_MEASUREMENT_INVALID 终止，且**不**落 verified
 *   ② 残缺源模型 → 在**量化之前**就失败（不浪费 GPU），且不调用量化执行器
 *   ③ 健康路径不受影响（该签收的仍然签收）
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { makeSandbox } from '../helpers/sandbox.mjs'
import { knobs, resetKnobs } from '../helpers/fake-executor.mjs'
import { safetensorsBuffer, minimalModelTensors } from '../helpers/safetensors.mjs'
import { createRuntime } from '../../src/config.mjs'
import { quantizeOperation } from '../../src/quantize.mjs'
import { loadMatrix } from '../../src/state/matrix.mjs'
import { promises as fs } from 'node:fs'
import { join } from 'node:path'

async function setup({ gates = { max_accuracy_drop_pct: 15, min_perf_gain_pct: 20 } } = {}) {
  resetKnobs()
  const sb = await makeSandbox()
  const runtime = await createRuntime({
    root: sb.root,
    options: { ...sb.options, executor: 'fake', quantImage: 'fake/image:1', allowUnverifiedImage: true },
  })
  runtime.__packageRoot = new URL('../../', import.meta.url).pathname
  runtime.policy.gates = gates
  // 造一份"已验证"的镜像元数据（否则第一道闸门就拒绝，测不到后面要测的东西）
  const { atomicWriteJson, mkdirp } = await import('../../src/util/fsx.mjs')
  await mkdirp(runtime.paths.imageMetaDir)
  await atomicWriteJson(join(runtime.paths.imageMetaDir, 'fake_image_1.json'), {
    image: 'fake/image:1', digest: 'fake/image@sha256:' + 'a'.repeat(64), verified: true, missing: [],
    scan: { scanned: true, scanner: 'test', gate: 'pass' }, signature: { signed: true }, toolchain: { ok: true, llmcompressor: 'fake' },
  })
  return { sb, runtime }
}

test('测量门禁: 基线与候选逐位相同 → E_MEASUREMENT_INVALID，且矩阵不落 verified', async () => {
  const { sb, runtime } = await setup()
  knobs.degenerateIndiscriminate = true
  try {
    await assert.rejects(
      () => quantizeOperation(runtime, { raw: { model: sb.modelPath } }),
      (e) => {
        assert.equal(e.code, 'E_MEASUREMENT_INVALID', `实际 ${e.code}: ${e.message}`)
        assert.match(e.message, /无法通过更换量化方法解决/)
        return true
      },
    )
    const m = await loadMatrix(runtime)
    assert.ok(m && m.runs.length === 1, '应写入了运行记录（失败也要留痕）')
    assert.equal(m.runs[0].status, 'failed')
    assert.equal(m.runs.filter((r) => r.status === 'verified').length, 0, '退化测量绝不能被标为 verified')
    // 测量不成立时不得留下任何精度结论（避免"dropPct: 0"被误读成无损）
    assert.equal(m.runs[0].accuracy, null, '测量不成立时不应写入精度结论')
    // 也不该白跑性能评测
    assert.equal(knobs.calls.some((c) => c.op === 'perf'), false, '测量不成立时不应继续测性能')
  } finally { await sb.cleanup() }
})

test('测量门禁: 逐条 meanNll 无方差 → E_MEASUREMENT_INVALID（与输入无关的数不算测量）', async () => {
  const { sb, runtime } = await setup()
  knobs.degenerateItemVariance = true
  try {
    await assert.rejects(
      () => quantizeOperation(runtime, { raw: { model: sb.modelPath } }),
      (e) => e.code === 'E_MEASUREMENT_INVALID',
    )
    const m = await loadMatrix(runtime)
    assert.equal(m.runs[0].status, 'failed')
  } finally { await sb.cleanup() }
})

test('测量门禁: 报告里必须给出可复核的成立性证据（reasons + checks）', async () => {
  const { sb, runtime } = await setup()
  knobs.degenerateIndiscriminate = true
  try {
    const err = await quantizeOperation(runtime, { raw: { model: sb.modelPath } }).then(() => null, (e) => e)
    assert.ok(err, '应失败')
    const att = err.details.attempts[0]
    assert.equal(att.measurementValid, false)
    assert.ok(att.measurementReasons.includes('indiscriminate'), JSON.stringify(att.measurementReasons))
    // 报告也确实落盘了（失败路径也要可复核）
    assert.ok(err.details.reportPath)
    const report = JSON.parse(await fs.readFile(err.details.reportPath, 'utf8'))
    assert.equal(report.measurementValidity.valid, false)
    assert.equal(report.measurementValidity.perAttempt[0].valid, false)
    assert.ok(report.measurementValidity.perAttempt[0].checks.bitIdenticalMeanNll === true)
    assert.ok(report.measurementValidity.sourceValidation.tensors > 0, '报告里同时带上源模型预检证据')
  } finally { await sb.cleanup() }
})

test('源模型预检: 残缺模型在量化之前就失败 —— 不调用量化执行器（不浪费 GPU）', async () => {
  const { sb, runtime } = await setup()
  try {
    // 复刻事故形态：文件长度正确，但数据区后半段全为 0x00
    const p = join(sb.modelPath, 'model.safetensors')
    const buf = await fs.readFile(p)
    const headerLen = Number(buf.readBigUInt64LE(0))
    buf.fill(0, 8 + headerLen + 64)
    await fs.writeFile(p, buf)

    await assert.rejects(
      () => quantizeOperation(runtime, { raw: { model: sb.modelPath } }),
      (e) => {
        assert.equal(e.code, 'E_SOURCE_MODEL_INVALID')
        assert.match(e.remediation, /重新完整下载/)
        return true
      },
    )
    assert.equal(knobs.calls.some((c) => c.op === 'quantize'), false, '绝不能在残缺模型上真跑量化')
    assert.equal(knobs.calls.some((c) => c.op === 'prepareDataset'), false, '预检排在数据集准备之前，坏输入要最便宜地失败')
  } finally { await sb.cleanup() }
})

test('源模型预检: 单看"文件没被改过"是过不了的 —— 哈希通过但结构残缺仍必须拦下', async () => {
  const { sb, runtime } = await setup()
  try {
    const p = join(sb.modelPath, 'model.safetensors')
    const buf = await fs.readFile(p)
    const headerLen = Number(buf.readBigUInt64LE(0))
    buf.fill(0, 8 + headerLen + 64)
    await fs.writeFile(p, buf)

    const err = await quantizeOperation(runtime, { raw: { model: sb.modelPath } }).then(() => null, (e) => e)
    assert.ok(err)
    // 这正是当初的盲区：目录哈希（内容是否被改动）全程都会通过，因为没有人改它 ——
    // 它一开始就是残缺的。结构预检必须独立回答第二个问题。
    assert.equal(err.code, 'E_SOURCE_MODEL_INVALID')
    const details = err.details
    assert.ok(details.problems.some((x) => x.kind === 'zero-filled'), JSON.stringify(details.problems))
  } finally { await sb.cleanup() }
})

test('源模型预检: 正常模型不受影响，且报告里留下预检证据', async () => {
  const { sb, runtime } = await setup()
  try {
    const out = await quantizeOperation(runtime, { raw: { model: sb.modelPath } })
    assert.equal(out.ok, true, JSON.stringify(out).slice(0, 300))
    assert.equal(out.report.source.validation.ok, true)
    assert.ok(out.report.source.validation.evidence.totalTensors > 0)
    assert.equal(out.report.source.validation.evidence.totalZeroTensors, 0)
    // 哈希校验（内容未变）与结构预检（是完整模型）两条证据并列存在
    assert.equal(out.report.source.unchanged, true)
  } finally { await sb.cleanup() }
})

test('测量门禁: 健康路径仍按数值判 —— 测量成立时矩阵记录 accuracy.measurementValid=true', async () => {
  const { sb, runtime } = await setup()
  try {
    const out = await quantizeOperation(runtime, { raw: { model: sb.modelPath } })
    assert.equal(out.signed, true)
    const m = await loadMatrix(runtime)
    assert.equal(m.runs[0].status, 'verified')
    assert.equal(m.runs[0].accuracy.measurementValid, true, '矩阵里的精度结论必须带成立性证据')
    assert.ok(m.runs[0].accuracy.baselineMeanNll !== m.runs[0].accuracy.candidateMeanNll,
      '基线与候选的 meanNll 必须不同 —— 这正是旧事故里缺的那个区分力')
    assert.equal(out.report.measurementValidity.valid, true)
  } finally { await sb.cleanup() }
})
