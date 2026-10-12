/**
 * 集成测试：完整量化闭环（用假执行器，但产出真实文件、走真实生产代码路径）。
 *
 * 覆盖：去泄漏门禁、源模型完整性、输出目录安全、方法预检、产物校验、
 * 精度/性能双门禁、降级链、矩阵写入、报告落盘、指标。
 *
 * 不需要 GPU / Docker —— 这正是"执行器接缝"存在的意义，也是"不只靠 mock"的落点：
 * 假执行器写的是真文件，闭环里其余每一步都是生产代码。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { makeSandbox } from '../helpers/sandbox.mjs'
import { knobs, resetKnobs } from '../helpers/fake-executor.mjs'
import { createRuntime } from '../../src/config.mjs'
import { quantizeOperation } from '../../src/quantize.mjs'
import { loadMatrix, findRun } from '../../src/state/matrix.mjs'
import { readAudit, verifyAuditChain } from '../../src/audit.mjs'
import { promises as fs } from 'node:fs'
import { join } from 'node:path'

async function setup({ gates = null, imageVerified = true } = {}) {
  resetKnobs()
  const sb = await makeSandbox()
  const runtime = await createRuntime({
    root: sb.root,
    options: { ...sb.options, executor: 'fake', quantImage: 'fake/image:1', allowUnverifiedImage: !imageVerified },
  })
  runtime.__packageRoot = new URL('../../', import.meta.url).pathname
  if (gates) runtime.policy.gates = gates
  if (imageVerified) {
    // 造一份"已验证"的镜像元数据（否则量化的第一道闸门就会拒绝）
    const { atomicWriteJson, mkdirp } = await import('../../src/util/fsx.mjs')
    await mkdirp(runtime.paths.imageMetaDir)
    await atomicWriteJson(join(runtime.paths.imageMetaDir, 'fake_image_1.json'), {
      image: 'fake/image:1', digest: 'fake/image@sha256:' + 'a'.repeat(64), verified: true, missing: [],
      scan: { scanned: true, scanner: 'test', gate: 'pass' }, signature: { signed: true }, toolchain: { ok: true, llmcompressor: 'fake' },
    })
  }
  return { sb, runtime }
}

const raw = (sb, extra = {}) => ({ model: sb.modelPath, ...extra })

test('闭环: 正常量化 —— 精度/性能双门禁通过、矩阵标记 verified、报告落盘', async () => {
  // L1(awq-w4a16) 的伪精度损失 = 12%，阈值给 15% 让它一档通过
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 15, min_perf_gain_pct: 20 } })
  try {
    const out = await quantizeOperation(runtime, { raw: raw(sb) })
    assert.equal(out.ok, true, `应签收：${JSON.stringify(out).slice(0, 300)}`)
    assert.equal(out.signed, true)
    assert.equal(out.report.outcome, 'gated')
    assert.equal(out.report.gates.decision, 'sign')
    assert.equal(out.report.degraded, 0, '默认档就应通过')

    // 产出真实文件且带量化配置
    assert.ok(out.report.output)
    const cfg = JSON.parse(await fs.readFile(join(out.report.output, 'config.json'), 'utf8'))
    assert.equal(cfg.quantization_config.quant_method, 'compressed-tensors')
    assert.equal(cfg.quantization_config.config_groups.group_0.weights.num_bits, 4)

    // 矩阵：五要素齐全、状态 verified
    const m = await loadMatrix(runtime)
    assert.equal(m.runs.length, 1)
    const rec = m.runs[0]
    assert.equal(rec.status, 'verified')
    assert.equal(rec.method, 'awq-w4a16')
    assert.ok(rec.accuracy.dropPct !== undefined)
    assert.ok(rec.performance.decodeTpsGainPct !== undefined)
    assert.ok(rec.platform.gpu)
    for (const k of ['method', 'accuracy', 'performance', 'platform', 'status']) {
      assert.ok(rec[k] !== undefined && rec[k] !== null, `五要素缺 ${k}`)
    }
    // 报告文件确实落盘
    assert.ok(await fs.stat(out.report.reportPath).then((s) => s.isFile()))
  } finally { await sb.cleanup() }
})

test('闭环: 去泄漏门禁是硬门禁 —— 泄漏时中止且不进入量化', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 5, min_perf_gain_pct: 1 } })
  try {
    // 在数据集准备后把测试集污染成与校准集相交（模拟容器产出被篡改）
    const { itemHash } = await import('../../src/dataset/leakage.mjs')
    const origPrepare = knobs.prepareDatasetFails
    void origPrepare
    const { fakeExecutor, knobs: k } = await import('../helpers/fake-executor.mjs')
    const orig = fakeExecutor.prepareDataset
    fakeExecutor.prepareDataset = async (rt, args) => {
      const r = await orig(rt, args)
      // 污染：把测试集第一条改成校准集第一条的内容
      const calib = (await fs.readFile(join(args.outDir, 'calib.jsonl'), 'utf8')).split('\n').filter(Boolean)
      const first = JSON.parse(calib[0])
      await fs.writeFile(join(args.outDir, 'test.jsonl'), JSON.stringify({ h: first.h, t: first.t }) + '\n')
      void itemHash; void k
      return r
    }
    try {
      const err = await quantizeOperation(runtime, { raw: raw(sb) }).catch((e) => e)
      assert.equal(err.code, 'E_CALIB_LEAKAGE_DETECTED', `应被去泄漏门禁拦下，实际 ${err.code}: ${err.message}`)
      // 不该有任何量化产出
      assert.equal(knobs.calls.filter((c) => c.op === 'quantize').length, 0, '泄漏时不得进入量化')
      const m = await loadMatrix(runtime)
      assert.equal(m.runs.filter((r) => r.status === 'verified').length, 0)
    } finally { fakeExecutor.prepareDataset = orig }
  } finally { await sb.cleanup() }
})

test('闭环: 源模型被改 → 哈希校验失败并中止（原模型只读是硬约束）', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 5, min_perf_gain_pct: 1 } })
  try {
    const { fakeExecutor } = await import('../helpers/fake-executor.mjs')
    const orig = fakeExecutor.quantize
    // 模拟"量化过程误写了源目录"
    fakeExecutor.quantize = async (rt, args) => {
      await fs.writeFile(join(args.sourcePath, 'model.safetensors'), Buffer.alloc(64, 9))
      return orig(rt, args)
    }
    try {
      const err = await quantizeOperation(runtime, { raw: raw(sb) }).catch((e) => e)
      assert.equal(err.code, 'E_SOURCE_MODEL_MODIFIED')
      assert.match(err.remediation, /只读/)
    } finally { fakeExecutor.quantize = orig }
  } finally { await sb.cleanup() }
})

test('闭环: 输出目录位于源目录内部 → 一票否决（第一道防线）', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 5, min_perf_gain_pct: 1 } })
  try {
    const err = await quantizeOperation(runtime, {
      raw: raw(sb, { outputRoot: sb.modelPath, outputName: 'sub' }),
    }).catch((e) => e)
    assert.equal(err.code, 'E_OUTPUT_DIR_UNSAFE')
  } finally { await sb.cleanup() }
})

test('闭环: 降级链 —— L1 精度不达标自动降到 L2 并成功（验收专项第 3 条）', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 5, min_perf_gain_pct: 20 } })
  try {
    // L1(awq-w4a16) 的伪困惑度退化 = 12% > 5% → 必须降级
    const out = await quantizeOperation(runtime, { raw: raw(sb) })
    assert.equal(out.ok, true, `应在 L2 成功：${JSON.stringify(out).slice(0, 400)}`)
    assert.equal(out.report.degraded, 1, '应降级 1 档')
    assert.equal(out.report.method, 'int8-w8a8')
    assert.equal(out.report.attempts.length, 2)
    assert.equal(out.report.attempts[0].failureCode, 'E_ACCURACY_GATE_FAILED')
    assert.match(out.report.chain, /L1:awq-w4a16\(E_ACCURACY_GATE_FAILED\) → L2:int8-w8a8/)
    // 矩阵里也该记录降级轨迹
    const m = await loadMatrix(runtime)
    assert.equal(m.runs[0].degrade.steps, 1)
    assert.equal(m.runs[0].level, 2)
  } finally { await sb.cleanup() }
})

test('闭环: 全部档位不达标 → 产出 QAT 计划并停下（不执行训练）', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 0.5, min_perf_gain_pct: 1 } })
  try {
    const out = await quantizeOperation(runtime, { raw: raw(sb) })
    assert.equal(out.ok, false)
    assert.equal(out.code, 'E_QAT_REQUIRES_APPROVAL')
    const plan = out.details.qatPlan
    assert.ok(plan, '必须给出 QAT 计划')
    // 计划四要素（用户裁决 #1：数据/配置/预期收益/回退路径）
    for (const k of ['data', 'config', 'expectedGain', 'rollbackPath']) {
      assert.ok(plan[k], `QAT 计划缺 ${k}`)
    }
    assert.equal(plan.executes, false, '计划不得声称会执行')
    assert.equal(plan.approval.cacheable, false, 'QAT 授权不得缓存')
    assert.ok(plan.risks.some((r) => r.id === 'sandbox'))
    assert.match(plan.expectedGain.caveat, /经验估计/)
    const m = await loadMatrix(runtime)
    assert.equal(m.runs[0].status, 'qat-plan-required')
  } finally { await sb.cleanup() }
})

test('闭环: 量化本身失败也会降级；不可降级失败立即中止', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 15, min_perf_gain_pct: 20 } })
  try {
    knobs.quantizeFailLevels = [1, 2]
    const out = await quantizeOperation(runtime, { raw: raw(sb) })
    assert.equal(out.ok, true)
    assert.equal(out.report.method, 'fp8-dynamic', 'L1/L2 失败后应在 L3 成功')
    assert.equal(out.report.degraded, 2)

    // 换成不可降级失败：只尝试一次
    // 注意换 seed —— 否则会命中已准备的数据集目录（复用路径），注入的失败根本不会发生
    resetKnobs()
    knobs.prepareDatasetFails = true
    const err = await quantizeOperation(runtime, { raw: raw(sb, { outputName: 'x2', calibrationSeed: 777 }) }).catch((e) => e)
    assert.equal(err.code, 'E_DATASET_UNAVAILABLE')
    resetKnobs()
  } finally { await sb.cleanup() }
})

test('闭环: 产出缺 quantization_config → 判失败（"跑完了"不等于"量化了"）', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 15, min_perf_gain_pct: 20 } })
  try {
    knobs.degradeOutput = true
    // 四档的产出都不合格 → 降级链走完 → 落到 QAT 计划（不是抛错，而是"停下等裁决"）
    const out = await quantizeOperation(runtime, { raw: raw(sb) }).catch((e) => e)
    const code = out.code ?? out.message
    assert.ok(code === 'E_QAT_REQUIRES_APPROVAL' || code === 'E_DEGRADE_EXHAUSTED', `实际 ${code}`)
    assert.equal(out.ok, false)
    assert.equal(out.report.attempts.length, 4, '四档都该被试过')
    assert.ok(out.report.attempts.every((a) => a.failureCode === 'E_QUANTIZE_FAILED'))
    const m = await loadMatrix(runtime)
    assert.equal(m.runs.filter((r) => r.status === 'verified').length, 0)
  } finally { await sb.cleanup() }
})

test('闭环: 阈值缺失 → 拒绝执行（不猜默认值）', async () => {
  const { sb, runtime } = await setup({ gates: null })
  try {
    const err = await quantizeOperation(runtime, { raw: raw(sb) }).catch((e) => e)
    assert.equal(err.code, 'E_THRESHOLD_NOT_PROVIDED')
    assert.equal(knobs.calls.filter((c) => c.op === 'quantize').length, 0, '阈值缺失时不该做任何量化')
  } finally { await sb.cleanup() }
})

test('闭环: 镜像未验证 → 拒绝；显式授权则放行并留痕', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 15, min_perf_gain_pct: 20 }, imageVerified: false })
  try {
    const err = await quantizeOperation(runtime, { raw: raw(sb) }).catch((e) => e)
    assert.equal(err.code, 'E_IMAGE_NOT_VERIFIED')
    // 显式授权
    const out = await quantizeOperation(runtime, { raw: raw(sb, { allowUnverifiedImage: true }) })
    assert.equal(out.ok, true)
  } finally { await sb.cleanup() }
})

test('闭环: 指标与报告反映真实过程', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 5, min_perf_gain_pct: 20 } })
  try {
    const out = await quantizeOperation(runtime, { raw: raw(sb) })
    const { readMetrics } = await import('../../src/metrics.mjs')
    const m = await readMetrics(runtime)
    assert.ok(m.counters.quantize_attempts >= 2, '两次尝试（L1 失败 + L2 成功）')
    assert.ok(m.counters.degrade_steps >= 1)
    assert.equal(m.counters.gate_pass, 1)
    assert.ok(m.timings.quantize_ms_total > 0)
    // 报告里保留全部尝试的失败码，便于复盘
    assert.ok(out.report.attempts.every((a) => a.method))
    assert.equal(out.report.leakage.verified, true)
    assert.equal(out.report.source.unchanged, true)
    assert.equal(out.report.testingCore.implementation.includes('internal'), true)
  } finally { await sb.cleanup() }
})

test('闭环: 同源+同方法+同切分 → 同一 runKey（幂等标识）', async () => {
  const { sb, runtime } = await setup({ gates: { max_accuracy_drop_pct: 15, min_perf_gain_pct: 20 } })
  try {
    const a = await quantizeOperation(runtime, { raw: raw(sb) })
    const b = await quantizeOperation(runtime, { raw: raw(sb, { outputName: 'second' }) })
    assert.equal(a.report.runKey, b.report.runKey, 'runKey 只由源+方法+切分决定')
  } finally { await sb.cleanup() }
})
