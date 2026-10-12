/**
 * 集成测试：`validate-existing`（M2 的"验收已有 checkpoint"入口）完整闭环。
 *
 * 用**该包自己的执行器接缝**（`options.executor='fake'`，见 `src/executors/index.mjs` 的设计说明）：
 * 假执行器写**真实文件**、给**确定性数值**，而闭环里其余每一步（产出形状校验、阈值解析、
 * 数据集与去泄漏、parity、测量成立性、精度门禁、记录落盘）走的都是**生产代码**。
 *
 * 为什么这条测试重要：M2 的入口在真机上要起容器 + 占 GPU。若它的**接线与门禁**只在真机才能验，
 * 那每次改动都要赌一次 GPU 窗口。这条测试把"接线 + 门禁语义"钉在无 GPU 的 CI 里，
 * 真机只负责回答"serving/容器能不能加载"这一个问题。
 *
 * 覆盖：
 *   ① 候选形状（缺 quantization_config ⇒ E_QUANTIZE_FAILED）
 *   ② scheme 从候选读回（`8bit/channel`）且 `schemeRecognized=true`
 *   ③ 精度达标 ⇒ `sign`；不达标 ⇒ `degrade`（两类结论可区分）
 *   ④ 阈值缺失 ⇒ E_THRESHOLD_NOT_PROVIDED（不猜默认值）
 *   ⑤ `gate='accuracy+perf'` ⇒ 与 quantize 同一次 `evaluateGates`
 *   ⑥ baseline == candidate ⇒ E_INPUT_CONFLICT
 *   ⑦ 验收记录落盘在 `<runDir>/validate-existing/<id>.json`，且**不写量化矩阵**
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import { join } from 'node:path'

import { makeSandbox } from '../helpers/sandbox.mjs'
import { knobs, resetKnobs } from '../helpers/fake-executor.mjs'
import { safetensorsBuffer, minimalModelTensors } from '../helpers/safetensors.mjs'
import { createRuntime } from '../../src/config.mjs'
import { validateExistingOperation } from '../../src/validate-existing.mjs'
import { ERROR_CODES } from '../../src/errors.mjs'

/** 造一个"已量化"的候选目录：真文件 + 真 quantization_config（compressed-tensors 方言）。 */
async function makeCandidate(base, { name = 'qat-int8', numBits = 8, strategy = 'channel', withQc = true, method = 'qat-int8' } = {}) {
  const dir = join(base, name)
  await fs.mkdir(dir, { recursive: true })
  const cfg = { architectures: ['LlamaForCausalLM'], model_type: 'llama', max_position_embeddings: 4096, num_hidden_layers: 2, hidden_size: 64, torch_dtype: 'bfloat16' }
  if (withQc) {
    cfg.quantization_config = {
      quant_method: 'compressed-tensors', format: 'int-quantized', version: '0.19.0',
      config_groups: {
        group_0: {
          targets: ['Linear'], format: 'int-quantized',
          weights: { num_bits: numBits, type: 'int', symmetric: true, strategy, dynamic: false, group_size: null },
          input_activations: { num_bits: 8, type: 'int', symmetric: true, strategy: 'token', dynamic: true, group_size: null },
        },
      },
      ignore: ['lm_head'],
    }
  }
  await fs.writeFile(join(dir, 'config.json'), JSON.stringify(cfg))
  await fs.writeFile(join(dir, 'model.safetensors'), safetensorsBuffer(minimalModelTensors()))
  // 假执行器靠这个文件知道"这份产出是哪个方法造的"，从而给出确定性的 dropPct
  if (method) await fs.writeFile(join(dir, '.fake-method'), method)
  return dir
}

async function setup({ imageVerified = true } = {}) {
  resetKnobs()
  const sb = await makeSandbox()
  const runtime = await createRuntime({
    root: sb.root,
    options: { ...sb.options, executor: 'fake', quantImage: 'fake/image:1', allowUnverifiedImage: !imageVerified },
  })
  runtime.__packageRoot = new URL('../../', import.meta.url).pathname
  if (imageVerified) {
    const { atomicWriteJson, mkdirp } = await import('../../src/util/fsx.mjs')
    await mkdirp(runtime.paths.imageMetaDir)
    await atomicWriteJson(join(runtime.paths.imageMetaDir, 'fake_image_1.json'), {
      image: 'fake/image:1', digest: 'fake/image@sha256:' + 'a'.repeat(64), verified: true, missing: [],
      scan: { scanned: true, scanner: 'test', gate: 'pass' }, signature: { signed: true }, toolchain: { ok: true, llmcompressor: 'fake' },
    })
  }
  return { sb, runtime }
}

test('validate-existing: 精度达标 ⇒ sign，scheme 从候选读回，记录落盘且不写矩阵', async () => {
  const { sb, runtime } = await setup()
  try {
    knobs.dropByMethod = { 'qat-int8': 0.8 }        // QAT 的伪精度损失 0.8% < 阈值 2%
    knobs.gainByMethod = { 'qat-int8': 5 }
    const cand = await makeCandidate(sb.base)
    const out = await validateExistingOperation(runtime, {
      raw: { model: sb.modelPath, candidate: cand, max_accuracy_drop_pct: 2 },
    })

    assert.equal(out.gateMode, 'accuracy')
    assert.equal(out.candidate.scheme, '8bit/channel', 'scheme 必须从候选产出读回')
    assert.equal(out.candidate.schemeRecognized, true)
    assert.equal(out.candidate.weightBits, undefined, '不重复造字段（明细在 quantization_config 之外只给 scheme）')
    assert.equal(out.quantization_config, undefined, '记录里不内联整份 quantization_config（避免把配置当结论）')
    assert.equal(out.measurement.valid, true, '测量必须成立')
    assert.equal(out.gate.pass, true)
    assert.equal(out.gate.decision, 'sign')
    assert.ok(Math.abs(out.parity.dropPct - 0.8) < 1e-9, `dropPct=0.8（实际 ${out.parity.dropPct}）`)
    assert.equal(out.ok, true)

    // 记录落盘 + 形状
    const rec = JSON.parse(await fs.readFile(out.recordPath, 'utf8'))
    assert.equal(rec.schema, 'inference-quantization/validate-existing@1')
    assert.equal(rec.candidate.scheme, '8bit/channel')
    assert.equal(rec.gate.pass, true)
    assert.ok(out.recordPath.includes('/validate-existing/'), '记录落在 runDir/validate-existing/ 下')

    // **不写量化矩阵**（那会改变既有 list / get_report 的行为）
    assert.equal(await sb.readMatrixText(), null, 'validate-existing 不写量化矩阵')
  } finally { await sb.cleanup() }
})

test('validate-existing: 精度不达标 ⇒ degrade（不是 undetermined）', async () => {
  const { sb, runtime } = await setup()
  try {
    knobs.dropByMethod = { 'qat-bad': 4.5 }        // 4.5% > 阈值 2%
    const cand = await makeCandidate(sb.base, { name: 'qat-bad', method: 'qat-bad' })
    const out = await validateExistingOperation(runtime, {
      raw: { model: sb.modelPath, candidate: cand, max_accuracy_drop_pct: 2 },
    })
    assert.equal(out.gate.pass, false)
    assert.equal(out.gate.decision, 'degrade', '数值未达标 ⇒ degrade；与"测量不成立"可区分')
    assert.equal(out.measurement.valid, true)
  } finally { await sb.cleanup() }
})

test('validate-existing: 候选缺 quantization_config ⇒ E_QUANTIZE_FAILED', async () => {
  const { sb, runtime } = await setup()
  try {
    const cand = await makeCandidate(sb.base, { name: 'float-only', withQc: false })
    await assert.rejects(
      () => validateExistingOperation(runtime, { raw: { model: sb.modelPath, candidate: cand, max_accuracy_drop_pct: 2 } }),
      (e) => e.code === ERROR_CODES.E_QUANTIZE_FAILED,
      '只输出 float 的目录不得被验收为量化产出（与 quantize.verifyQuantOutput 同判据）',
    )
  } finally { await sb.cleanup() }
})

test('validate-existing: 阈值必须显式给出（不猜默认值）', async () => {
  const { sb, runtime } = await setup()
  try {
    const cand = await makeCandidate(sb.base)
    await assert.rejects(
      () => validateExistingOperation(runtime, { raw: { model: sb.modelPath, candidate: cand } }),
      (e) => e.code === ERROR_CODES.E_THRESHOLD_NOT_PROVIDED,
      '缺 max_accuracy_drop_pct ⇒ 拒绝执行',
    )
  } finally { await sb.cleanup() }
})

test('validate-existing: baseline == candidate ⇒ E_INPUT_CONFLICT（验收没有对象）', async () => {
  const { sb, runtime } = await setup()
  try {
    const cand = await makeCandidate(sb.base)
    await assert.rejects(
      () => validateExistingOperation(runtime, { raw: { model: cand, candidate: cand, max_accuracy_drop_pct: 2 } }),
      (e) => e.code === ERROR_CODES.E_INPUT_CONFLICT,
    )
  } finally { await sb.cleanup() }
})

test('validate-existing: gate=accuracy+perf ⇒ 与 quantize 同一次 evaluateGates（两阈值都要）', async () => {
  const { sb, runtime } = await setup()
  try {
    knobs.dropByMethod = { 'qat-int8': 0.8 }
    knobs.gainByMethod = { 'qat-int8': 30 }
    const cand = await makeCandidate(sb.base)
    const out = await validateExistingOperation(runtime, {
      raw: { model: sb.modelPath, candidate: cand, max_accuracy_drop_pct: 2, min_perf_gain_pct: 20, gate: 'accuracy+perf' },
    })
    assert.equal(out.gateMode, 'accuracy+perf')
    assert.equal(out.gate.pass, true)
    assert.equal(out.gate.decision, 'sign')
    assert.ok(out.gate.perf && typeof out.gate.perf.gainPct === 'number', '双门禁档必须给出性能数')
    assert.ok(out.perf && out.perf.gainDecodePct === 30)
    assert.equal(out.gate.thresholds.minPerfGainPct, 20)
  } finally { await sb.cleanup() }
})

test('validate-existing: 精度接近上限 ⇒ confirm-near-limit（不是"直接签收"）', async () => {
  const { sb, runtime } = await setup()
  try {
    knobs.dropByMethod = { 'qat-near': 1.9 }       // 1.9 / 2.0 = 95% ≥ 90% nearLimitFactor
    const cand = await makeCandidate(sb.base, { name: 'qat-near', method: 'qat-near' })
    const out = await validateExistingOperation(runtime, {
      raw: { model: sb.modelPath, candidate: cand, max_accuracy_drop_pct: 2 },
    })
    assert.equal(out.gate.decision, 'confirm-near-limit')
    assert.equal(out.gate.needsConfirmation, true)
    assert.equal(out.gate.signable, false)
  } finally { await sb.cleanup() }
})
