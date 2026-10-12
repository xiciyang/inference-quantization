/**
 * validate-existing —— **验收一个已存在的量化 checkpoint**（M2 缺口）。
 *
 * ## 为什么需要它（缺口原文）
 *
 * `quantize.mjs` 的门禁**恒**以 `baselineModelPath = 源模型` /
 * `candidateModelPath = 本次新量化产出` 对比（`quantize.mjs:204-207`）。
 * 于是任何**外部**产出的量化 checkpoint（例如 `training-qat` 的 QAT 产出）都无法被单独验收：
 * 唯一的门禁入口会先**再量化一次**，对已量化的产出构成二次量化。
 * 闭环在量化侧断开 —— 本文件就是那一环。
 *
 * ## 边界（**本文件刻意只做接口层**）
 *
 * - **不改** `quantize.mjs` / `gate.mjs` / `testing/parity.mjs` 一行；
 * - **复用同一套实现**：精度对比 = `parityCheck`（`testing/parity.mjs`，它的执行器就是
 *   `quantize` 用的那个）；测量成立性 = `gate.assertMeasurementValid`；
 *  精度门禁 = `gate.evaluateAccuracyGate`；双门禁 = `gate.evaluateGates`；
 *  阈值解析 = `gate.resolveThresholds`（**同一条"显式 > gates.yaml > 拒绝"链，不新增默认值**）；
 *  产出形状校验 = `quantize.verifyQuantOutput`（**同一个函数**，同一条"没有
 *   `quantization_config` 就是没真的量化"判据）；数据集与去泄漏 = `dataset/prepare`（同一入口）。
 * - **不写量化矩阵**：`quantization-matrix.yaml` 是 `quantize` run 的名册。把验收记录塞进去
 *   会改变既有 `list` / `get_report` 的语义（即"改变既有工具行为"），因此验收记录写在
 *   `<runDir>/validate-existing/<id>.json` 并进审计链。
 * - **两种判定档，语义都来自既有代码**：
 *     `gate: 'accuracy'`（默认）—— 只用精度维：`evaluateAccuracyGate` + `assertMeasurementValid`，
 *       决策阶梯与 `evaluateGates` 对精度维**逐字一致**（`sign` / `confirm-near-limit` /
 *       `degrade` / `undetermined`），`nearLimitFactor` 取 `gate.mjs` 的默认 0.9。
 *       为什么默认只判精度：验收一个**已有的**量化产物时，性能是产出方（PTQ 档位 / QAT）的事，
 *       而"精度是否被这个 checkpoint 保住"才是被验收的东西；且 `resolveThresholds` 要求
 *       `min_perf_gain_pct` 也非空，强制它会把纯精度验收逼成一个假选项。
 *     `gate: 'accuracy+perf'` —— 走 `evaluateGates`，**与 `quantize` 完全同一次调用**，
 *       需要两个阈值都给出。
 *
 * 诚实性：返回里的 `scheme` 一律**从候选产出自身读回**（`verifyQuantOutput`），
 * 并给出 `schemeRecognized`（`config_groups.group_0.weights.{num_bits,strategy}` 是否可识别）。
 * 本入口不声明任何它没有读到的方案。
 */
import { existsSync, promises as fs } from 'node:fs'
import { join } from 'node:path'
import { fail, ERROR_CODES, warn, WARNING_CODES } from './errors.mjs'
import { runInputPipeline } from './input/pipeline.mjs'
import { prepareDataset } from './dataset/prepare.mjs'
import { assertNoLeakage } from './dataset/leakage.mjs'
import { assertSandbox } from './sandbox.mjs'
import { assertImageVerified } from './image.mjs'
import { resolveThresholds, evaluateAccuracyGate, assertMeasurementValid, evaluateGates } from './gate.mjs'
import { parityCheck } from './testing/parity.mjs'
import { verifyQuantOutput } from './quantize.mjs'
import { shortId } from './util/hash.mjs'
import { mkdirp } from './util/fsx.mjs'
import { assertOutputDirSafe } from './integrity.mjs'

/** 候选必须是一个**已存在的本地模型目录**（本入口不接受需要下载的引用）。 */
async function resolveCandidate(raw, runtime) {
  const cand = raw?.candidate ?? raw?.checkpoint ?? raw?.verify ?? null
  if (!cand || typeof cand !== 'string') {
    fail(ERROR_CODES.E_INPUT_MISSING, 'validate-existing 需要 candidate（一个已存在的量化 checkpoint 目录）', {
      remediation: '传 candidate=<本地绝对路径>；本入口只验收已有的本地目录，不解析模型 ID / URL。',
      details: { accepted: ['candidate'], got: raw?.candidate ?? null },
    })
  }
  const path = String(cand)
  if (!existsSync(path)) {
    fail(ERROR_CODES.E_MODEL_UNRESOLVED, `候选目录不存在：${path}`, {
      remediation: '确认路径；本入口不下载、不生成候选 —— 它只验收已有产物。', details: { candidate: path },
    })
  }
  const stat = await fs.stat(path)
  if (!stat.isDirectory()) {
    fail(ERROR_CODES.E_INPUT_INVALID, `候选不是目录：${path}`, { details: { candidate: path } })
  }
  // 形状校验 = **quantize 用的同一个函数**（同一个"没真的量化就是失败"判据）
  const shape = await verifyQuantOutput(path, null)
  if (!shape.ok) {
    fail(ERROR_CODES.E_QUANTIZE_FAILED, `候选不是可验收的量化产出：${shape.reason}`, {
      remediation: '候选目录必须有 config.json（内含 quantization_config）与 *.safetensors|*.bin 权重。'
        + '只输出 float 权重的目录**不能**被验收为量化产出。',
      details: { candidate: path, reason: shape.reason },
    })
  }
  const srcPath = raw?.baseline ?? raw?.model ?? raw?.path ?? null
  if (srcPath && String(srcPath) === path) {
    fail(ERROR_CODES.E_INPUT_CONFLICT, 'baseline 与 candidate 是同一个目录：验收没有对象', {
      remediation: 'baseline 必须是**量化前**的原始模型，candidate 是待验收的量化产出。',
      details: { baseline: srcPath, candidate: path },
    })
  }
  assertOutputDirSafe(String(srcPath ?? ''), path)
  return { path, shape, realPath: await fs.realpath(path) }
}

/**
 * **精度维门禁**（纯函数 —— 为可穷举测试而单独取出）。
 *
 * 与 `gate.evaluateGates` 的精度维**逐字同构**：
 *   - 同 `evaluateAccuracyGate`（含同一个默认 `nearLimitFactor = 0.9`）；
 *   - 同一条决策阶梯：测量不成立 / 未测得 ⇒ `undetermined`；否则
 *     `sign`（通过且未近上限）/ `confirm-near-limit`（通过但接近上限）/ `degrade`（不通过）；
 *   - 同一条"测量不成立即不可签收"的原则（`pass = validityOk && accuracy.passed`）。
 *
 * 为什么需要它：`evaluateGates` 要求**性能维也有数**（`perf` 为 null 时性能门禁落到
 * `undetermined`，整个判定跟着变成 `undetermined`）。验收一个**已有的**量化产物时，
 * 性能是产出方的事，强行要求它会把纯精度验收逼成一个假选项。
 * 因此这里只把**精度维**单独取出来，并且保证与 `evaluateGates` 对精度维的处置一致
 * —— 由 `tests/unit/validate-existing.test.mjs` 逐点对照断言，不靠"看着一样"。
 *
 * @returns {{pass, decision, signable, needsConfirmation, measurementValid, measurement, accuracy, thresholds, reasons}}
 */
export function accuracyOnlyGate({ dropPct, thresholds, measurement = null, nearLimitFactor = 0.9 }) {
  const acc = evaluateAccuracyGate({ dropPct, thresholdPct: thresholds.maxAccuracyDropPct, nearLimitFactor })
  const measurementInvalid = Boolean(measurement && measurement.valid === false)
  const undetermined = measurementInvalid || dropPct === null || dropPct === undefined
  const decision = undetermined ? 'undetermined' : (acc.passed ? (acc.nearLimit ? 'confirm-near-limit' : 'sign') : 'degrade')
  const reasons = []
  if (measurementInvalid) for (const r of measurement.reasons ?? []) reasons.push(`测量：${r.message}`)
  if (dropPct === null || dropPct === undefined) reasons.push('精度：dropPct 未测得（E_ACCURACY_GATE_MISSING）')
  reasons.push(acc.reason)
  return {
    pass: !undetermined && acc.passed === true,
    decision,
    signable: decision === 'sign',
    needsConfirmation: decision === 'confirm-near-limit',
    measurementValid: measurement ? measurement.valid !== false : null,
    measurement: measurement ?? null,
    accuracy: acc,
    thresholds,
    reasons,
  }
}

/**
 * 验收已有 checkpoint。
 * @returns {{ok, id, scheme, schemeRecognized, parity, measurement, gate, thresholds, ...}}
 */
export async function validateExistingOperation(runtime, { raw, log = null, signal = null, onPhase = null } = {}) {
  const startedAt = Date.now()
  const phase = (name, data) => { log?.phase?.(name, data); onPhase?.({ name, ...(data ?? {}), at: Date.now() - startedAt }) }

  // ---------- 0. 候选（已存在的量化产出） ----------
  phase('candidate')
  const cand = await resolveCandidate(raw, runtime)
  log?.info('候选量化产出已解析', { candidate: cand.path, scheme: cand.shape.scheme, weightFiles: cand.shape.weightFiles })

  // ---------- 1. 基线（量化前的原始模型）----------
  phase('baseline')
  // `baseline` 是本入口的**别名**，但输入流水线只认 `model` / `path` / `url` / `env`。
  // 不做这次映射的话，传 `baseline=` 会被判成"没有输入"并抛 E_INPUT_MISSING
  // （本机实测踩到：跨插件对账第一次就是这么失败的）。映射是**单向**的：只补 `model`，
  // 不覆盖调用方已给的 `model`/`path`，避免把显式输入改掉。
  const baselineRaw = (() => {
    if (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) {
      const { candidate: _c, checkpoint: _k, gate: _g, baseline, ...rest } = raw
      const hasPrimary = rest.model !== undefined || rest.path !== undefined || rest.url !== undefined || rest.env !== undefined
      return hasPrimary || baseline === undefined ? rest : { ...rest, model: baseline }
    }
    return raw
  })()
  if (!baselineRaw || (typeof baselineRaw === 'object' && baselineRaw.model === undefined && baselineRaw.path === undefined && baselineRaw.url === undefined && baselineRaw.env === undefined)) {
    fail(ERROR_CODES.E_INPUT_MISSING, 'validate-existing 需要 baseline（量化前的原始模型）', {
      remediation: '传 baseline=<本地路径>（或 model / path，二者等价）。baseline 是**量化前**的模型，作为精度基线。',
      details: { accepted: ['baseline', 'model', 'path'], got: raw?.baseline ?? raw?.model ?? null },
    })
  }
  const { standard, env } = await runInputPipeline(runtime, baselineRaw, { log })

  // ---------- 2. 沙箱 + 镜像（与 quantize 同一条前置链）----------
  phase('sandbox')
  await assertSandbox(runtime, runtime.policy.risk.sandbox?.R2 ?? 'container')
  phase('image')
  await assertImageVerified(runtime, {
    image: runtime.options.quantImage,
    allowUnverified: standard.allowUnverifiedImage === true,
  })

  // ---------- 3. 阈值（**不猜默认值**；同一条解析链）----------
  phase('thresholds')
  const mode = (raw?.gate ?? 'accuracy') === 'accuracy+perf' ? 'accuracy+perf' : 'accuracy'
  let thresholds
  if (mode === 'accuracy+perf') {
    thresholds = resolveThresholds(runtime, {
      maxAccuracyDropPct: standard.thresholds.maxAccuracyDropPct,
      minPerfGainPct: standard.thresholds.minPerfGainPct,
    })
  } else {
    // 精度维只需要精度阈值；性能阈值不参与 ⇒ 不能借用 resolveThresholds（它要求两者都非空）
    const acc = standard.thresholds.maxAccuracyDropPct
    const from = (acc !== null && acc !== undefined && Number.isFinite(Number(acc)))
      ? { value: Number(acc), from: 'caller' }
      : (() => {
        const v = runtime?.policy?.gates?.max_accuracy_drop_pct
        return (v !== undefined && v !== null && Number.isFinite(Number(v))) ? { value: Number(v), from: 'gates.yaml' } : { value: null, from: null }
      })()
    if (from.value === null) {
      fail(ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, '缺少门禁阈值：max_accuracy_drop_pct（精度容忍）', {
        remediation: '在调用里传 max_accuracy_drop_pct，或把它写进 '
          + `${runtime?.paths?.gatesFile ?? '~/.ai-infra/config/gates.yaml'}。`
          + '本插件**不会**替你猜一个默认阈值 —— 那会让"精度门禁独立"失效。',
        details: { missing: ['max_accuracy_drop_pct'], gate: mode },
      })
    }
    if (from.value < 0) fail(ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, `max_accuracy_drop_pct 不能为负：${from.value}`, { details: { value: from.value } })
    thresholds = { maxAccuracyDropPct: from.value, minPerfGainPct: null, source: { maxAccuracyDropPct: from.from, minPerfGainPct: null } }
  }

  // ---------- 4. 测试集（同一条去泄漏门禁）----------
  phase('dataset')
  const ds = await prepareDataset(runtime, {
    repo: standard.dataset.repo, config: standard.dataset.config, split: standard.dataset.split,
    seed: standard.seed, calibrationSamples: standard.calibrationSamples, testSamples: standard.dataset.testSamples,
    log,
  })
  assertNoLeakage(ds.verification)
  const testPath = join(ds.dir, 'test.jsonl')

  const id = `qv_${shortId(`${standard.source.path}|${cand.path}|${Date.now()}`, 12)}`
  const samples = Number.isFinite(Number(raw?.samples)) ? Number(raw.samples) : 0

  // ---------- 5. 精度对比（**同一个 parityCheck**）----------
  phase('parity')
  const parity = await parityCheck(runtime, {
    testSetPath: testPath,
    baselineModelPath: standard.source.path,
    candidateModelPath: cand.path,
    limit: samples,
    label: 'validate-existing',
    log,
  })
  const measurement = assertMeasurementValid({ baseline: parity.baseline, candidate: parity.candidate })

  // ---------- 6. 门禁 ----------
  let gate
  let perf = null
  if (mode === 'accuracy+perf') {
    phase('perf')
    const { executorFor } = await import('./executors/index.mjs')
    await import('./executors/docker.mjs')
    perf = await executorFor(runtime).perf(runtime, {
      baselineModelPath: standard.source.path, candidateModelPath: cand.path,
      workload: standard.perf.workload, requests: standard.perf.requests, concurrency: standard.perf.concurrency, log,
    })
    if (!perf?.ok) {
      fail(ERROR_CODES.E_SERVING_UNAVAILABLE, `性能评测失败：${perf?.error ?? perf?.code ?? 'unknown'}`, {
        remediation: '双门禁档需要真实端点。先跑 inference-serving doctor，或改用 gate="accuracy" 只判精度。',
        details: perf ?? null,
      })
    }
    const g = evaluateGates({
      accuracy: { dropPct: parity.drop.pct },
      perf: { gainPct: perf.gain.decodeTpsPct },
      thresholds,
      measurement,
    })
    // 形状归一：`evaluateGates` 的字段是 `passed`，而本入口的另一档（accuracy）用 `pass`。
    // 两个档位必须给出**同一个**判定字段，否则调用方要按档位分支读 —— 那就是接口分叉。
    gate = { ...g, pass: g.passed === true, dropPct: parity.drop.pct, gainPct: perf.gain.decodeTpsPct }
  } else {
    gate = accuracyOnlyGate({ dropPct: parity.drop.pct, thresholds, measurement })
  }

  // ---------- 7. 记账 ----------
  const schemeRecognized = Boolean(cand.shape.scheme)
  if (!schemeRecognized) {
    runtime.warnings.push(warn(WARNING_CODES.W_MEASUREMENT_VALIDITY,
      `候选产出的 quantization_config 里没有 config_groups.group_0.weights.{num_bits,strategy}：`
      + `本插件无法识别它的 scheme（quantize.mjs:71 的读取方式）`,
      { candidate: cand.path, quantizationMethod: cand.shape.quantMethod ?? null }))
  }

  const record = {
    schema: 'inference-quantization/validate-existing@1',
    id,
    at: new Date().toISOString(),
    durationMs: Date.now() - startedAt,
    gateMode: mode,
    baseline: { path: standard.source.path, integrity: standard.sourceIntegrity?.before?.hash ?? null },
    candidate: {
      path: cand.path, realPath: cand.realPath,
      scheme: cand.shape.scheme, schemeRecognized,
      quantMethod: cand.shape.quantMethod, format: cand.shape.format, version: cand.shape.version,
      weightFiles: cand.shape.weightFiles, bytes: cand.shape.bytes,
      compressionRatio: cand.shape.compressionRatio,
    },
    thresholds,
    dataset: { dir: ds.dir, testPath, testDigest: ds.verification?.testListDigest ?? null, leakage: ds.verification },
    parity: {
      metric: parity.metric,
      dropPct: parity.drop.pct, nllDelta: parity.drop.nllDelta, samples: parity.samples,
      baseline: { perplexity: parity.baseline.perplexity, meanNll: parity.baseline.meanNll, samples: parity.baseline.scoredSamples },
      candidate: { perplexity: parity.candidate.perplexity, meanNll: parity.candidate.meanNll, samples: parity.candidate.scoredSamples },
    },
    perf: perf ? { gainDecodePct: perf.gain.decodeTpsPct, gainThroughputPct: perf.gain.throughputPct, cleanupOk: perf.cleanup?.ok ?? null } : null,
    measurement: { valid: measurement.valid, reasons: measurement.reasons, checks: measurement.checks },
    gate,
    evidence: {
      parityContractVersion: parity.evidence?.contractVersion ?? null,
      implementation: 'inference-quantization/validate-existing（复用 parityCheck / gate / verifyQuantOutput）',
      image: standard.env?.image ?? null,
      gpu: standard.env?.gpu ?? null,
    },
  }
  const outDir = join(runtime.paths.runDir, 'validate-existing')
  await mkdirp(outDir)
  const recordPath = join(outDir, `${id}.json`)
  await fs.writeFile(recordPath, `${JSON.stringify(record, null, 2)}\n`)
  log?.info('验收记录已写出', { recordPath, pass: gate.pass, dropPct: parity.drop.pct, scheme: cand.shape.scheme })

  return { ...record, recordPath, ok: true }
}
