/**
 * 编程接口（`exports["./api"]`）。CLI 与 MCP server 都只走这一层，避免两份语义。
 *
 * 与 serving 的一致约定：
 *   - 自己 createRuntime()（路径 / 策略 / headless 判定）
 *   - 过 executeOperation() 的能力→风险→沙箱→确认→审计闸门
 *   - **永不抛异常**，只返回结构化结果（`safe()` 包装）
 *   - `confirm` 支持字符串或数组，第一参数与 options 里都认
 */
import { createRuntime, materializePolicies } from './config.mjs'
import { executeOperation, drain, inFlightCount } from './ops.mjs'
import { quantizeOperation } from './quantize.mjs'
import { validateExistingOperation } from './validate-existing.mjs'
import { withMatrix, findRun, summarizeRun, completeness, verifiedRuns } from './state/matrix.mjs'
import { readMetrics } from './metrics.mjs'
import { verifyAuditChain, listAuditFiles } from './audit.mjs'
import { sha256, shortId } from './util/hash.mjs'
import { getManifest } from './capability.mjs'
import { getSandboxProbe } from './sandbox.mjs'
import { probeProxy } from './netpolicy.mjs'
import { toErrorPayload, fail, ERROR_CODES } from './errors.mjs'
import { describeContract } from './testing/parity.mjs'
import { prepareDataset, verifyDatasetDir } from './dataset/prepare.mjs'
import { verifyImage, readImageMeta, buildImage } from './image.mjs'
import { buildQatPlan, authorizeQat } from './qat.mjs'
import { allMethods } from './backends/index.mjs'
import { probeEnvironment } from './probe/env.mjs'
import { assertOutputDirSafe } from './integrity.mjs'
import { promises as fs } from 'node:fs'
import { MCP_INTERFACE_VERSION, CLI_INTERFACE_VERSION, PACKAGE_VERSION } from './version.mjs'

async function bootstrap({ root, options = {}, env = process.env, probeNetwork = false } = {}) {
  const runtime = await createRuntime({ root, options, env })
  runtime.__packageRoot = new URL('../', import.meta.url).pathname
  if (options.materializePolicies !== false) {
    runtime.materializedPolicies = await materializePolicies(runtime.paths)
    runtime.materializedGates = await materializeGates(runtime)
  }
  if (probeNetwork) runtime.proxyProbe = await probeProxy(runtime)
  return runtime
}

/** gates.yaml：只生成"空模板"（两个字段都是 null），**不填默认值**（裁决 #2）。 */
async function materializeGates(runtime) {
  const { exists, atomicWrite } = await import('./util/fsx.mjs')
  const { stringify } = await import('./util/yaml.mjs')
  const { GATES_TEMPLATE } = await import('./policy-defaults.mjs')
  if (await exists(runtime.paths.gatesFile)) return null
  await atomicWrite(runtime.paths.gatesFile, stringify(GATES_TEMPLATE, {
    header: 'inference-quantization 门禁阈值模板 —— 必须显式填写，null 表示未设置（本插件会拒绝执行，不猜默认值）',
  }))
  return runtime.paths.gatesFile
}

function safe(fn) {
  return async (...args) => {
    try { return await fn(...args) } catch (err) { return { ...toErrorPayload(err), ok: false } }
  }
}

function resolveTokens(...sources) {
  const out = []
  for (const s of sources) {
    if (typeof s === 'string' && s.length > 0) out.push(s)
    else if (Array.isArray(s)) out.push(...s.filter((t) => typeof t === 'string' && t.length > 0))
  }
  return [...new Set(out)]
}

/**
 * 确认指纹：**必须排除记账字段**（confirm / options）。
 *
 * 踩过的坑：早期直接序列化整个 input，于是 `confirm` 被算进指纹 ——
 * 第一次调用拿到的令牌，带回重试时指纹已经变了，令牌永远不匹配，
 * 确认流程**永远走不通**，而错误信息还显示"需要确认"，看起来像用户没确认。
 * 指纹应当只由"这次要做什么"决定，与"怎么授权的"无关。
 */
function fingerprintOf(input) {
  if (typeof input === 'string') return `quantize:${sha256(input).slice(0, 16)}`
  const { confirm: _c, options: _o, ...rest } = (input ?? {})
  return `quantize:${sha256(JSON.stringify(rest)).slice(0, 16)}`
}

/* ------------------------------------------------------------------ quantize */

export const quantize = safe(async (input, {
  root, options = {}, confirm = null, signal = null, onPhase = null,
} = {}) => {
  const runtime = await bootstrap({ root, options })
  // 令牌四个位置都要认：第一参数的 confirm、input.options.confirm、
  // 第二参数的 confirm、第二参数 options.confirm。
  // 少认一个，就会出现"用户明明传了令牌却又被要一次确认"的鬼打墙。
  const tokens = resolveTokens(input?.confirm, input?.options?.confirm, options.confirm, confirm)
  // allowUnverifiedImage 可能来自第二参数的 options（CLI 就是这么传的），
  // 也可能来自 input.options —— 两种写法都要认，否则用户显式授权会被静默忽略。
  const allowUnverifiedImage = options.allowUnverifiedImage === true || input?.options?.allowUnverifiedImage === true
  // 门禁阈值同样可能在第二参数的 options 里（CLI 就是这么传的），也可能在 input 里，
  // 还可能用 snake_case 别名。三种写法都要认 —— 否则用户显式传的阈值会被静默忽略，
  // 而那恰好是本插件最不该发生的事（阈值缺失必须显式拒绝，而不是"看起来传了其实没传"）。
  const pick = (...vals) => {
    for (const v of vals) if (v !== undefined && v !== null && Number.isFinite(Number(v))) return Number(v)
    return undefined
  }
  const maxAccuracyDropPct = pick(input?.maxAccuracyDropPct, input?.max_accuracy_drop_pct,
    input?.options?.maxAccuracyDropPct, input?.options?.max_accuracy_drop_pct,
    options.maxAccuracyDropPct, options.max_accuracy_drop_pct)
  const minPerfGainPct = pick(input?.minPerfGainPct, input?.min_perf_gain_pct,
    input?.options?.minPerfGainPct, input?.options?.min_perf_gain_pct,
    options.minPerfGainPct, options.min_perf_gain_pct)
  const extra = {
    allowUnverifiedImage,
    ...(maxAccuracyDropPct !== undefined ? { maxAccuracyDropPct } : {}),
    ...(minPerfGainPct !== undefined ? { minPerfGainPct } : {}),
  }
  const raw = typeof input === 'object' && input !== null && !Array.isArray(input)
    ? { ...input, ...extra, options: { ...(input.options ?? {}), confirm: tokens, ...extra } }
    : { model: input, ...extra, options: { confirm: tokens, ...extra } }

  return executeOperation(runtime, {
    capability: 'quantize',
    resources: ['model:read', 'gpu:write', 'disk:write', 'state:write', 'network:egress'],
    action: 'quantize',
    targetFingerprint: fingerprintOf(input),
    confirmTokens: tokens,
    confirmSummary: `对 ${typeof input === 'string' ? input : (input?.model ?? input?.path ?? '(输入)')} 做量化：`
      + '会在容器内跑 PTQ（占用 GPU 数分钟）、下载校准数据、把产出写到独立目录，并跑完整精度/性能评测',
  }, async ({ log }) => {
    const out = await quantizeOperation(runtime, { raw, log, signal, onPhase })
    // 注意：quantizeOperation 在"需确认/需 QAT 裁决/失败"时**返回**结构而不是抛，
    // 因此这里原样透传，保持单一事实来源。
    return out
  })
})

/* ---------------------------------------------------------------- validate */

/**
 * 验收一个**已存在**的量化 checkpoint（M2 缺口的接口层补口）。
 *
 * 复用 `quantize` 的**同一套实现与阈值语义**（parityCheck / gate / verifyQuantOutput /
 * dataset prepare），但**不重新量化**：baseline = 量化前的原始模型，candidate = 已有产出。
 * 不写量化矩阵（那会改变既有 `list` / `get_report` 的行为），验收记录落在
 * `<runDir>/validate-existing/<id>.json` 并进审计链。
 */
export const validate = safe(async (input, {
  root, options = {}, confirm = null, signal = null, onPhase = null,
} = {}) => {
  const runtime = await bootstrap({ root, options })
  const tokens = resolveTokens(input?.confirm, input?.options?.confirm, options.confirm, confirm)
  const pick = (...vals) => {
    for (const v of vals) if (v !== undefined && v !== null && Number.isFinite(Number(v))) return Number(v)
    return undefined
  }
  const maxAccuracyDropPct = pick(input?.maxAccuracyDropPct, input?.max_accuracy_drop_pct,
    input?.options?.maxAccuracyDropPct, input?.options?.max_accuracy_drop_pct,
    options.maxAccuracyDropPct, options.max_accuracy_drop_pct)
  const minPerfGainPct = pick(input?.minPerfGainPct, input?.min_perf_gain_pct,
    input?.options?.minPerfGainPct, input?.options?.min_perf_gain_pct,
    options.minPerfGainPct, options.min_perf_gain_pct)
  const extra = {
    ...(maxAccuracyDropPct !== undefined ? { maxAccuracyDropPct } : {}),
    ...(minPerfGainPct !== undefined ? { minPerfGainPct } : {}),
  }
  const raw = (typeof input === 'object' && input !== null && !Array.isArray(input))
    ? { ...input, ...extra, options: { ...(input.options ?? {}), confirm: tokens, ...extra } }
    : { model: input, ...extra, options: { confirm: tokens, ...extra } }

  const baseRef = (typeof input === 'object' && input !== null) ? (input.baseline ?? input.model ?? input.path ?? '(输入)') : input
  const candidateRef = (typeof input === 'object' && input !== null) ? (input.candidate ?? '(candidate)') : '(candidate)'
  const summary = `验收已有量化 checkpoint：baseline=${baseRef} candidate=${candidateRef}。`
    + '会起容器跑完整精度评测（占用 GPU 数分钟），并写验收记录 + 审计。'

  return executeOperation(runtime, {
    capability: 'validate',
    resources: ['model:read', 'gpu:write', 'disk:write', 'state:write', 'network:egress'],
    action: 'validate-existing',
    targetFingerprint: fingerprintOf({ candidate: input?.candidate ?? null, baseline: input?.baseline ?? input?.model ?? input?.path ?? null }),
    confirmTokens: tokens,
    confirmSummary: summary,
  }, async ({ log }) => validateExistingOperation(runtime, { raw, log, signal, onPhase }))
})

/* ---------------------------------------------------------------- get_report */

export const getReport = safe(async ({ selector = null, runId = null, includeReport = true } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, {
    capability: 'get_report', resources: ['state:read'], action: 'get_report',
  }, async () => {
    const m = await withMatrix(runtime, (x) => JSON.parse(JSON.stringify(x)), { backup: false })
    const sel = runId ?? selector
    if (!sel) {
      return {
        ok: true, count: m.runs.length,
        runs: m.runs.map((r) => ({ ...summarizeRun(r), completeness: completeness(r) })),
        verified: verifiedRuns(m).length,
      }
    }
    const rec = findRun(m, sel)
    if (!rec) {
      fail(ERROR_CODES.E_RUN_NOT_FOUND, `找不到量化记录：${sel}`, {
        remediation: `可用选择器：runId、runKey、产出路径、源模型路径。当前矩阵有 ${m.runs.length} 条记录。`,
        details: { selector: sel, known: m.runs.slice(-10).map((r) => ({ id: r.id, sourceModel: r.sourceModel, status: r.status })) },
      })
    }
    let report = null
    if (includeReport && rec.reportPath) {
      const { readText } = await import('./util/fsx.mjs')
      const raw = await readText(rec.reportPath, null)
      if (raw) { try { report = JSON.parse(raw) } catch { report = null } }
    }
    return {
      ok: true, run: summarizeRun(rec), completeness: completeness(rec),
      record: rec, report, reportPath: rec.reportPath ?? null,
      qatPlan: report?.qatPlan ?? null,
    }
  })
})

/* ------------------------------------------------------------------- list */

export const list = safe(async ({ status = null, sourceModel = null, verifiedOnly = false } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, {
    capability: 'list', resources: ['state:read'], action: 'list',
  }, async () => {
    const m = await withMatrix(runtime, (x) => JSON.parse(JSON.stringify(x)), { backup: false })
    let runs = m.runs
    if (status) runs = runs.filter((r) => r.status === status)
    if (sourceModel) runs = runs.filter((r) => r.sourceModel === sourceModel)
    if (verifiedOnly) runs = runs.filter((r) => r.status === 'verified')
    return {
      ok: true, count: runs.length,
      matrixFile: runtime.paths.matrixFile,
      runs: runs.map((r) => ({ ...summarizeRun(r), completeness: completeness(r) })),
      byStatus: m.runs.reduce((acc, r) => { acc[r.status] = (acc[r.status] ?? 0) + 1; return acc }, {}),
      capabilities: { fiveElements: ['method', 'accuracy', 'performance', 'platform', 'status'] },
    }
  })
})

/* --------------------------------------------------------------- rollback */

export const rollback = safe(async ({ selector, purgeOutput = false, confirm: argConfirm = null } = {}, { root, options = {}, confirm = null } = {}) => {
  const runtime = await bootstrap({ root, options })
  // 令牌可以放在第一参数（`{selector, confirm}`）或第二参数（`{confirm}` / `options.confirm`）——
  // 三种写法必须等价，否则"带令牌重试"会被静默忽略、又抛一次一模一样的错（serving 踩过这个坑）。
  const tokens = resolveTokens(argConfirm, confirm, options.confirm)
  return executeOperation(runtime, {
    capability: 'rollback',
    resources: ['disk:write', 'state:write'],
    action: 'rollback',
    targetFingerprint: `rollback:${selector}`,
    confirmTokens: tokens,
    confirmSummary: `回退量化 run ${selector}：标记为已回退${purgeOutput ? '并删除产出目录' : '（保留产出目录）'}`,
  }, async ({ log }) => {
    const m = await withMatrix(runtime, (x) => JSON.parse(JSON.stringify(x)), { backup: false })
    const rec = findRun(m, selector)
    if (!rec) {
      fail(ERROR_CODES.E_NO_QUANTIZATION_TARGET, `找不到可回退的量化记录：${selector}`, {
        remediation: `可用选择器：runId / runKey / 产出路径 / 源模型路径。当前 ${m.runs.length} 条记录，可先执行 list。`,
        details: { selector, known: m.runs.slice(-10).map((r) => r.id) },
      })
    }
    const actions = []
    // 安全：只删本插件登记的产出目录，且必须与源模型目录不同
    if (purgeOutput && rec.outputPath) {
      try {
        assertOutputDirSafe(rec.sourceModel ?? '/', rec.outputPath)
        const st = await fs.stat(rec.outputPath).catch(() => null)
        if (st && st.isDirectory()) {
          await fs.rm(rec.outputPath, { recursive: true, force: true })
          actions.push({ action: 'purge-output', path: rec.outputPath, ok: true })
          log?.info('已删除量化产出目录', { path: rec.outputPath })
        } else {
          actions.push({ action: 'purge-output', path: rec.outputPath, ok: false, reason: '目录不存在' })
        }
      } catch (err) {
        actions.push({ action: 'purge-output', path: rec.outputPath, ok: false, reason: String(err.message) })
      }
    }
    const updated = await withMatrix(runtime, (x) => {
      const r = findRun(x, selector)
      const before = r.status
      r.status = 'rolled-back'
      r.rolledBackAt = new Date().toISOString()
      const { pushHistory } = require_pushHistory()
      pushHistory(r, 'rolled-back', { from: before, purgeOutput, actions })
      return JSON.parse(JSON.stringify(r))
    })
    const { bump } = await import('./metrics.mjs')
    await bump(runtime, 'rollback')
    return { ok: true, run: summarizeRun(updated), actions, sourceModelUntouched: true, note: '回退不触碰源模型（源模型只读 + 哈希校验）' }
  })
})

// 避免顶层循环依赖：延迟取 pushHistory
function require_pushHistory() {
  // eslint-disable-next-line global-require
  return { pushHistory: (run, event, detail) => { if (!Array.isArray(run.history)) run.history = []; run.history.push({ at: new Date().toISOString(), event, detail }) } }
}

/* -------------------------------------------------------------------- qat */

export const qat = safe(async ({ runId = null, selector = null, authorize = false, allowSandboxDegrade = false, confirm: argConfirm = null } = {}, { root, options = {}, confirm = null } = {}) => {
  const runtime = await bootstrap({ root, options })
  const tokens = resolveTokens(argConfirm, confirm, options.confirm)
  return executeOperation(runtime, {
    capability: 'qat_plan',
    resources: ['state:write', 'model:read'],
    action: authorize ? 'qat_authorize' : 'qat_plan',
    targetFingerprint: `qat:${runId ?? selector ?? 'latest'}`,
    confirmTokens: tokens,
    confirmSummary: '产出/读取 QAT 计划（不执行训练）；若 authorize=true 还会记录沙箱降级授权',
  }, async () => {
    const m = await withMatrix(runtime, (x) => JSON.parse(JSON.stringify(x)), { backup: false })
    const rec = findRun(m, runId ?? selector ?? m.runs.at(-1)?.id)
    if (!rec) {
      fail(ERROR_CODES.E_RUN_NOT_FOUND, '找不到可用于产出 QAT 计划的 run', {
        remediation: '先跑一次 quantize；QAT 计划建立在"哪些 PTQ 档位失败了"之上。',
        details: { runId, selector, runs: m.runs.length },
      })
    }
    const { readText } = await import('./util/fsx.mjs')
    const raw = rec.reportPath ? await readText(rec.reportPath, null) : null
    let report = null
    if (raw) { try { report = JSON.parse(raw) } catch { report = null } }
    let plan = report?.qatPlan ?? null
    if (!plan) {
      // 用记录里的信息重建一份计划（矩阵里有精度/性能证据）
      plan = await buildQatPlan(runtime, {
        standard: {
          source: { path: rec.sourceModel, input: rec.sourceModel },
          method: rec.method, dataset: { repo: runtime.options.datasetRepo, config: runtime.options.datasetConfig, split: runtime.options.datasetSplit },
          calibrationSamples: runtime.options.calibrationSamples, seed: runtime.options.calibrationSeed,
          maxSeqLen: 2048,
        },
        env: await probeEnvironment(runtime, { deep: false }),
        attempts: [{ level: rec.level, method: rec.method, failureCode: 'E_ACCURACY_GATE_FAILED', parity: { dropPct: rec.accuracy?.dropPct, samples: rec.accuracy?.samples } }],
        thresholds: { maxAccuracyDropPct: rec.accuracy?.thresholdPct ?? 0, minPerfGainPct: rec.performance?.thresholdPct ?? 0 },
      })
    }
    if (!authorize) return { ok: true, plan, approves: false, note: '这是计划，不是执行；QAT 需显式授权后才由你自行执行' }
    const auth = authorizeQat(runtime, { plan, confirmToken: tokens.find((t) => t.startsWith('qat_')), allowSandboxDegrade })
    return { ok: true, plan, authorization: auth }
  })
})

/* ------------------------------------------------------------------ 运维面 */

export const datasetPrepare = safe(async ({ repo = null, config = null, split = null, seed = null, calib = null, test = null, force = false } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, {
    capability: 'dataset_prepare',
    resources: ['network:egress', 'disk:write', 'state:write'],
    action: 'dataset_prepare',
    targetFingerprint: `dataset:${repo ?? runtime.options.datasetRepo}:${config ?? runtime.options.datasetConfig}`,
  }, async ({ log }) => {
    const out = await prepareDataset(runtime, {
      repo, config, split, seed,
      calibrationSamples: calib ?? undefined, testSamples: test ?? undefined, log, force,
    })
    return {
      ok: true, dir: out.dir, reused: out.reused,
      leakage: out.verification,
      lineage: out.lineage?.source ?? null,
      counts: out.verification.counts,
    }
  })
})

export const datasetVerify = safe(async ({ dir } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, {
    capability: 'get_report', resources: ['state:read'], action: 'dataset_verify',
  }, async () => {
    if (!dir) {
      fail(ERROR_CODES.E_INPUT_MISSING, '需要 dir（数据集目录）', {
        remediation: '用 `inference-quantization dataset list` 查看已准备的数据集目录。', details: null,
      })
    }
    const out = await verifyDatasetDir(runtime, dir)
    return { ok: true, dir, verified: out.verification.verified, verification: out.verification, lineage: out.lineage }
  })
})

export const datasetList = safe(async ({ root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, { capability: 'get_report', resources: ['state:read'], action: 'dataset_list' }, async () => {
    const { listDir } = await import('./util/fsx.mjs')
    const names = await listDir(runtime.paths.datasetDir)
    const out = []
    for (const n of names) {
      // eslint-disable-next-line no-await-in-loop
      const line = await fs.stat(`${runtime.paths.datasetDir}/${n}`).then((s) => s.isDirectory()).catch(() => false)
      if (!line) continue
      // eslint-disable-next-line no-await-in-loop
      const v = await verifyDatasetDir(runtime, `${runtime.paths.datasetDir}/${n}`).catch((e) => ({ verification: { verified: false, problems: [String(e.message)] } }))
      out.push({ dir: `${runtime.paths.datasetDir}/${n}`, verified: v.verification.verified, counts: v.verification.counts ?? null, problems: v.verification.problems ?? [] })
    }
    return { ok: true, root: runtime.paths.datasetDir, count: out.length, datasets: out }
  })
})

export const imageVerify = safe(async ({ image = null, build = false } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, {
    capability: 'image_verify',
    resources: ['network:egress', 'disk:write', 'state:write'],
    action: 'image_verify',
    targetFingerprint: `image:${image ?? runtime.options.quantImage}:${build}`,
  }, async ({ log }) => {
    const meta = await verifyImage(runtime, { image, log, build })
    return { ok: true, image: meta.image, verified: meta.verified, missing: meta.missing, scan: meta.scan, signature: meta.signature, toolchain: meta.toolchain, digest: meta.digest }
  })
})

export const imageBuild = safe(async ({ image = null, dockerfile = null } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, {
    capability: 'image_build',
    resources: ['network:egress', 'disk:write', 'state:write'],
    action: 'image_build',
    targetFingerprint: `image-build:${image ?? runtime.options.quantImage}`,
  }, async ({ log }) => {
    const out = await buildImage(runtime, { image, dockerfile: dockerfile ?? 'docker/Dockerfile.quant', log })
    const meta = await verifyImage(runtime, { image: out.image, log })
    return { ok: true, ...out, verified: meta.verified, missing: meta.missing }
  })
})

export const metrics = safe(async ({ root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, { capability: 'get_report', resources: ['state:read'], action: 'metrics' },
    async () => ({ ok: true, metrics: await readMetrics(runtime), file: runtime.paths.metricsFile }))
})

export const auditVerify = safe(async ({ date = null } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, { capability: 'get_report', resources: ['state:read'], action: 'audit_verify' },
    async () => ({
      ok: true, files: await listAuditFiles(runtime),
      verification: await verifyAuditChain(runtime, date ? new Date(date) : new Date()),
      auditDir: runtime.paths.auditDir,
    }))
})

export const doctor = safe(async ({ deep = false } = {}, { root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options, probeNetwork: true })
  return executeOperation(runtime, { capability: 'get_report', resources: ['state:read'], action: 'doctor' }, async () => {
    const env = await probeEnvironment(runtime, { deep })
    const sandbox = await getSandboxProbe()
    const manifest = getManifest()
    const imageMeta = await readImageMeta(runtime, runtime.options.quantImage)
    return {
      ok: true,
      version: { package: PACKAGE_VERSION, mcp: MCP_INTERFACE_VERSION, cli: CLI_INTERFACE_VERSION },
      manifest: { name: manifest.name, capabilities: [...manifest.capabilities.keys()], notProvided: [...manifest.notProvided.keys()] },
      methods: allMethods().map((m) => ({ id: m.id, level: m.level, bits: m.bits, lossy: m.lossy, available: m.isAvailable(env) })),
      headless: runtime.headless,
      policySources: runtime.policy.sources,
      gates: runtime.policy.gates ? { loaded: true } : { loaded: false, note: 'gates.yaml 未设置阈值 → 调用时必须显式传 max_accuracy_drop_pct / min_perf_gain_pct' },
      proxyProbe: runtime.proxyProbe ?? null,
      sandbox,
      image: imageMeta ? { image: imageMeta.image, verified: imageMeta.verified, missing: imageMeta.missing, digest: imageMeta.digest, toolchain: imageMeta.toolchain } : { image: runtime.options.quantImage, verified: false, note: '尚未验证：先跑 image verify' },
      testingCore: describeContract(),
      environment: {
        gpu: env.gpu.gpus, memory: env.memory, container: env.container, ports: env.ports,
        modelRoots: env.filesystem.modelRoots, capabilities: env.capability,
      },
      paths: runtime.paths,
      warnings: runtime.warnings.map((w) => (typeof w.toJSON === 'function' ? w.toJSON() : w)),
    }
  })
})

export const contract = safe(async ({ root, options = {} } = {}) => {
  const runtime = await bootstrap({ root, options })
  return executeOperation(runtime, { capability: 'get_report', resources: ['state:read'], action: 'contract' },
    async () => ({ ok: true, testingCore: describeContract(), mcp: { version: MCP_INTERFACE_VERSION }, cli: { version: CLI_INTERFACE_VERSION } }))
})

export { drain, inFlightCount, prepareDataset }
export const INTERFACE = { MCP: MCP_INTERFACE_VERSION, CLI: CLI_INTERFACE_VERSION, PACKAGE: PACKAGE_VERSION }
