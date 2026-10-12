/**
 * 输入处理流水线（量化版，覆盖题面的 7 类输入场景）。
 *
 *   原始输入 → normalize → validate → fill_defaults
 *            → detect_conflicts → resolve_ambiguity → confirm_if_needed → 标准化输入
 *
 * 与 serving 的三条共同不变量：
 *   A. 冲突**不静默修改**（检出即报，附可选方案）
 *   B. 假设**必留痕**（每个自动补的默认值进 assumptions[]）
 *   C. 多义**可选优但必须报告**（唯一高分自动选定 + 警告；并列且无法询问则中止）
 *
 * 量化特有的一条：**精度阈值不在本流水线补默认值**。
 * 缺阈值不是"参数缺失"，而是"拒绝执行"（见 gate.mjs 的裁决）。
 */
import { basename } from 'node:path'
import { fail, warn, ERROR_CODES, WARNING_CODES } from '../errors.mjs'
import { resolveModelRef, looksLikeCheckpoint, resolveRealPath, extractModelFacts } from '../probe/model.mjs'
import { probeEnvironment } from '../probe/env.mjs'
import { getMethod, listMethods, allMethods } from '../backends/index.mjs'
import { hashDirectory } from '../integrity.mjs'

const ENV_REF_RE = /^(?:env:|\$)([A-Za-z_][A-Za-z0-9_]*)$/
const URL_RE = /^https?:\/\//i
const PATHISH_RE = /^(?:[~.]?\/|\.\.?\/)/
const ORG_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/
const BARE_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/

/** 每次调用的覆盖项：顶层与 options.* 两种写法等价（serving 踩过"静默忽略"的坑）。 */
export function liftPerCallOptions(raw) {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return raw
  const merged = { ...(raw.options ?? {}), ...raw }
  merged.options = { ...(raw.options ?? {}) }
  return merged
}

export function normalizeInput(raw) {
  const out = []
  const push = (kind, value, note) => out.push({ kind, value, note: note ?? null })
  if (raw === null || raw === undefined) return out
  if (typeof raw === 'object' && !Array.isArray(raw)) {
    if (raw.env) push('env', String(raw.env))
    if (raw.path) push('local_path', String(raw.path))
    if (raw.url) push('url', String(raw.url))
    if (raw.model) {
      const v = String(raw.model)
      if (URL_RE.test(v)) push('url', v)
      else if (PATHISH_RE.test(v)) push('local_path', v)
      else if (ENV_REF_RE.test(v)) push('env', ENV_REF_RE.exec(v)[1])
      else if (ORG_NAME_RE.test(v)) push('model_id', v)
      else push('fuzzy', v)
    }
    if (raw.modelId) push('model_id', String(raw.modelId))
    if (raw.fuzzy || raw.description) push('fuzzy', String(raw.fuzzy ?? raw.description))
    return out
  }
  const s = String(raw).trim()
  if (s === '') return out
  const em = ENV_REF_RE.exec(s)
  if (em) { push('env', em[1]); return out }
  if (URL_RE.test(s)) { push('url', s); return out }
  if (PATHISH_RE.test(s)) { push('local_path', s); return out }
  if (ORG_NAME_RE.test(s)) { push('model_id', s); return out }
  if (BARE_NAME_RE.test(s)) { push('model_id', s); push('fuzzy', s); return out }
  push('fuzzy', s)
  return out
}

export function validateCandidates(candidates) {
  if (candidates.length === 0) {
    fail(ERROR_CODES.E_INPUT_MISSING, '没有提供任何模型输入', {
      remediation: '请提供：本地路径（/data/models/xxx）、模型 ID（org/name 或目录名）、env:VAR、或一句模糊描述（会列出候选）。',
      details: { accepted: ['path', 'model', 'modelId', 'env:VAR', 'fuzzyDescription'] },
    })
  }
  return true
}

/** 冲突检测：只报告，不改写。 */
export function detectConflicts(raw, env) {
  const c = []
  const o = (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) ? raw : {}
  if (o.path && (o.url || (o.model && URL_RE.test(String(o.model))))) {
    c.push({
      field: 'model', problem: '同时给了本地路径与远端地址',
      options: [{ id: 'use-path', description: '只用本地路径（离线、快）' }, { id: 'use-remote', description: '只用远端（会先下载）' }],
    })
  }
  if (o.method && !getMethod(o.method)) {
    c.push({
      field: 'method',
      problem: `未知量化方法 "${o.method}"；已注册：${listMethods().join(', ')}`,
      options: allMethods().map((m) => ({ id: m.id, description: `${m.displayName}（L${m.level}）` })),
    })
  }
  const drop = o.maxAccuracyDropPct ?? o.max_accuracy_drop_pct ?? o.options?.maxAccuracyDropPct
  if (drop !== undefined && drop !== null && Number(drop) < 0) {
    c.push({
      field: 'max_accuracy_drop_pct', problem: `精度容忍不能为负：${drop}`,
      options: [{ id: 'zero', description: '用 0（不容忍任何下降，通常不可达）' }, { id: 'positive', description: '给一个正数容忍度' }],
    })
  }
  const gain = o.minPerfGainPct ?? o.min_perf_gain_pct ?? o.options?.minPerfGainPct
  if (gain !== undefined && gain !== null && Number(gain) < 0) {
    c.push({
      field: 'min_perf_gain_pct', problem: `性能收益目标不能为负：${gain}`,
      options: [{ id: 'zero', description: '用 0（不要求收益，只要求不变慢）' }, { id: 'positive', description: '给一个正数目标' }],
    })
  }
  return c
}

/** 补默认值（**不含精度阈值**）。 */
export async function fillDefaults(runtime, input, env, { sourcePath, sourceDisplay }) {
  const assumptions = []
  const assume = (f, v, r) => { assumptions.push({ field: f, value: v, reason: r }); return v }

  let methodId = input.method ?? null
  if (!methodId) {
    const available = allMethods().filter((m) => m.isAvailable(env))
    if (available.length === 0) {
      fail(ERROR_CODES.E_METHOD_UNAVAILABLE, '本机没有可用的量化方法（需要 GPU + 容器 + nvidia runtime）', {
        remediation: `环境探测：gpu=${env.gpu.present} docker=${env.container.available} nvidiaRuntime=${env.container.nvidiaRuntime}`,
        details: { env: env.capability },
      })
    }
    methodId = assume('method', runtime.options.defaultMethod, '默认档（题面：默认 AWQ + INT4）')
  }
  const method = getMethod(methodId)
  if (!method) {
    fail(ERROR_CODES.E_METHOD_UNKNOWN, `未知量化方法：${methodId}`, {
      remediation: `已注册方法：${listMethods().join(', ')}。新方法通过 src/backends/registry.mjs 的 register() 插件式加入。`,
      details: { requested: methodId, available: listMethods() },
    })
  }

  const outputRoot = input.outputRoot ?? runtime.options.outputRoot ?? runtime.paths.outputDir
  const outName = input.outputName ?? assume('outputName', `${sourceDisplay}-${methodId}`, '源模型名 + 方法 id（保证与源不同名）')
  const outputPath = `${outputRoot.replace(/\/+$/, '')}/${outName}`

  const calibrationSamples = input.calibrationSamples ?? runtime.options.calibrationSamples
  if (input.calibrationSamples === undefined) assume('calibrationSamples', calibrationSamples, '默认校准样本数（够用且快）')
  const seed = input.calibrationSeed ?? runtime.options.calibrationSeed
  if (input.calibrationSeed === undefined) assume('calibrationSeed', seed, '固定种子 → 切分可复现')

  return {
    methodId, method, outputPath, calibrationSamples, seed,
    dataset: {
      repo: input.datasetRepo ?? runtime.options.datasetRepo,
      config: input.datasetConfig ?? runtime.options.datasetConfig,
      split: input.datasetSplit ?? runtime.options.datasetSplit,
      testSamples: input.testSamples ?? 400,
    },
    thresholds: {
      // 只从显式参数与策略文件取；不在这里补默认（gate.mjs 会拒绝）
      maxAccuracyDropPct: input.maxAccuracyDropPct ?? input.max_accuracy_drop_pct ?? null,
      minPerfGainPct: input.minPerfGainPct ?? input.min_perf_gain_pct ?? null,
    },
    perf: {
      workload: input.workload ?? 'chat-medium',
      requests: input.requests ?? 8,
      concurrency: input.concurrency ?? 2,
    },
    assumptions,
  }
}

/**
 * 全流水线：返回标准化输入 + 环境 + 源模型信息（含哈希）。
 * 抛出的每个错误都带 remediation。
 */
export async function runInputPipeline(runtime, raw, { log = null, skipHash = false } = {}) {
  raw = liftPerCallOptions(raw)
  const candidates = normalizeInput(raw)
  validateCandidates(candidates)
  const env = await probeEnvironment(runtime, { deep: false })

  const attempts = []
  let resolved = null
  for (const cand of candidates) {
    try {
      // eslint-disable-next-line no-await-in-loop
      const r = await resolveModelRef(runtime, cand)
      attempts.push({ candidate: cand, status: r.status })
      if (r.status === 'resolved') { resolved = { ...r, matchedCandidate: cand }; break }
      if (r.status === 'ambiguous') {
        const list = r.candidates.map((c) => `  - ${c.path}（匹配度 ${c.score}）`).join('\n')
        if (runtime.headless) {
          const top = r.candidates[0]
          runtime.warnings.push(warn(WARNING_CODES.W_AMBIGUITY_RESOLVED, `headless 下按最高匹配度自动选定：${top.path}`,
            { query: r.display, chosen: top.path }))
          // eslint-disable-next-line no-await-in-loop
          resolved = { ...(await resolveModelRef(runtime, { kind: 'local_path', value: top.path })), matchedCandidate: cand }
          break
        }
        fail(ERROR_CODES.E_INPUT_AMBIGUOUS, `输入 "${r.display}" 匹配到多个模型`, {
          remediation: `请指定完整路径：\n${list}`, details: { query: r.display, candidates: r.candidates.slice(0, 8) },
        })
      }
    } catch (err) {
      attempts.push({ candidate: cand, status: 'error', code: err.code ?? null, message: err.message })
      if (candidates.length === 1) throw err
    }
  }
  if (!resolved) {
    fail(ERROR_CODES.E_MODEL_UNRESOLVED, `无法解析为可量化的模型：${JSON.stringify(raw)}`, {
      remediation: `已尝试：${attempts.map((a) => `${a.candidate.kind}:${a.candidate.value}→${a.status}`).join('；')}`,
      details: { attempts, modelRoots: runtime.options.modelRoots },
    })
  }
  if (resolved.needsDownload) {
    fail(ERROR_CODES.E_MODEL_UNRESOLVED, '量化需要本地 checkpoint，不支持直接从远端 URL 量化', {
      remediation: '先把模型下到本地模型根目录（如 /data/models），再对本地路径量化。',
      details: { downloadPlan: resolved.downloadPlan },
    })
  }

  const input = (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) ? raw : {}
  const conflicts = detectConflicts(input, env)
  if (conflicts.length > 0) {
    fail(ERROR_CODES.E_INPUT_CONFLICT, `输入存在 ${conflicts.length} 处冲突，未做任何静默修改`, {
      remediation: conflicts.map((c) => `[${c.field}] ${c.problem} → 可选：${c.options.map((o) => o.id).join(' | ')}`).join('\n'),
      details: { conflicts },
    })
  }

  const filled = await fillDefaults(runtime, input, env, {
    sourcePath: resolved.path, sourceDisplay: resolved.display,
  })

  // 源模型完整性：**在一切写操作之前**算基线哈希
  let sourceIntegrity = null
  if (!skipHash) {
    log?.phase('source-hash', { path: resolved.path })
    const h = await hashDirectory(resolved.path, { log })
    sourceIntegrity = { dir: resolved.path, before: h, algorithm: 'sha256(relpath\\0filehash 排序拼接)' }
    log?.info('源模型基线哈希', { hash: h.hash.slice(0, 16), files: h.files, bytes: h.bytes })
  }

  const standard = {
    source: {
      input: resolved.display, path: resolved.path, kind: resolved.kind,
      wasSymlink: resolved.wasSymlink === true, facts: resolved.facts, license: resolved.license ?? null,
    },
    method: filled.methodId,
    methodSpec: filled.method.describe(env),
    // 镜像可信度放宽：显式授权才放行（会记入审计）。两种写法都认（顶层 / options.*）
    allowUnverifiedImage: (input.allowUnverifiedImage ?? input.options?.allowUnverifiedImage) === true,
    outputPath: filled.outputPath,
    calibrationSamples: filled.calibrationSamples,
    seed: filled.seed,
    dataset: filled.dataset,
    thresholds: filled.thresholds,
    perf: filled.perf,
    assumptions: filled.assumptions,
    sourceIntegrity,
    env: {
      gpu: env.gpu.gpus?.[0]?.name ?? null,
      unifiedMemory: env.capability.unifiedMemory,
      availableMemoryBytes: env.capability.availableMemoryBytes,
      docker: env.container.available,
      nvidiaRuntime: env.container.nvidiaRuntime,
      sandboxTiers: env.capability.sandboxTiers,
      image: runtime.options.quantImage,
    },
  }
  return { standard, env, resolved, attempts }
}

export { looksLikeCheckpoint, resolveRealPath, extractModelFacts, basename }
