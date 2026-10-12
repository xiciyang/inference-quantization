/**
 * 量化矩阵 `~/.ai-infra/state/quantization-matrix.yaml`（题面 §共享状态）。
 *
 * 与 serving 的端点注册表同构（文件锁 + 原子写 + 写前快照），但语义不同：
 * 一条记录 = **一次量化实验**（一个 source model × 一种方法 × 一档精度），
 * 记录 方法 / 精度 / 性能 / 平台 / 状态 五要素（量化专项验收第 4 条）。
 *
 * 关键设计：**矩阵同时是"证据账本"**。去泄漏血缘、源模型哈希、门禁数值都写在这里，
 * 使得"原模型未被修改""校准集与测试集无交集"这两条可以在事后被独立复核，
 * 而不是只能相信当时的日志。
 */
import { promises as fs } from 'node:fs'
import { stringify, parse } from '../util/yaml.mjs'
import { atomicWrite, readText, mkdirp, listDir } from '../util/fsx.mjs'
import { withLock } from '../util/lock.mjs'
import { MATRIX_FORMAT_VERSION, PACKAGE_VERSION } from '../version.mjs'
import { fail, ERROR_CODES } from '../errors.mjs'

export function emptyMatrix() {
  return { version: MATRIX_FORMAT_VERSION, updatedAt: new Date().toISOString(), pkg: PACKAGE_VERSION, runs: [] }
}

export function parseMatrix(text) {
  if (!text || text.trim() === '') return emptyMatrix()
  let doc
  try {
    doc = parse(text)
  } catch (err) {
    fail(ERROR_CODES.E_INTERNAL, `量化矩阵解析失败：${err.message}`, {
      remediation: '矩阵已损坏。可用 `inference-quantization list --raw` 查看原文，或从 state/history/ 取备份。',
      details: { parseError: String(err.message) }, cause: err,
    })
  }
  if (!doc) return emptyMatrix()
  const version = doc.version ?? 1
  if (version > MATRIX_FORMAT_VERSION) {
    fail(ERROR_CODES.E_INTERNAL, `量化矩阵格式版本 ${version} 高于本插件支持的 ${MATRIX_FORMAT_VERSION}`, {
      remediation: '请升级 inference-quantization；本插件拒绝以旧格式覆盖新格式（防止写坏）。',
      details: { found: version, supported: MATRIX_FORMAT_VERSION },
    })
  }
  return { version, updatedAt: doc.updatedAt ?? null, pkg: doc.pkg ?? null, runs: Array.isArray(doc.runs) ? doc.runs : [] }
}

export function serializeMatrix(m) {
  return stringify(
    { ...m, updatedAt: new Date().toISOString(), pkg: PACKAGE_VERSION },
    { header: 'inference-quantization 量化矩阵 —— 由插件维护；手工编辑请保持受限 YAML 子集（字符串需双引号）' },
  )
}

export async function loadMatrix(runtime) {
  return parseMatrix(await readText(runtime.paths.matrixFile, null))
}

export async function saveMatrix(runtime, m) {
  await mkdirp(runtime.paths.stateDir)
  await atomicWrite(runtime.paths.matrixFile, serializeMatrix(m))
  return m
}

/** 读-改-写事务（与 serving 同一套锁/原子写/快照机制）。 */
export async function withMatrix(runtime, fn, { backup = true } = {}) {
  return withLock(runtime.paths.lockDir, 'matrix', async () => {
    const before = await readText(runtime.paths.matrixFile, null)
    const m = parseMatrix(before)
    if (backup && before) await snapshot(runtime, before)
    const result = await fn(m)
    await saveMatrix(runtime, m)
    return result
  })
}

async function snapshot(runtime, text) {
  try {
    await mkdirp(runtime.paths.historyDir)
    const name = `${new Date().toISOString().replace(/[:.]/g, '-')}.matrix.yaml`
    await atomicWrite(`${runtime.paths.historyDir}/${name}`, text)
    const names = (await listDir(runtime.paths.historyDir)).filter((n) => n.endsWith('.matrix.yaml')).sort()
    for (const old of names.slice(0, Math.max(0, names.length - 20))) {
      await fs.rm(`${runtime.paths.historyDir}/${old}`, { force: true })
    }
  } catch { /* 快照是便利设施，不是安全属性（安全属性是审计链） */ }
}

/* --------------------------------------------------------------- 查询辅助 */

export function findRun(m, selector) {
  if (!selector) return null
  const s = String(selector).trim()
  return m.runs.find((r) => r.id === s)
    ?? m.runs.find((r) => r.runKey === s)
    ?? m.runs.find((r) => r.outputPath === s)
    ?? m.runs.find((r) => r.sourceModel === s)
    ?? m.runs.find((r) => r.sourceModel && r.sourceModel.endsWith(`/${s}`))
    ?? null
}

export function verifiedRuns(m) {
  return m.runs.filter((r) => r.status === 'verified')
}

/** 同源模型的全部 run（用于回退链与"重复量化"判定）。 */
export function runsForSource(m, sourceModel) {
  return m.runs.filter((r) => r.sourceModel === sourceModel).sort((a, b) => (a.createdAt ?? '').localeCompare(b.createdAt ?? ''))
}

export function pushHistory(run, event, detail = null) {
  if (!Array.isArray(run.history)) run.history = []
  run.history.push({ at: new Date().toISOString(), event, detail })
  if (run.history.length > 200) run.history = run.history.slice(-200)
  return run
}

/**
 * 给一条 run 打**机读失效标记**（F-5）。
 *
 * ## 为什么必须是机读的
 *
 * 集成测试实测：`qr_25b26474a21b` 至今在矩阵里是 `status: verified` / `accuracy.passed: true` /
 * `ppl = 151935.97`（= vocab_size，均匀输出签名），而它产生于一个 223/338 张量全零的残缺模型。
 * 作废只写在 `DEPLOYMENT_REPORT.md` 与 `ACCEPTANCE_MATRIX.md` 的**散文**里 ——
 * 下游（serving / tuning / 编排器 / RSI）读的是状态文件，**读不到的不存在**。
 *
 * 而且 serving **确实据此上线过端点**（`ep_4d4a641dc2`）—— 这条链真的漏下去过。
 *
 * 标记形态见 `inference-contracts/state-schema/quantization-matrix.schema.json` 的 `invalid`。
 * **不改 status**：状态记录的是"当时发生了什么"，失效标记记录的是"事后判定它不可用"，
 * 两者是不同的事实，混在一起会丢掉历史。
 */
export function markRunInvalid(run, { code, reason, evidence = null, at = null }) {
  if (!run) return null
  run.invalid = {
    code,
    at: at ?? new Date().toISOString(),
    reason: String(reason ?? '').slice(0, 600),
    evidence,
  }
  pushHistory(run, 'marked-invalid', { code, reason: String(reason ?? '').slice(0, 200) })
  return run
}

/** 一条 run 是否已被判定不可用（下游据此**不要**把它当作验收证据）。 */
export function isRunInvalid(run) {
  return Boolean(run && run.invalid && run.invalid.code)
}

/**
 * 矩阵条目的五要素摘要（量化专项验收第 4 条：方法 / 精度 / 性能 / 平台 / 状态）。
 * 刻意把"该有的字段缺失"显式暴露为 null，而不是省略 —— 缺失本身是验收要看的东西。
 */
export function summarizeRun(run) {
  return {
    id: run.id,
    sourceModel: run.sourceModel,
    method: run.method ?? null,
    level: run.level ?? null,
    bits: run.bits ?? null,
    accuracy: run.accuracy ?? null,
    performance: run.performance ?? null,
    platform: run.platform ?? null,
    status: run.status ?? null,
    outputPath: run.outputPath ?? null,
    sourceHashUnchanged: run.sourceIntegrity?.unchanged ?? null,
    leakageFree: run.leakage?.clean ?? null,
    createdAt: run.createdAt ?? null,
    verifiedAt: run.verifiedAt ?? null,
    degradeSteps: run.degrade?.steps ?? null,
    // 失效标记必须出现在列表里 —— 否则 `list` 会把一条已作废的 run 显示得像正常记录
    invalid: run.invalid ?? null,
  }
}

/** 一条 run 是否可以作为验收证据：status=verified **且**没有失效标记。 */
export function isEvidenceGrade(run) {
  return run?.status === 'verified' && !isRunInvalid(run)
}

/** 五要素是否齐全（供 list / 验收自检使用）。 */
export function completeness(run) {
  const need = ['method', 'accuracy', 'performance', 'platform', 'status']
  const missing = need.filter((k) => run[k] === undefined || run[k] === null)
  return { complete: missing.length === 0, missing }
}
