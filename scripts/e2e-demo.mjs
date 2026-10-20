#!/usr/bin/env node
/**
 * 真机端到端验收演示（`npm run demo`）。
 *
 * 用**真实容器 + 真实 GPU + 真实 serving 端点**跑完整闭环，并逐条核对量化专项验收。
 * 产物：demo-out/e2e-demo.json（机器可读）+ e2e-demo.md（人读）+ console.log
 *
 * 用法：
 *   node scripts/e2e-demo.mjs --model PATH --accuracy-drop 5 --perf-gain -5 [--out DIR] [--skip-degrade]
 */
import { promises as fs } from 'node:fs'
import { join, resolve } from 'node:path'
import * as api from '../src/api.mjs'
import { loadMatrix } from '../src/state/matrix.mjs'
import { createRuntime } from '../src/config.mjs'
import { hashDirectory } from '../src/integrity.mjs'
import { verifyDatasetDir } from '../src/dataset/prepare.mjs'

const argv = process.argv.slice(2)
const getArg = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d }
const has = (k) => argv.includes(k)

const MODEL = getArg('--model', process.env.DSH_DEMO_MODEL || '')
if (!MODEL) { console.error('缺少 --model：请给出本地模型路径，例如 --model /path/to/models/<model>'); process.exit(2) }
const ACC_DROP = Number(getArg('--accuracy-drop', '5'))
const PERF_GAIN = Number(getArg('--perf-gain', '-5'))
const OUT = resolve(getArg('--out', join(process.cwd(), 'demo-out')))
const CALIB = Number(getArg('--calibration-samples', '128'))
const SKIP_DEGRADE = has('--skip-degrade')

const t0 = Date.now()
const steps = []
const notes = []
const checks = []

function log(m) { process.stdout.write(`${m}\n`) }
function record(name, data) { steps.push({ name, atMs: Date.now() - t0, ...data }) }

async function timed(name, fn) {
  const s = Date.now()
  log(`\n\u001b[1m▶ ${name}\u001b[0m`)
  try {
    const out = await fn()
    record(name, { ok: true, ms: Date.now() - s, detail: tidy(out) })
    return out
  } catch (err) {
    record(name, { ok: false, ms: Date.now() - s, error: String(err?.message ?? err) })
    throw err
  }
}

function tidy(o) {
  if (!o || typeof o !== 'object') return o ?? null
  const { report, warnings, ...rest } = o
  const r = report
  return {
    ...rest,
    ...(r ? {
      outcome: r.outcome, method: r.method, degraded: r.degraded, chain: r.chain,
      parityDropPct: r.parity?.drop?.pct ?? null,
      perfGainPct: r.perf?.metrics?.decodeTpsGainPct ?? null,
      leakageVerified: r.leakage?.verified, sourceUnchanged: r.source?.unchanged,
      attempts: r.attempts?.length, durationMs: r.durationMs,
    } : {}),
    warnings: warnings?.length ?? 0,
  }
}

/** 令牌累积重放（本插件有三类令牌：cf_ 风险 / nlq_ 近上限 / qat_ 授权）。 */
async function withConfirm(invoke) {
  const tokens = []
  for (let i = 0; i < 6; i += 1) {
    const out = await invoke(tokens)
    if (out.ok) return out
    const t = out.details?.confirmToken
    if (!t || tokens.includes(t)) return out
    log(`  \u001b[33m需确认\u001b[0m ${out.code} → ${t}`)
    tokens.push(t)
  }
  return invoke(tokens)
}

function mark(id, ok, evidence, requirement) {
  checks.push({ id, requirement, passed: Boolean(ok), evidence })
  log(`  ${ok ? '\u001b[32m✓\u001b[0m' : '\u001b[31m✗\u001b[0m'} [${id}] ${requirement}`)
  if (!ok) log(`      \u001b[31m证据：${evidence}\u001b[0m`)
}

async function main() {
  await fs.mkdir(OUT, { recursive: true })
  log('\u001b[1m══ inference-quantization 真机端到端验收演示 ══\u001b[0m')
  log(`模型：${MODEL}`)
  log(`阈值：精度损失 ≤ ${ACC_DROP}%   性能收益 ≥ ${PERF_GAIN}%   校准样本 ${CALIB}`)

  // ---------- 0. 环境 ----------
  const doc = await timed('doctor：环境 / 方法 / 镜像 / 契约自检', () => api.doctor({}, {}))
  log(`  GPU ${doc.environment.gpu?.[0]?.name}  可用 ${(doc.environment.capabilities.availableMemoryBytes / 1024 ** 3).toFixed(1)} GiB`)
  log(`  沙箱 进程=${doc.sandbox.process.available ? '✓' : '✗'} 容器=${doc.sandbox.container.available ? '✓' : '✗'} VM=${doc.sandbox.vm.available ? '✓' : '✗'}`)
  log(`  方法 ${doc.methods.map((m) => `L${m.level}:${m.id}`).join('  ')}`)
  mark('ENV-1', doc.environment.container.available && doc.environment.container.nvidiaRuntime, `container=${doc.environment.container.detail}`, '环境：容器级 + nvidia runtime 可用（R2 沙箱）')
  mark('ENV-2', doc.sandbox.container.available, `sandbox.container=${doc.sandbox.container.available}`, '安全 1：沙箱分层可用性已实测（不是查二进制）')
  mark('ENV-3', doc.testingCore?.internal === true && doc.testingCore.contractVersion, `contract=${doc.testingCore?.contractVersion} impl=${doc.testingCore?.implementation}`, '工程：testing-core 不存在 → 内置 @internal 实现，契约已声明')

  // ---------- 1. 镜像验证 ----------
  const img = await timed('image verify（可复现 + 依赖清单 + 扫描 + 签名）', () => api.imageVerify({}, {}))
  log(`  镜像 ${img.image}  digest=${String(img.digest).slice(0, 40)}`)
  log(`  扫描 ${img.scan?.scanned ? img.scan.scanner + ' → ' + img.scan.gate : '未扫描（缺扫描器）'}`)
  log(`  签名 ${img.signature?.signed ? img.signature.signer : '未签名（缺签名工具）'}`)
  log(`  依赖 ${JSON.stringify(img.toolchain ?? {})}`)
  mark('IMG-1', Boolean(img.digest), `digest=${img.digest}`, '工程：量化镜像可复现（固定 base digest + 依赖钉版）')
  mark('IMG-2', img.verified === true, `verified=${img.verified} 缺=[${(img.missing ?? []).join('；')}]`, '安全：镜像四要素（digest/依赖/扫描/签名）全部通过')
  if (!img.verified) notes.push(`量化镜像**未通过完整验证**（缺：${(img.missing ?? []).join('；')}）—— 本机没有 trivy/grype/cosign。演示将显式授权使用未验证镜像，并把它记入审计（这正是设计要的行为：缺什么就说什么，不假装通过）。`)

  // ---------- 2. 数据集 + 去泄漏 ----------
  const ds = await timed('dataset prepare（下载 → 内容哈希去重 → 切分 → 宿主独立复核去泄漏）', () =>
    withConfirm((t) => api.datasetPrepare({ calib: CALIB, test: 200 }, {})))
  log(`  目录 ${ds.dir}`)
  log(`  计数 ${JSON.stringify(ds.counts)}  criterion=${ds.leakage?.criterion}`)
  const runtime = await createRuntime({ options: {} })
  runtime.__packageRoot = new URL('../', import.meta.url).pathname
  const recheck = await verifyDatasetDir(runtime, ds.dir)
  mark('LEAK-1', recheck.verification.verified === true, `problems=${JSON.stringify(recheck.verification.problems)}`, '量化专项 6：校准集与测试集**内容哈希交集为空**（宿主独立重算，不采信容器自检）')
  mark('LEAK-2', recheck.verification.criterion === 'content-sha256-intersection-empty', `criterion=${recheck.verification.criterion}`, '安全：去泄漏判据是内容哈希（不是索引不重叠）')
  mark('LEAK-3', Boolean(recheck.lineage?.source?.repo), `repo=${recheck.lineage?.source?.repo} revision=${recheck.lineage?.source?.revision}`, '安全：校准数据血缘落盘（来源 URL + revision + 种子）')

  // ---------- 3. 源模型基线哈希 ----------
  const srcHashBefore = await timed('源模型基线哈希（量化前）', () => hashDirectory(MODEL))
  log(`  hash=${srcHashBefore.hash.slice(0, 24)}…  files=${srcHashBefore.files}  ${(srcHashBefore.bytes / 1024 ** 3).toFixed(2)} GiB`)

  // ---------- 4. 主闭环：量化 + 双门禁 ----------
  const runA = await timed(`quantize（真实 PTQ + 精度/性能双门禁；阈值 ${ACC_DROP}% / ${PERF_GAIN}%）`, () =>
    withConfirm((tokens) => api.quantize(
      { model: MODEL, calibrationSamples: CALIB, allowUnverifiedImage: true, confirm: tokens },
      { options: { maxAccuracyDropPct: ACC_DROP, minPerfGainPct: PERF_GAIN, calibrationSamples: CALIB } },
    )))

  if (runA.ok && runA.signed) {
    const r = runA.report
    log(`  \u001b[32m✓ 已签收\u001b[0m  runId=${r.runId}  方法=${r.method}  降级 ${r.degraded} 档  耗时 ${(r.durationMs / 1000).toFixed(0)}s`)
    log(`  精度  损失 ${r.parity.drop.pct.toFixed(3)}% ≤ ${ACC_DROP}%   (ppl ${r.parity.baseline.perplexity.toFixed(3)} → ${r.parity.candidate.perplexity.toFixed(3)}, n=${r.parity.samples})`)
    log(`  性能  解码吞吐 ${r.perf.metrics.decodeTpsGainPct.toFixed(2)}% ≥ ${PERF_GAIN}%`)
    log(`  产出  ${r.output}`)
    notes.push(`主闭环实测：${r.method} 把困惑度从 ${r.parity.baseline.perplexity.toFixed(3)} 变成 ${r.parity.candidate.perplexity.toFixed(3)}`
      + `（退化 ${r.parity.drop.pct.toFixed(3)}%，测试集 ${r.parity.samples} 条）；解码吞吐变化 ${r.perf.metrics.decodeTpsGainPct.toFixed(2)}%。`)
    mark('ACC-1', r.parity.drop.pct <= ACC_DROP, `drop=${r.parity.drop.pct}% threshold=${ACC_DROP}%`, `量化专项 1：量化前后精度损失 ≤ 阈值`)
    mark('PERF-1', r.perf.metrics.decodeTpsGainPct >= PERF_GAIN, `gain=${r.perf.metrics.decodeTpsGainPct}% target=${PERF_GAIN}%`, '量化专项 2：性能收益 ≥ 目标')
    mark('GATE-1', r.gates?.accuracy?.passed === true && r.gates?.perf?.passed === true, `decision=${r.gates?.decision}`, '安全：双门禁独立判定（精度由外部阈值签收，量化不自签）')
    mark('SRC-1', r.source.unchanged === true, `before=${String(r.source.hashBefore).slice(0, 16)} after=${String(r.source.hashAfter).slice(0, 16)}`, '量化专项 5：原始模型未被修改（内容哈希校验）')

    const m = await loadMatrix(runtime)
    const rec = m.runs.find((x) => x.id === r.runId)
    const five = ['method', 'accuracy', 'performance', 'platform', 'status']
    const missing = five.filter((k) => rec[k] === undefined || rec[k] === null)
    mark('MATRIX-1', missing.length === 0, `missing=${JSON.stringify(missing)}`, '量化专项 4：量化矩阵含方法/精度/性能/平台/状态 五要素')
    log(`  矩阵  ${rec.method} | acc ${rec.accuracy.dropPct.toFixed(3)}% | perf ${rec.performance.decodeTpsGainPct.toFixed(2)}% | platform ${rec.platform.gpu} | status ${rec.status}`)
    mark('MATRIX-2', rec.status === 'verified', `status=${rec.status}`, '功能：通过双门禁后写入矩阵并标记 verified')
  } else {
    log(`  \u001b[31m未签收\u001b[0m code=${runA.code}`)
    log(`  ${runA.message}`)
    notes.push(`主闭环**未签收**：${runA.code} — ${runA.message}。这是诚实的失败结论（不伪造通过）。`)
    mark('ACC-1', false, `runA.code=${runA.code}`, '量化专项 1：量化前后精度损失 ≤ 阈值')
    mark('PERF-1', false, `runA.code=${runA.code}`, '量化专项 2：性能收益 ≥ 目标')
  }

  // ---------- 5. 降级链（严格阈值逼出降级） ----------
  if (!SKIP_DEGRADE) {
    const strictDrop = Math.max(0.01, ACC_DROP / 50)
    const runB = await timed(`quantize（严格阈值 ${strictDrop}% → 逼出降级链）`, () =>
      withConfirm((tokens) => api.quantize(
        { model: MODEL, calibrationSamples: CALIB, allowUnverifiedImage: true, outputName: 'q-degrade-demo', confirm: tokens },
        { options: { maxAccuracyDropPct: strictDrop, minPerfGainPct: PERF_GAIN, calibrationSamples: CALIB } },
      )))
    const rB = runB.report
    log(`  outcome=${rB?.outcome} code=${runB.code ?? '(签收)'} 降级 ${rB?.degraded ?? '?'} 档`)
    if (rB?.chain) log(`  链：${rB.chain}`)
    mark('DEGRADE-1', (rB?.degraded ?? 0) >= 1, `degraded=${rB?.degraded} chain=${rB?.chain}`, '量化专项 3：降级链至少触发一次并成功降级')
    mark('DEGRADE-2', (rB?.attempts?.length ?? 0) <= 5, `attempts=${rB?.attempts?.length}`, '健壮性：降级链有界（最多 5 档）')
    if (rB?.qatPlan) {
      const p = rB.qatPlan
      mark('QAT-1', p.executes === false && p.approval?.cacheable === false, `executes=${p.executes} cacheable=${p.approval?.cacheable}`, '追加原则：QAT 只产出计划不执行训练，且授权每次确认不缓存')
      mark('QAT-2', Boolean(p.data && p.config && p.expectedGain && p.rollbackPath), `keys=${Object.keys(p).join(',')}`, '裁决 #1：QAT 计划含数据/配置/预期收益/回退路径')
      log(`  QAT 计划 ${p.planId}  授权令牌 ${p.approval.token}`)
      notes.push(`降级链实测：${rB.chain} → 产出 QAT 计划（不执行训练）。`)
    }
  }

  // ---------- 6. 报告可读 ----------
  const rep = await timed('get_report（读回量化报告）', () => api.getReport({ runId: runA.report?.runId ?? undefined }, {}))
  if (rep.ok && rep.report) {
    log(`  runId=${rep.run.runId} 五要素完整性=${rep.completeness.complete ? '✓' : '缺 ' + rep.completeness.missing.join(',')}`)
    log(`  报告文件 ${rep.reportPath}`)
    mark('REPORT-1', rep.completeness.complete === true, `missing=${JSON.stringify(rep.completeness.missing)}`, '工程：报告可读回且五要素完整')
  }

  // ---------- 7. 回退 ----------
  if (runA.ok && runA.report?.runId) {
    const rb = await timed('rollback（回退该 run）', () =>
      withConfirm((tokens) => api.rollback({ selector: runA.report.runId, confirm: tokens }, {})))
    log(`  status=${rb.run?.status}  actions=${JSON.stringify(rb.actions)}`)
    mark('ROLLBACK-1', rb.ok && rb.run?.status === 'rolled-back', `status=${rb.run?.status}`, '功能 4：回退可用且可复核')
    mark('ROLLBACK-2', rb.sourceModelUntouched === true, `sourceModelUntouched=${rb.sourceModelUntouched}`, '安全：回退不触碰源模型')
  }

  // ---------- 8. 源模型哈希复核 ----------
  const srcHashAfter = await timed('源模型哈希复核（量化全程结束后）', () => hashDirectory(MODEL))
  mark('SRC-2', srcHashAfter.hash === srcHashBefore.hash, `before=${srcHashBefore.hash.slice(0, 16)} after=${srcHashAfter.hash.slice(0, 16)}`, '量化专项 5（复核）：整个演示结束后源模型内容哈希未变')

  // ---------- 9. 审计与指标 ----------
  const av = await timed('audit-verify（哈希链完整性）', () => api.auditVerify({}, {}))
  log(`  审计文件 ${av.files.join(', ')}  链校验 ${av.verification.ok ? '✓' : '✗'}（${av.verification.count} 条）`)
  mark('AUDIT-1', av.verification.ok === true, `ok=${av.verification.ok} count=${av.verification.count}`, '功能 6 / 安全：全流程审计，哈希链完整可检出删改')
  const mt = await timed('metrics', () => api.metrics({}, {}))
  log(`  量化尝试 ${mt.metrics.counters.quantize_attempts}  降级 ${mt.metrics.counters.degrade_steps}  门禁通过 ${mt.metrics.counters.gate_pass}  去泄漏检查 ${mt.metrics.counters.leakage_checks}`)
  log(`  最长量化 ${mt.metrics.timings.quantize_ms_max}ms  末次压缩比 ${mt.metrics.resources.last_compression_ratio ?? '-'}`)
  mark('OBS-1', mt.metrics.counters.quantize_attempts >= 1 && mt.metrics.timings.quantize_ms_max > 0, `attempts=${mt.metrics.counters.quantize_attempts}`, '健壮性：可观测（结构化日志 + 指标）')

  // ---------- 收尾 ----------
  const summary = {
    schema: 'inference-quantization/e2e-demo@1',
    ranAt: new Date().toISOString(),
    model: MODEL,
    thresholds: { maxAccuracyDropPct: ACC_DROP, minPerfGainPct: PERF_GAIN },
    totalMs: Date.now() - t0,
    steps, notes, checks,
    passed: checks.filter((c) => c.passed).length,
    failed: checks.filter((c) => !c.passed).length,
    runA: runA.report ?? runA,
    matrix: (await loadMatrix(runtime)).runs.map((r) => ({
      id: r.id, method: r.method, level: r.level, status: r.status,
      accuracyDropPct: r.accuracy?.dropPct ?? null, perfGainPct: r.performance?.decodeTpsGainPct ?? null,
      platform: r.platform?.gpu ?? null, sourceUnchanged: r.sourceIntegrity?.unchanged ?? null, leakageFree: r.leakage?.clean ?? null,
    })),
  }
  await fs.writeFile(join(OUT, 'e2e-demo.json'), `${JSON.stringify(summary, null, 2)}\n`)
  await fs.writeFile(join(OUT, 'e2e-demo.md'), renderMd(summary))
  const audit = await (async () => {
    const { readAudit } = await import('../src/audit.mjs')
    const recs = await readAudit(runtime)
    return recs.slice(-12).map((r) => ({
      seq: r.seq, ts: r.ts, auditId: r.auditId, capability: r.capability, action: r.action,
      phase: r.phase, riskClass: r.riskClass, sandboxTier: r.sandboxTier, outcome: r.outcome,
      code: r.code ?? null, headless: r.headless, prevHash: r.prevHash, hash: r.hash,
    }))
  })()
  await fs.writeFile(join(OUT, 'audit-sample.jsonl'), `${audit.map((r) => JSON.stringify(r)).join('\n')}\n`)

  log(`\n\u001b[1m══ 演示完成 ══\u001b[0m  通过 ${summary.passed} / 失败 ${summary.failed}  总耗时 ${(summary.totalMs / 1000).toFixed(0)}s`)
  log(`产物：${join(OUT, 'e2e-demo.json')} / e2e-demo.md / audit-sample.jsonl`)
  if (summary.failed > 0) process.exitCode = 1
}

function renderMd(s) {
  const L = ['# inference-quantization 真机端到端演示', '']
  L.push(`- 时间：${s.ranAt}`)
  L.push(`- 模型：\`${s.model}\``)
  L.push(`- 阈值：精度损失 ≤ ${s.thresholds.maxAccuracyDropPct}%，性能收益 ≥ ${s.thresholds.minPerfGainPct}%`)
  L.push(`- 总耗时：${(s.totalMs / 1000).toFixed(0)}s`)
  L.push(`- 验收点：**通过 ${s.passed} / 失败 ${s.failed}**`)
  L.push('', '## 逐条验收点', '', '| ID | 要求 | 结果 | 证据 |', '|---|---|---|---|')
  for (const c of s.checks) L.push(`| ${c.id} | ${c.requirement} | ${c.passed ? '✅' : '❌'} | \`${String(c.evidence).slice(0, 200)}\` |`)
  L.push('', '## 步骤', '', '| 步骤 | 结果 | 耗时 |', '|---|---|---:|')
  for (const st of s.steps) L.push(`| ${st.name} | ${st.ok ? '✅' : '❌'} | ${(st.ms / 1000).toFixed(1)}s |`)
  L.push('', '## 结论要点', '')
  for (const n of s.notes) L.push(`- ${n}`)
  L.push('', '## 量化矩阵', '', '| runId | 方法 | 状态 | 精度损失 | 性能收益 | 平台 | 源未改 | 去泄漏 |', '|---|---|---|---:|---:|---|---|---|')
  for (const r of s.matrix) L.push(`| ${r.id} | ${r.method} (L${r.level}) | ${r.status} | ${r.accuracyDropPct?.toFixed(3) ?? '-'}% | ${r.perfGainPct?.toFixed(2) ?? '-'}% | ${r.platform ?? '-'} | ${r.sourceUnchanged ? '✅' : '❌'} | ${r.leakageFree ? '✅' : '❌'} |`)
  L.push('')
  return `${L.join('\n')}\n`
}

main().catch(async (err) => {
  process.stderr.write(`\n\u001b[31m演示失败：${err?.message ?? err}\u001b[0m\n${err?.stack ?? ''}\n`)
  try {
    await fs.mkdir(OUT, { recursive: true })
    await fs.writeFile(join(OUT, 'e2e-demo-error.json'), `${JSON.stringify({ error: String(err?.message ?? err), stack: err?.stack, steps, checks }, null, 2)}\n`)
  } catch { /* 尽力 */ }
  process.exit(1)
})
