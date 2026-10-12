/**
 * CLI（`inference-quantization <command>`）。
 *
 * 零依赖参数解析。与 MCP 工具走**同一层 api.mjs**，因此行为一致。
 *
 * 退出码约定（便于脚本与 CI 判断）：
 *   0 成功
 *   1 业务失败（错误已结构化打印）
 *   2 用法错误
 *   3 需要确认（错误码 E_CONFIRMATION_REQUIRED / E_HEADLESS_NEEDS_CONFIRMATION）
 */
import { promises as fs } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'
import * as api from './api.mjs'
import { ALL_TOOL_SPECS, TOOL_SPECS } from './tools.mjs'
import { ERROR_CODES } from './errors.mjs'

const COMMANDS = {
  quantize: 'quantize <model> --max-accuracy-drop-pct N --min-perf-gain-pct N [options]',
  report: 'report [selector] [--json]                    —— 读量化报告（不给 selector 则列全部 run）',
  list: 'list [--status S] [--verified] [--json]       —— 列出量化矩阵（含五要素完整性）',
  rollback: 'rollback <selector> [--purge-output]        —— 回退一次 run（可选删除产出目录）',
  qat: 'qat [--runId ID] [--authorize --allow-sandbox-degrade]  —— 产出 QAT 计划（不执行训练）',
  validate: 'validate <candidate> --baseline <model> --max-accuracy-drop-pct N [--gate accuracy|accuracy+perf]  —— 验收一个**已存在**的量化 checkpoint（不重新量化）',
  dataset: 'dataset <prepare|verify|list> [options]     —— 校准/测试集准备与去泄漏复核',
  image: 'image <verify|build> [--image IMG]           —— 量化镜像验证（扫描/签名/依赖/可复现）',
  metrics: 'metrics [--json]                           —— 指标',
  'audit-verify': 'audit-verify [--json]                —— 校验审计哈希链完整性',
  doctor: 'doctor [--deep] [--json]                   —— 环境、方法、镜像、沙箱、契约自检',
  contract: 'contract                                  —— 打印 testing-core 契约声明',
  tools: 'tools [--json]                             —— 列出对外工具与参数 schema',
  'install-skills': 'install-skills [--dir PATH]          —— 把随包 SKILL.md 复制到 skills 目录',
  version: 'version                                    —— 版本',
  help: 'help                                       —— 本帮助',
}

const FLAGS = new Set(['json', 'deep', 'headless', 'help', 'verified', 'purge-output', 'authorize', 'allow-sandbox-degrade', 'allow-unverified-image', 'force', 'build'])

export function parseArgv(argv) {
  const positional = []
  const options = {}
  const flags = new Set()
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i]
    if (a === '--') { positional.push(...argv.slice(i + 1)); break }
    if (a.startsWith('--')) {
      const body = a.slice(2)
      const eq = body.indexOf('=')
      if (eq >= 0) {
        options[body.slice(0, eq)] = body.slice(eq + 1)
      } else if (FLAGS.has(body)) {
        flags.add(body)
      } else {
        const next = argv[i + 1]
        if (next === undefined || next.startsWith('--')) { flags.add(body) } else { options[body] = next; i += 1 }
      }
    } else if (a.startsWith('-') && a.length > 1 && !/^-\d/.test(a)) {
      const body = a.slice(1)
      if (body === 'h') flags.add('help')
      else flags.add(body)
    } else {
      positional.push(a)
    }
  }
  return { positional, options, flags }
}

function asNumber(v) {
  if (v === undefined || v === null || v === '') return undefined
  const n = Number(v)
  return Number.isFinite(n) ? n : undefined
}

function asBool(v) {
  if (v === undefined) return undefined
  if (typeof v === 'boolean') return v
  return !['false', '0', 'no', 'off'].includes(String(v).toLowerCase())
}

function buildCommonOptions(options, flags) {
  const raw = {
    headless: flags.has('headless') ? true : asBool(options.headless),
    modelRoots: options['model-roots'] ? String(options['model-roots']).split(',').map((x) => x.trim()).filter(Boolean) : undefined,
    containerRuntime: options['container-runtime'],
    quantImage: options['quant-image'],
    servingPath: options['serving-path'],
    maxDegradeSteps: asNumber(options['max-degrade-steps']),
    quantTimeoutMs: asNumber(options['quant-timeout-ms']),
    evalTimeoutMs: asNumber(options['eval-timeout-ms']),
  }
  // 去掉 undefined：让默认值生效（否则显式 undefined 会覆盖掉默认值）
  return Object.fromEntries(Object.entries(raw).filter(([, v]) => v !== undefined))
}

/* ------------------------------------------------------------------ output */

const NO_COLOR = Boolean(process.env.NO_COLOR) || process.stdout?.isTTY === false
const STYLES = ['dim', 'bold', 'green', 'red', 'yellow', 'cyan']
const C = NO_COLOR
  ? Object.fromEntries(STYLES.map((s) => [s, (x) => String(x)]))
  : {
      dim: (s) => `\u001b[2m${s}\u001b[0m`,
      bold: (s) => `\u001b[1m${s}\u001b[0m`,
      green: (s) => `\u001b[32m${s}\u001b[0m`,
      red: (s) => `\u001b[31m${s}\u001b[0m`,
      yellow: (s) => `\u001b[33m${s}\u001b[0m`,
      cyan: (s) => `\u001b[36m${s}\u001b[0m`,
    }

function print(out, { json }) {
  if (json) {
    process.stdout.write(`${JSON.stringify(out, null, 2)}\n`)
    return
  }
  if (out && out.ok === false) {
    process.stdout.write(`${C.red('✗')} ${out.code}: ${out.message}\n`)
    if (out.remediation) process.stdout.write(`  ${C.dim('修复：')}${out.remediation}\n`)
    if (out.details) process.stdout.write(`  ${C.dim('细节：')}\n${indent(JSON.stringify(out.details, null, 2))}\n`)
    return
  }
  process.stdout.write(`${formatHuman(out)}\n`)
}

function indent(s, n = 4) {
  return String(s).split('\n').map((l) => ' '.repeat(n) + l).join('\n')
}

function formatHuman(out) {
  if (!out || typeof out !== 'object') return String(out)
  const lines = []

  if (out.endpoint && out.endpoint.url) {
    lines.push(`${C.green('✓')} ${C.bold('端点就绪')}${out.reused ? C.dim('（幂等命中，复用既有实例）') : ''}`)
    lines.push(`  URL       ${C.cyan(out.endpoint.url)}`)
    lines.push(`  模型名    ${out.endpoint.servedModelName}`)
    lines.push(`  模型      ${out.endpoint.model ?? out.report?.model?.resolvedPath ?? ''}`)
    lines.push(`  后端      ${out.endpoint.backend ?? out.report?.backend?.id ?? ''}`)
    if (out.report) {
      lines.push(`  耗时      ${(out.report.durationMs / 1000).toFixed(1)}s`)
      lines.push(`  降级      步数=${out.report.degrade?.steps ?? 0}${out.report.degrade?.lossy ? C.yellow('（含损档：精度）') : ''}`)
      if (out.report.attempts?.length) {
        lines.push(`  尝试      共 ${out.report.attempts.length} 次：${out.report.attempts.map((a) => `${a.tier}${a.failureCode ? `(${a.failureCode})` : ''}`).join(' → ')}`)
      }
      if (out.report.assumptions?.length) {
        lines.push(`  ${C.dim(`自动补全假设 ${out.report.assumptions.length} 条：`)}`)
        for (const a of out.report.assumptions.slice(0, 8)) lines.push(C.dim(`    · ${a.field} = ${JSON.stringify(a.value)}  ← ${a.reason}`))
      }
    }
    if (out.auditId) lines.push(C.dim(`  审计      ${out.auditId}`))
  }

  if (Array.isArray(out.endpoints)) {
    if (out.endpoints.length === 0) lines.push(C.dim('（无已注册端点）'))
    else {
      lines.push(C.bold(`共 ${out.count ?? out.endpoints.length} 个端点`))
      for (const e of out.endpoints) {
        const mark = e.status === 'serving' ? C.green('●') : e.status === 'torn_down' ? C.dim('○') : C.yellow('●')
        lines.push(`  ${mark} ${e.id}  ${e.endpoint ?? '-'}  ${e.model}  [${e.status}]`)
        if (e.consistency && !e.consistency.ok) lines.push(C.yellow(`      ⚠ ${e.consistency.message}`))
      }
    }
  } else if (out.endpoints && typeof out.endpoints === 'object') {
    const v = out.endpoints
    lines.push(`${C.bold('注册表')}  ${out.registry?.file}`)
    lines.push(`  端点 ${out.registry?.endpointCount} 个（存活 ${out.registry?.liveCount}）`)
    const t = Array.isArray(v) ? v[0] : v
    if (t) {
      lines.push(`  ${t.id}  ${t.endpoint ?? '-'}  [${t.status}]`)
      if (t.consistency) lines.push(`  ${t.consistency.ok ? C.green('✓') : C.red('✗')} ${t.consistency.message}`)
    }
    if (out.machine?.orphans?.length) lines.push(C.yellow(`  ⚠ 孤儿容器 ${out.machine.orphans.length} 个：${out.machine.orphans.map((o) => o.name).join(', ')}`))
  }

  if (out.verification) {
    lines.push(`${out.verification.clean ? C.green('✓ 清理干净') : C.red('✗ 存在残留')}`)
    for (const c of out.verification.checks) lines.push(`  ${c.ok ? C.green('✓') : C.red('✗')} ${c.name}: ${c.detail}`)
  }

  if (out.metrics) {
    const m = out.metrics
    lines.push(C.bold('指标'))
    for (const [k, v] of Object.entries(m.counters)) lines.push(`  ${k.padEnd(24)} ${v}`)
    lines.push(`  ${'deploy_ms_max'.padEnd(24)} ${m.timings.deploy_ms_max}`)
  }

  if (out.methods && out.testingCore) {
    const e = out.environment ?? {}
    lines.push(`${C.bold('inference-quantization doctor')}  v${out.version?.package}`)
    lines.push(`  manifest   ${out.manifest.name}  能力：${out.manifest.capabilities.join(', ')}`)
    lines.push(`  不做       ${out.manifest.notProvided.join(', ')}`)
    lines.push(`  沙箱       进程=${e.capabilities?.sandboxTiers?.process ? C.green('✓') : C.red('✗')} 容器=${e.capabilities?.sandboxTiers?.container ? C.green('✓') : C.red('✗')} VM=${e.capabilities?.sandboxTiers?.vm ? C.green('✓') : C.red('✗')}`)
    lines.push(`  GPU        ${e.gpu?.[0]?.name ?? '(无)'}${e.capabilities?.unifiedMemory ? ' [统一内存]' : ''}  available=${fmtBytes(e.capabilities?.availableMemoryBytes)}`)
    lines.push(`  Docker     ${e.container?.available ? C.green('✓') : C.red('✗')} ${C.dim(e.container?.detail ?? '')}`)
    lines.push(C.bold('  量化方法   ') + out.methods.map((m) => `L${m.level}:${m.id}(${m.bits}bit${m.available ? '' : ' 不可用'})`).join('  '))
    const img = out.image ?? {}
    lines.push(`  量化镜像   ${img.verified ? C.green('已验证') : C.yellow('未验证')} ${C.dim(img.image ?? '')}${img.missing?.length ? C.yellow(`  缺：${img.missing.join('；')}`) : ''}`)
    if (img.toolchain?.llmcompressor) lines.push(C.dim(`             llmcompressor=${img.toolchain.llmcompressor} compressed-tensors=${img.toolchain.compressed_tensors} transformers=${img.toolchain.transformers}`))
    lines.push(`  门禁阈值   ${out.gates?.loaded ? C.green('gates.yaml 已设置') : C.yellow('未设置 → 调用时必须显式传 --max-accuracy-drop-pct / --min-perf-gain-pct')}`)
    lines.push(`  testing    ${out.testingCore.name} v${out.testingCore.contractVersion} ${C.yellow('(' + out.testingCore.implementation + ')')}`)
    lines.push(`  策略来源   risk=${out.policySources?.risk} network=${out.policySources?.network} gates=${out.policySources?.gates}`)
    if (out.proxyProbe?.envProxy) lines.push(`  代理       ${C.dim(`${out.proxyProbe.envProxy} —— 已按策略直连（不使用）`)}`)
  }

  if (out.run && out.run.method) {
    const r = out.run
    const mark = r.status === 'verified' ? C.green('✓ verified') : r.status === 'qat-plan-required' ? C.yellow('QAT 计划待裁决') : C.red(`✗ ${r.status}`)
    lines.push(`${C.bold('量化 run')} ${r.id}  ${mark}`)
    lines.push(`  源模型    ${r.sourceModel}`)
    lines.push(`  方法      ${r.method}（L${r.level}, ${r.bits}bit）`)
    if (r.accuracy) lines.push(`  精度      损失 ${Number(r.accuracy.dropPct).toFixed(3)}% vs 阈值 ${r.accuracy.thresholdPct}%  ${r.accuracy.passed ? C.green('通过') : C.red('未过')}  ${C.dim(`(ppl ${r.accuracy.baselinePpl?.toFixed(3)} → ${r.accuracy.candidatePpl?.toFixed(3)}, n=${r.accuracy.samples})`)}`)
    if (r.performance) lines.push(`  性能      解码吞吐 ${Number(r.performance.decodeTpsGainPct).toFixed(2)}% vs 目标 ${r.performance.thresholdPct}%  ${r.performance.passed ? C.green('通过') : C.red('未过')}`)
    lines.push(`  平台      ${r.platform?.gpu ?? '?'}${r.platform?.unifiedMemory ? ' [统一内存]' : ''}  沙箱=${r.platform?.sandboxTier ?? '?'}`)
    lines.push(`  证据      源模型未改=${r.sourceHashUnchanged ? C.green('✓') : C.red('✗')}  去泄漏=${r.leakageFree ? C.green('✓') : C.red('✗')}  降级=${r.degradeSteps ?? 0} 步`)
    if (r.outputPath) lines.push(`  产出      ${r.outputPath}`)
    if (out.report?.chain) lines.push(`  降级链    ${C.dim(out.report.chain)}`)
    if (out.report?.qatPlan) lines.push(C.yellow(`  QAT 计划  ${out.report.qatPlan.planId}（不执行训练；授权令牌 ${out.report.qatPlan.approval.token}）`))
  }

  if (Array.isArray(out.runs) && out.runs.length && out.runs[0]?.method !== undefined) {
    lines.push(C.bold(`量化矩阵  共 ${out.count} 条`))
    for (const r of out.runs.slice(-10)) {
      const mark = r.status === 'verified' ? C.green('●') : r.status === 'failed' ? C.red('●') : C.yellow('●')
      lines.push(`  ${mark} ${r.id}  ${r.method ?? '-'}  acc=${r.accuracy ? Number(r.accuracy.dropPct).toFixed(2) + '%' : '-'}  perf=${r.performance ? Number(r.performance.decodeTpsGainPct).toFixed(1) + '%' : '-'}  [${r.status}]  ${r.completeness?.complete ? '' : C.yellow('缺: ' + (r.completeness?.missing ?? []).join(','))}`)
    }
  }

  if (out.qatPlan && out.plan === undefined) {
    lines.push(`${C.bold('QAT 计划')} ${out.qatPlan.planId}  ${C.yellow('（不执行训练）')}`)
    lines.push(`  为什么    ${out.qatPlan.why}`)
    lines.push(`  数据      ${out.qatPlan.data?.calibrationSet?.repo} / ${out.qatPlan.data?.calibrationSet?.config}，${out.qatPlan.data?.calibrationSet?.samples} 样本，seed=${out.qatPlan.data?.calibrationSet?.seed}`)
    lines.push(`  配置      起点=${out.qatPlan.config?.startFrom}  epochs=${out.qatPlan.config?.epochsSuggested}  lr=${out.qatPlan.config?.learningRateSuggested}`)
    lines.push(`  预期      ${out.qatPlan.expectedGain?.caveat ?? ''}`)
    lines.push(`  回退      ${out.qatPlan.rollbackPath?.command}`)
    lines.push(`  授权令牌  ${out.qatPlan.approval?.token} ${C.dim(`(cacheable=${out.qatPlan.approval?.cacheable})`)}`)
  }

  if (out.plan) {
    const p = out.plan
    lines.push(`${C.bold('迁移计划')}  needed=${p.needed}`)
    lines.push(`  源格式    ${p.source.format}`)
    lines.push(`  目标      ${p.target.backend}`)
    for (const s of p.steps) lines.push(`  · [${s.id}] ${s.description}`)
    if (p.risks?.length) { lines.push(C.yellow('  风险：')); for (const r of p.risks) lines.push(`    · (${r.severity}) ${r.description}`) }
  }

  if (out.verification === undefined && out.environment && out.manifest) {
    const e = out.environment
    lines.push(`${C.bold('inference-quantization doctor')}  v${out.version?.package}`)
    lines.push(`  manifest   ${out.manifest.name}  能力：${out.manifest.capabilities.join(', ')}`)
    lines.push(`  沙箱       进程=${e.capabilities?.sandboxTiers?.process ? C.green('✓') : C.red('✗')} 容器=${e.capabilities?.sandboxTiers?.container ? C.green('✓') : C.red('✗')} VM=${e.capabilities?.sandboxTiers?.vm ? C.green('✓') : C.red('✗')}`)
    lines.push(`  GPU        ${e.gpu?.[0]?.name ?? '(无)'}${e.capabilities?.unifiedMemory ? ' [统一内存]' : ''}`)
    lines.push(`  显存       total=${fmtBytes(e.gpu?.[0]?.totalBytes)} available=${fmtBytes(e.capabilities?.availableMemoryBytes)} ${e.capabilities?.memoryIsEstimate ? C.dim('(估算)') : ''}`)
    lines.push(`  Docker     ${e.container?.available ? C.green('✓') : C.red('✗')} ${C.dim(e.container?.detail ?? '')}`)
    lines.push(`  端口池     ${e.ports?.host}:${e.ports?.pool?.join('-')}`)
    lines.push(`  策略来源   risk=${out.policySources?.risk} network=${out.policySources?.network} headless=${out.policySources?.headless}`)
    lines.push(`  headless   ${out.headless}`)
    if (out.proxyProbe?.envProxy) lines.push(`  代理       ${C.dim(`${out.proxyProbe.envProxy} —— 已按策略直连（不使用）`)}`)
  }

  if (lines.length === 0) return JSON.stringify(out, null, 2)
  return lines.join('\n')
}

function fmtBytes(b) {
  if (b === null || b === undefined) return 'n/a'
  return `${(b / 1024 ** 3).toFixed(1)} GiB`
}

/* -------------------------------------------------------------------- run */

export async function run(argv = process.argv.slice(2)) {
  const { positional, options, flags } = parseArgv(argv)
  const cmd = positional[0]
  const args = positional.slice(1)
  const json = flags.has('json')

  if (!cmd || cmd === 'help' || flags.has('help')) {
    printHelp()
    return 0
  }
  if (cmd === 'version') {
    const out = await api.doctor({ deep: false })
    process.stdout.write(`${out.version?.package ?? '?'}\n`)
    return 0
  }

  const common = buildCommonOptions(options, flags)
  const root = options.root

  let out
  switch (cmd) {
    case 'quantize': {
      const target = args[0] ?? options.model ?? options.path
      if (!target) { usageError('quantize 需要一个模型参数（本地路径或模型 ID）'); return 2 }
      out = await api.quantize(
        options.path ? { path: options.path } : { model: target },
        {
          root,
          options: {
            ...common,
            maxAccuracyDropPct: asNumber(options['max-accuracy-drop-pct']),
            minPerfGainPct: asNumber(options['min-perf-gain-pct']),
            method: options.method,
            outputName: options['output-name'],
            outputRoot: options['output-root'],
            calibrationSamples: asNumber(options['calibration-samples']),
            calibrationSeed: asNumber(options['calibration-seed']),
            datasetRepo: options['dataset-repo'],
            datasetConfig: options['dataset-config'],
            testSamples: asNumber(options['test-samples']),
            workload: options.workload,
            requests: asNumber(options.requests),
            concurrency: asNumber(options.concurrency),
            allowUnverifiedImage: flags.has('allow-unverified-image'),
          },
          confirm: options.confirm ?? null,
        },
      )
      break
    }
    case 'report': out = await api.getReport({ selector: args[0] ?? options.selector ?? null, runId: options.runId ?? null }, { root, options: common }); break
    case 'list': out = await api.list({ status: options.status ?? null, sourceModel: options['source-model'] ?? null, verifiedOnly: flags.has('verified') }, { root, options: common }); break
    case 'rollback':
      if (!args[0] && !options.selector) { usageError('rollback 需要 selector'); return 2 }
      out = await api.rollback({ selector: args[0] ?? options.selector, purgeOutput: flags.has('purge-output') }, { root, options: { ...common, confirm: options.confirm } })
      break
    case 'qat':
      out = await api.qat({ runId: options.runId ?? args[0] ?? null, authorize: flags.has('authorize'), allowSandboxDegrade: flags.has('allow-sandbox-degrade') }, { root, options: { ...common, confirm: options.confirm } })
      break
    case 'validate': {
      if (!args[0] && !options.candidate) { usageError('validate 需要 candidate（已存在的量化 checkpoint 目录）'); return 2 }
      const gateMode = options.gate === 'accuracy+perf' ? 'accuracy+perf' : 'accuracy'
      out = await api.validate({
        candidate: args[0] ?? options.candidate,
        baseline: options.baseline ?? options.model ?? options.path ?? null,
        max_accuracy_drop_pct: asNumber(options['max-accuracy-drop-pct']),
        min_perf_gain_pct: asNumber(options['min-perf-gain-pct']),
        gate: gateMode,
        samples: asNumber(options.samples),
        testSamples: asNumber(options['test-samples']),
        allowUnverifiedImage: flags.has('allow-unverified-image'),
      }, { root, options: { ...common, confirm: options.confirm } })
      break
    }
    case 'dataset': {
      const sub = args[0]
      if (sub === 'prepare') out = await api.datasetPrepare({ repo: options.repo, config: options.config, split: options.split, seed: asNumber(options.seed), calib: asNumber(options.calib), test: asNumber(options.test), force: flags.has('force') }, { root, options: common })
      else if (sub === 'verify') out = await api.datasetVerify({ dir: args[1] ?? options.dir }, { root, options: common })
      else if (sub === 'list') out = await api.datasetList({ root, options: common })
      else { usageError('dataset 子命令：prepare | verify | list'); return 2 }
      break
    }
    case 'image': {
      const sub = args[0]
      if (sub === 'verify') out = await api.imageVerify({ image: options.image, build: flags.has('build') }, { root, options: common })
      else if (sub === 'build') out = await api.imageBuild({ image: options.image, dockerfile: options.dockerfile }, { root, options: common })
      else { usageError('image 子命令：verify | build'); return 2 }
      break
    }
    case 'metrics': out = await api.metrics({ root, options: common }); break
    case 'audit-verify': out = await api.auditVerify({ root, options: common }); break
    case 'doctor': out = await api.doctor({ deep: flags.has('deep') }, { root, options: common }); break
    case 'contract': out = await api.contract({ root, options: common }); break
    case 'tools': out = { ok: true, tools: ALL_TOOL_SPECS.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })), count: ALL_TOOL_SPECS.length }; break
    case 'install-skills': out = await installSkills({ dir: options.dir }); break
    default: usageError(`未知命令：${cmd}`); return 2
  }

  print(out, { json })

  if (out && out.ok === false) {
    if (out.code === ERROR_CODES.E_CONFIRMATION_REQUIRED || out.code === ERROR_CODES.E_HEADLESS_NEEDS_CONFIRMATION) return 3
    return 1
  }
  return 0
}

function usageError(msg) {
  process.stderr.write(`${C.red('用法错误')}: ${msg}\n\n`)
  printHelp(process.stderr)
}

function printHelp(stream = process.stdout) {
  stream.write(`${C.bold('inference-quantization')} —— 量化模型并**证明它达标**\n\n`)
  stream.write(`${C.bold('用法')}  inference-quantization <command> [args] [--options]\n\n`)
  for (const [k, v] of Object.entries(COMMANDS)) stream.write(`  ${k.padEnd(16)}${v}\n`)
  stream.write(`\n${C.bold('门禁阈值（必填或写进 gates.yaml，本插件不猜默认值）')}\n`)
  stream.write(`  --max-accuracy-drop-pct N   允许的最大精度损失（困惑度相对退化 %）\n`)
  stream.write(`  --min-perf-gain-pct N       要求的最小解码吞吐提升（%）\n`)
  stream.write(`\n${C.bold('通用选项')}\n`)
  stream.write(`  --root PATH                 共享状态根目录（默认 ~/.ai-infra）\n`)
  stream.write(`  --method ID                 量化方法（默认 awq-w4a16）\n`)
  stream.write(`  --calibration-samples N     校准样本数（默认 512）\n`)
  stream.write(`  --calibration-seed N        切分种子（同种子 → 同切分）\n`)
  stream.write(`  --dataset-repo/--dataset-config  校准数据集（默认 Salesforce/wikitext + wikitext-2-raw-v1）\n`)
  stream.write(`  --confirm TOKEN             确认令牌（首次调用会在错误里给出）\n`)
  stream.write(`  --headless                  需确认的操作一律中止\n`)
  stream.write(`  --json                      机器可读输出\n`)
  stream.write(`\n${C.bold('示例')}\n`)
  stream.write(`  inference-quantization doctor\n`)
  stream.write(`  inference-quantization image verify --build\n`)
  stream.write(`  inference-quantization quantize /data/models/Qwen2.5-1.5B-Instruct \\\n`)
  stream.write(`      --max-accuracy-drop-pct 5 --min-perf-gain-pct 20\n`)
  stream.write(`  inference-quantization list --json\n`)
  stream.write(`  inference-quantization report <runId>\n`)
}

/** 把随包 SKILL.md 安装到 skills 目录（显式动作，不在加载时偷偷写）。 */
export async function installSkills({ dir = null } = {}) {
  const here = dirname(fileURLToPath(import.meta.url))
  const srcRoot = resolve(join(here, '..', 'skills'))
  const targetRoot = resolve(dir ?? join(process.env.DSH_HOME ?? join(process.env.HOME ?? '.', '.dsh'), 'skills'))
  const installed = []
  try {
    const names = await fs.readdir(srcRoot)
    for (const n of names) {
      const src = join(srcRoot, n, 'SKILL.md')
      try {
        const content = await fs.readFile(src, 'utf8')
        const destDir = join(targetRoot, n)
        await fs.mkdir(destDir, { recursive: true })
        await fs.writeFile(join(destDir, 'SKILL.md'), content, 'utf8')
        installed.push(join(destDir, 'SKILL.md'))
      } catch (err) {
        if (err.code !== 'ENOENT') throw err
      }
    }
  } catch (err) {
    return { ok: false, code: ERROR_CODES.E_INTERNAL, message: `安装 skills 失败：${err.message}`, remediation: `确认源目录存在：${srcRoot}`, details: { srcRoot, targetRoot } }
  }
  return { ok: true, source: srcRoot, target: targetRoot, installed, count: installed.length }
}

export { printHelp, formatHuman }
