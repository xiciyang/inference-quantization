/**
 * 运行时配置：路径布局、默认参数、策略加载。
 *
 * 共享状态**外置**在 `~/.ai-infra/`（题目原则 3.1.4），不放在包内 —— 包目录在
 * npm 安装后是只读的，且升级会覆盖。
 */
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import * as fs from 'node:fs/promises'
import { parseFile, parse, stringify } from './util/yaml.mjs'
import { atomicWrite, mkdirp, expandHome, exists } from './util/fsx.mjs'
import { fail, warn, WARNING_CODES, ERROR_CODES } from './errors.mjs'
import {
  DEFAULT_RISK_CLASSIFICATION,
  DEFAULT_NETWORK_POLICY,
  DEFAULT_HEADLESS_POLICY,
} from './policy-defaults.mjs'

export const AI_INFRA_ROOT = '~/.ai-infra'

export function resolvePaths(root = AI_INFRA_ROOT) {
  const base = resolve(expandHome(root))
  return {
    root: base,
    stateDir: join(base, 'state'),
    // 量化矩阵（题面 §共享状态）
    matrixFile: join(base, 'state', 'quantization-matrix.yaml'),
    // 源模型结构预检缓存（键含 size+mtime，文件一变即失效）
    modelValidationFile: join(base, 'state', 'model-validation.json'),
    // 数据集血缘与缓存
    datasetDir: join(base, 'state', 'datasets'),
    lineageDir: join(base, 'state', 'datasets', 'lineage'),
    cacheDir: join(base, 'cache', 'datasets'),
    // 量化产出根目录（与模型根分离，避免污染既有模型）
    outputDir: join(base, 'quantized'),
    confirmationsFile: join(base, 'state', 'confirmations.json'),
    metricsFile: join(base, 'state', 'metrics.json'),
    historyDir: join(base, 'state', 'history'),
    logsDir: join(base, 'logs'),
    auditDir: join(base, 'logs', 'audit', 'inference-quantization'),
    appLogDir: join(base, 'logs', 'inference-quantization'),
    configDir: join(base, 'config'),
    riskFile: join(base, 'config', 'risk-classification.yaml'),
    networkFile: join(base, 'config', 'network-policy.yaml'),
    headlessFile: join(base, 'config', 'headless-policy.yaml'),
    // 门禁阈值（用户裁决 #2：MCP 未传 max_accuracy_drop 时从这里读；两者都无 → 拒绝）
    gatesFile: join(base, 'config', 'gates.yaml'),
    // 量化镜像元数据（digest / 构建日志 / SBOM / 扫描结果）
    imageMetaDir: join(base, 'state', 'images'),
    runDir: join(base, 'run', 'inference-quantization'),
    tmpDir: join(base, 'run', 'inference-quantization', 'tmp'),
    lockDir: join(base, 'run', 'inference-quantization', 'locks'),
    reportsDir: join(base, 'reports', 'inference-quantization'),
  }
}

/** 默认参数。每一项都可通过 `deploy` 的 `options` 覆盖 —— 不硬编码可调参数。 */
export function defaultOptions() {
  return {
    // 端口
    bindHost: '127.0.0.1',
    portPoolStart: 8100,
    portPoolEnd: 8199,

    // 时间预算（**墙钟**，不是调用次数 —— 见 RESULTS.md §2.2 的 A12 反例）
    deployTimeoutMs: 900_000,
    healthTimeoutMs: 420_000,
    healthPollIntervalMs: 3_000,
    cleanupTimeoutMs: 60_000,
    drainTimeoutMs: 15_000,
    commandTimeoutMs: 60_000,

    // 降级
    maxDegradeSteps: 4,

    // 模型
    modelRoots: (process.env.DSH_MODEL_ROOTS || '/data/models').split(':').filter(Boolean),
    containerModelMount: 'identical',

    // 量化
    defaultMethod: 'awq-w4a16',
    quantImage: 'inference-quantization/toolchain:0.1.0',
    outputRoot: null,
    calibrationSamples: 512,
    calibrationSeed: 20260923,
    coarseScreenMinSamples: 200,
    coarseScreenRelaxFactor: 2.0,
    maxDegradeSteps: 4,
    quantTimeoutMs: 3_600_000,
    evalTimeoutMs: 2_400_000,
    datasetRepo: 'Salesforce/wikitext',
    datasetConfig: 'wikitext-2-raw-v1',
    datasetSplit: 'train',
    maxSourceModelBytes: 64 * 1024 ** 3,
    servingPackage: 'inference-serving',

    // 后端
    defaultBackend: 'nvidia-vllm',
    containerRuntime: 'nvidia',
    dockerImageDefault: 'vllm/vllm-openai:nightly',

    // 健康检查
    healthForwardProbe: true,
    healthForwardMaxTokens: 4,

    // 其它
    headless: null, // null = 自动探测
    continueOnAuditFailure: false,
    reportDir: null,
  }
}

export function isHeadless(opts, env = process.env) {
  if (opts.headless === true) return true
  if (opts.headless === false) return false
  if (env.DSH_HEADLESS === '1' || env.INFERENCE_SERVING_HEADLESS === '1') return true
  if (env.CI === 'true' || env.CI === '1') return true
  // 无 TTY 且未显式要求交互 → 视为 headless（保守：宁可中止，不静默放行）
  if (process.stdin && process.stdin.isTTY === false && process.stdout && process.stdout.isTTY === false) return true
  return false
}

async function loadPolicyFile(path, defaults, warnings) {
  if (!(await exists(path))) {
    warnings.push(warn(WARNING_CODES.W_POLICY_DEFAULTED, `策略文件不存在，使用内置默认值：${path}`, { path }))
    return { value: defaults, source: 'builtin', path }
  }
  const raw = await fs.readFile(path, 'utf8')
  let parsed
  try {
    parsed = parse(raw)
  } catch (err) {
    fail(ERROR_CODES.E_INPUT_INVALID, `策略文件解析失败：${path}`, {
      remediation: `修正该 YAML 语法（本包支持受限 YAML 子集，见 src/util/yaml.mjs），或删除该文件以回落到内置默认值。`,
      details: { path, parseError: String(err.message) },
      cause: err,
    })
  }
  return { value: parsed, source: 'file', path }
}

/** 确保 `~/.ai-infra/` 目录树存在。幂等。 */
export async function ensureLayout(paths) {
  for (const d of [paths.root, paths.stateDir, paths.historyDir, paths.logsDir, paths.auditDir, paths.appLogDir, paths.configDir, paths.runDir, paths.tmpDir, paths.lockDir, paths.reportsDir, paths.datasetDir, paths.lineageDir, paths.cacheDir, paths.outputDir, paths.imageMetaDir]) {
    await mkdirp(d)
  }
  return paths
}

/** 首次运行：把内置策略落盘成配置文件，便于人工复核与修改。幂等（不覆盖既有）。 */
export async function materializePolicies(paths) {
  const written = []
  const pairs = [
    [paths.riskFile, DEFAULT_RISK_CLASSIFICATION],
    [paths.networkFile, DEFAULT_NETWORK_POLICY],
    [paths.headlessFile, DEFAULT_HEADLESS_POLICY],
  ]
  for (const [file, value] of pairs) {
    if (await exists(file)) continue
    await atomicWrite(file, stringify(value, { header: `${value.note ?? 'inference-quantization policy'} — 由 inference-quantization 内置默认值生成` }))
    written.push(file)
  }
  return written
}

export async function loadPolicy(paths, which, warnings) {
  if (which === 'risk') return loadPolicyFile(paths.riskFile, DEFAULT_RISK_CLASSIFICATION, warnings)
  if (which === 'network') return loadPolicyFile(paths.networkFile, DEFAULT_NETWORK_POLICY, warnings)
  if (which === 'headless') return loadPolicyFile(paths.headlessFile, DEFAULT_HEADLESS_POLICY, warnings)
  throw new Error(`unknown policy: ${which}`)
}

/** 组装一次调用的完整运行时上下文。 */
export async function createRuntime({ root, options = {}, env = process.env } = {}) {
  const paths = resolvePaths(root ?? env.INFERENCE_SERVING_ROOT ?? AI_INFRA_ROOT)
  // 丢弃 undefined 的覆盖项：`{...defaults, ...options}` 里一个显式的 `undefined`
  // 会把默认值清掉（CLI 的 buildCommonOptions 就会产出 undefined 字段）。
  // 这是"默认值被静默清空"的经典陷阱 —— 一律在合并前剔除。
  const overrides = Object.fromEntries(Object.entries(options ?? {}).filter(([, v]) => v !== undefined))
  const opts = { ...defaultOptions(), ...overrides }
  const warnings = []
  await ensureLayout(paths)
  const [risk, network, headlessPolicy] = await Promise.all([
    loadPolicy(paths, 'risk', warnings),
    loadPolicy(paths, 'network', warnings),
    loadPolicy(paths, 'headless', warnings),
  ])
  // 门禁阈值：**不从默认值猜**（用户裁决 #2）。缺失就是缺失，由 ops 层拒绝执行。
  const gates = await loadPolicyFile(paths.gatesFile, null, [])
  return {
    paths,
    options: opts,
    env,
    headless: isHeadless(opts, env),
    policy: {
      risk: risk.value,
      network: network.value,
      headless: headlessPolicy.value,
      gates: gates.value,
      sources: { risk: risk.source, network: network.source, headless: headlessPolicy.source, gates: gates.source },
    },
    warnings,
  }
}

export { homedir }
