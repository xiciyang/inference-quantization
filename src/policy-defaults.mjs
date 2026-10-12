/**
 * 内置策略默认值（量化版）。
 *
 * 与 serving 的关系：风险分级 / 网络策略 / headless 三份**沿用 serving 已固化的文件**
 * （同一台机器、同一套资产规则），量化只补自己特有的一份：
 *   - `gates.yaml`：门禁阈值（**故意没有内置默认** —— 缺失即拒绝，见 gate.mjs）
 *   - 方法次序兜底常量（仅当方法注册表不可用时用于展示，不参与判定）
 */
import { WARNING_CODES } from './errors.mjs'

// 以下三份默认策略**与 serving 完全一致**（同一台机器、同一套资产规则）。
// 正常运行时会直接读到 serving 已固化的 ~/.ai-infra/config/*.yaml；
// 这里的常量只在文件缺失时兜底，确保 standalone 也能跑起来。

/** 方法注册表不可用时的兜底次序（只有 id 与 level，不参与判定）。 */
export const METHODS_ORDER_FALLBACK = ['awq-w4a16', 'int8-w8a8', 'fp8-dynamic', 'w8a16']

/**
 * 门禁阈值模板（**仅用于生成示例文件，不作为运行时默认**）。
 *
 * 关键：两个字段都是 null，并写明"必须显式设置，否则拒绝执行"。
 * 这样既给了人类一个可编辑入口，又不违反"不猜默认阈值"的裁决。
 */
export const GATES_TEMPLATE = {
  version: 1,
  generated: 'template',
  note: '必须显式设置这两个阈值；null 表示未设置，本插件会拒绝执行（不猜默认值）',
  max_accuracy_drop_pct: null,
  min_perf_gain_pct: null,
  near_limit_factor: 0.9,
  coarse_screen: {
    min_samples: 200,
    relax_factor: 2.0,
    note: '粗筛仅用于非最终档位；最终签收必须完整评测',
  },
}

export const DEFAULT_WARNINGS = [
  { code: WARNING_CODES.W_POLICY_DEFAULTED, message: '策略文件缺失，已使用内置默认值' },
  { code: WARNING_CODES.W_VM_SANDBOX_ABSENT, message: '本机无 VM 级沙箱，R3–R5 操作一律拒绝' },
  { code: WARNING_CODES.W_INTERNAL_SHARED_LIB, message: '@ai-infra/testing-core 不存在，正在使用包内 @internal 过渡实现' },
]

/** 风险档位（与 serving / RISK_CLASSIFICATION_DELTA.md 一致）。 */
export const RISK = Object.freeze({
  R0_READ: 'R0',
  R1_LOW: 'R1',
  R2_MEDIUM: 'R2',
  R3_HIGH: 'R3',
  R4_SYSTEM: 'R4',
  R5_PRODUCTION: 'R5',
  R6_FORBIDDEN: 'R6',
})

export const RISK_ORDER = [RISK.R0_READ, RISK.R1_LOW, RISK.R2_MEDIUM, RISK.R3_HIGH, RISK.R4_SYSTEM, RISK.R5_PRODUCTION, RISK.R6_FORBIDDEN]

export function maxRisk(a, b) {
  return RISK_ORDER.indexOf(a) >= RISK_ORDER.indexOf(b) ? a : b
}

export const DEFAULT_RISK_CLASSIFICATION = {
  version: 1,
  generated: 'builtin',
  note: '由 inference-serving 内置默认值生成；修改后请勿删除 version 字段',
  sandbox: {
    // 每个风险档要求的最低沙箱层级
    R0: 'none',
    R1: 'process',
    R2: 'container',
    R3: 'vm',
    R4: 'vm',
    R5: 'vm',
    R6: 'none',
  },
  confirmation: {
    // 确认策略：per-class-once（每类一次，TTL 内复用）/ always（每次）/ never
    R0: 'never',
    R1: 'per-class-once',
    R2: 'per-class-once',
    R3: 'always',
    R4: 'always',
    R5: 'always',
    R6: 'never',
    ttlHours: 24,
    // 确认键必须包含目标指纹 —— 同一 capability 对不同模型/环境不复用确认
    keyIncludesTarget: true,
  },
  sandbox_fallback: {
    // VM 级沙箱不存在时的行为：refuse（拒绝）/ degrade（降级到容器级）
    onUnavailable: 'refuse',
    allowDegradeWithExplicitGrant: true,
  },
  assets: {
    // 八类系统资产：写白名单。空数组 = 该资产不可写。
    filesystem: { writable: ['~/.ai-infra', '{modelRoot}'], foreignWrite: 'deny' },
    process: { ownOnly: true, protected: [], protectedContainers: ['qwen3-embedding', 'xagentlab-'], note: '只允许操作本插件登记的进程/容器；既有服务一律保护' },
    network: { policy: 'network-policy.yaml' },
    device: { writable: [], via: 'docker --gpus' },
    kernel: { writable: [] },
    account: { writable: [] },
    service: { writable: [] },
    audit: { appendOnly: true, dir: '~/.ai-infra/logs/audit', forbidDelete: true },
  },
}

export const DEFAULT_NETWORK_POLICY = {
  version: 1,
  generated: 'builtin',
  default: 'ask',
  proxy: {
    mode: 'bypass',
    envRespected: false,
    probeOnStart: true,
    probeTimeoutMs: 2000,
    probeTarget: 'https://registry.npmmirror.com',
  },
  allow: [
    { host: 'hf-mirror.com', purpose: 'HF 模型镜像', cache: true },
    { host: 'cdn-lfs.hf-mirror.com', purpose: 'HF 大文件 CDN', cache: true },
    { host: 'www.modelscope.cn', purpose: 'ModelScope 模型源', cache: true },
    { host: 'modelscope.cn', purpose: 'ModelScope 模型源', cache: true },
    { host: 'registry.npmmirror.com', purpose: 'npm registry', cache: false },
    { host: 'pypi.org', purpose: 'pip 官方源', cache: false },
    { host: 'files.pythonhosted.org', purpose: 'pip 包体', cache: false },
    { host: 'mirrors.aliyun.com', purpose: 'pip/apt 镜像', cache: false },
    { host: 'mirrors.tuna.tsinghua.edu.cn', purpose: 'apt 镜像', cache: false },
    { host: 'archive.ubuntu.com', purpose: 'apt 源', cache: false },
    { host: 'github.com', purpose: '代码与 Release', cache: false },
    { host: 'codeload.github.com', purpose: '代码归档', cache: false },
    { host: 'raw.githubusercontent.com', purpose: '原始文件', cache: false },
    { host: 'objects.githubusercontent.com', purpose: 'Release 资源', cache: false },
  ],
  deny: [
    { host: 'huggingface.co', reason: 'DNS 投毒，直连与代理均不可达（实测）；请改用 hf-mirror.com' },
    { host: 'registry.npmjs.org', reason: '直连与代理均不可达（实测）；请改用 registry.npmmirror.com' },
    { host: 'openai.com', reason: '本任务不需要，收敛出口面' },
    { host: 'anthropic.com', reason: '本任务不需要，收敛出口面' },
  ],
  local: ['127.0.0.0/8', '::1', '192.168.0.0/16', '10.0.0.0/8', '172.16.0.0/12', '.local', '.internal'],
  license: {
    permissive: ['apache-2.0', 'mit', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'cc0-1.0', 'unlicense'],
    restricted: ['llama', 'cc-by-nc', 'cc-by-nc-sa', 'research-only', 'non-commercial'],
    onMissing: 'ask',
    onRestricted: 'ask',
    onPermissive: 'allow',
  },
  ports: {
    host: '127.0.0.1',
    poolStart: 8100,
    poolEnd: 8199,
    allowPublicBind: false,
    denyPrivileged: true,
  },
}

export const DEFAULT_HEADLESS_POLICY = {
  version: 1,
  generated: 'builtin',
  mode: 'abort',
  overrides: {
    fillDefaults: 'allow',
    ambiguityResolved: 'allow-with-report',
    perClassConfirmation: 'require-grant',
    alwaysConfirmation: 'abort',
    probeMismatch: 'abort',
    degradeExhausted: 'abort',
    egressAsk: 'abort',
    licenseAsk: 'abort',
  },
}

