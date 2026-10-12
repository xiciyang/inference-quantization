/**
 * DSH 插件入口（bundle 形态）—— 与 serving 同构。
 *
 * 契约：导出 `name` + `apply(ctx, config)`；服务依赖声明在 `inject`；一切注册走 `ctx.effect()`。
 * 只硬依赖 `tools`；`webServer` 用 `ctx.get('webServer')` 可选取用（headless profile 也能挂载）。
 *
 * 与 DSH 契约的一处刻意偏离（同 serving）：本包是零运行时依赖包，不引入
 * `@deepseek-ai/schemastery`，配置改为手写校验并在非法时**加载期响亮失败**（语义等价）。
 */
import { ALL_TOOL_SPECS, toDshTool } from './tools.mjs'
import { drain, inFlightCount } from './ops.mjs'
import { createRuntime } from './config.mjs'
import { probeProxy } from './netpolicy.mjs'
import { getSandboxProbe } from './sandbox.mjs'
import { createLogger } from './log.mjs'
import { allMethods } from './backends/index.mjs'
import { readImageMeta } from './image.mjs'
import { PACKAGE_VERSION, MCP_INTERFACE_VERSION } from './version.mjs'

export const name = 'inference-quantization'
export const inject = ['tools']

const ALLOWED_CONFIG_KEYS = new Set([
  'root', 'headless', 'modelRoots', 'containerRuntime', 'quantImage', 'outputRoot',
  'calibrationSamples', 'calibrationSeed', 'datasetRepo', 'datasetConfig', 'datasetSplit',
  'maxDegradeSteps', 'quantTimeoutMs', 'evalTimeoutMs', 'drainTimeoutMs',
  'servingPath', 'exposeAdminTools', 'registerHttpRoute', 'logLevel',
])

export function validateConfig(config) {
  const cfg = config ?? {}
  if (typeof cfg !== 'object' || Array.isArray(cfg)) throw new Error('[inference-quantization] config 必须是对象')
  const unknown = Object.keys(cfg).filter((k) => !ALLOWED_CONFIG_KEYS.has(k))
  if (unknown.length > 0) {
    throw new Error(`[inference-quantization] 未知配置键：${unknown.join(', ')}。允许的键：${[...ALLOWED_CONFIG_KEYS].join(', ')}`)
  }
  if (cfg.root !== undefined && typeof cfg.root !== 'string') throw new Error('[inference-quantization] root 必须是字符串路径')
  if (cfg.headless !== undefined && typeof cfg.headless !== 'boolean') throw new Error('[inference-quantization] headless 必须是布尔值')
  if (cfg.maxDegradeSteps !== undefined && (!Number.isInteger(cfg.maxDegradeSteps) || cfg.maxDegradeSteps < 1)) {
    throw new Error('[inference-quantization] maxDegradeSteps 必须是 ≥1 的整数')
  }
  return cfg
}

export function apply(ctx, config) {
  const cfg = validateConfig(config)
  const options = {}
  for (const k of ['headless', 'containerRuntime', 'quantImage', 'outputRoot', 'calibrationSamples', 'calibrationSeed',
    'datasetRepo', 'datasetConfig', 'datasetSplit', 'maxDegradeSteps', 'quantTimeoutMs', 'evalTimeoutMs',
    'drainTimeoutMs', 'servingPath']) {
    if (cfg[k] !== undefined) options[k] = cfg[k]
  }
  if (cfg.modelRoots) options.modelRoots = cfg.modelRoots

  const specs = cfg.exposeAdminTools === false
    ? ALL_TOOL_SPECS.filter((t) => !t.name.includes('dataset_') && !t.name.includes('image_') && !t.name.includes('metrics') && !t.name.includes('audit_') && !t.name.includes('doctor'))
    : ALL_TOOL_SPECS

  const registered = []
  const failed = []
  for (const spec of specs) {
    ctx.effect(() => {
      try {
        ctx.tools.register(toDshTool(spec))
        registered.push(spec.name)
      } catch (err) {
        failed.push({ name: spec.name, error: String(err?.message ?? err) })
        ctx.logger?.('inference-quantization')?.warn?.(`工具注册失败：${spec.name} — ${err?.message ?? err}`)
        return () => {}
      }
      return () => {}
    }, `inference-quantization: register ${spec.name}`)
  }

  // 启动自检（只读，不阻断挂载）
  ctx.effect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const runtime = await createRuntime({ root: cfg.root, options })
        runtime.__packageRoot = new URL('../', import.meta.url).pathname
        const log = createLogger(runtime, { level: cfg.logLevel ?? null })
        if (cancelled) return
        const sandbox = await getSandboxProbe()
        const proxy = await probeProxy(runtime)
        const imageMeta = await readImageMeta(runtime, runtime.options.quantImage)
        log.info('inference-quantization 已挂载', {
          packageVersion: PACKAGE_VERSION, mcpInterface: MCP_INTERFACE_VERSION,
          tools: registered.length, methods: allMethods().map((m) => m.id),
          sandboxTiers: { process: sandbox.process.available, container: sandbox.container.available, vm: sandbox.vm.available },
          proxyEnv: proxy.envProxy ?? null, proxyRespected: false,
          policySources: runtime.policy.sources, headless: runtime.headless,
          imageVerified: imageMeta?.verified ?? false,
        })
        await log.flush()
        const lg = ctx.logger?.('inference-quantization')
        lg?.info?.(`量化方法：${allMethods().map((m) => `${m.id}(L${m.level})`).join(', ')}`)
        if (!imageMeta) lg?.info?.(`量化镜像尚未验证（${runtime.options.quantImage}）：首次量化前请先跑 image verify`)
        else if (!imageMeta.verified) lg?.warn?.(`量化镜像验证未通过（缺：${imageMeta.missing.join('；')}）：未显式授权时量化会被拒绝`)
        if (!sandbox.vm.available) lg?.info?.('本机无 VM 级沙箱：QAT（R3）一律拒绝；只产出计划')
      } catch (err) {
        ctx.logger?.('inference-quantization')?.warn?.(`启动自检失败（不影响工具可用性）：${err?.message ?? err}`)
      }
    })()
    return () => { cancelled = true }
  }, 'inference-quantization: startup self-check')

  const webServer = ctx.get?.('webServer')
  if (webServer && cfg.registerHttpRoute !== false) {
    ctx.effect(() => webServer.register({
      kind: 'http',
      path: '/api/inference-quantization/status',
      handler: async (req, res) => {
        try {
          const { doctor } = await import('./api.mjs')
          const out = await doctor({ deep: false }, { root: cfg.root })
          const body = JSON.stringify({ package: PACKAGE_VERSION, inFlight: inFlightCount(), ...out })
          res.statusCode = out?.ok === false ? 503 : 200
          res.setHeader('content-type', 'application/json; charset=utf-8')
          res.setHeader('cache-control', 'no-store')
          res.end(body)
        } catch (err) {
          try { res.statusCode = 500; res.end(JSON.stringify({ ok: false, message: String(err?.message ?? err) })) } catch { /* 连接可能已断 */ }
        }
      },
    }), 'inference-quantization: register /api/inference-quantization/status')
  }

  ctx.effect(() => () => drain({ timeoutMs: options.drainTimeoutMs ?? 15_000 }).then((r) => {
    if (!r.drained) ctx.logger?.('inference-quantization')?.warn?.(`卸载时仍有 ${r.remaining} 个操作未完成（已等待 ${r.waitedMs}ms）`)
  }), 'inference-quantization: drain on dispose')

  return { registered: () => [...registered], failed: () => [...failed] }
}

export { ALL_TOOL_SPECS, toDshTool }
export default { name, inject, apply, validateConfig }
