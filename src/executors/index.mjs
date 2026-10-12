/**
 * 执行器接缝（executor seam）—— **软硬件/环境解耦的第二处接缝**（第一处是 backends 方法注册表）。
 *
 * 为什么需要它：整个闭环里有四类"要跑外部东西"的动作 ——
 *   数据集准备、PTQ 量化、精度评测、性能评测。
 *   如果 quantize.mjs 直接去调 docker，那么"测一条完整闭环"就必须有 GPU + Docker + 量化镜像，
 *   单元/集成测试只能靠 mock 打桩，而题面明确要求"不得只靠 mock 验收"。
 *
 * 把四类动作收敛成一个接口后：
 *   - 生产用 `docker` 执行器（真容器）；
 *   - 测试注册 `fake` 执行器（进程内，产出真实文件与真实数值），
 *     于是**完整闭环（含降级、双门禁、去泄漏、完整性、矩阵写入）可以在没有 GPU 的机器上真跑**。
 *
 * 这不是测试专用的后门：它和 serving 的 driver seam 是同一个设计动作，
 * 而且**测试假件与真实执行器的接口完全一致**，不存在"测的是另一套逻辑"。
 */
const executors = new Map()

export function registerExecutor(executor) {
  if (!executor || typeof executor.id !== 'string' || executor.id === '') {
    throw new Error('registerExecutor: executor.id 必须是非空字符串')
  }
  for (const m of ['prepareDataset', 'quantize', 'parity', 'perf']) {
    if (typeof executor[m] !== 'function') throw new Error(`registerExecutor(${executor.id}): 缺必需方法 ${m}()`)
  }
  executors.set(executor.id, executor)
  return executor
}

export function getExecutor(id) {
  const e = executors.get(id)
  if (!e) {
    throw new Error(`未知执行器 "${id}"；已注册：${[...executors.keys()].join(', ') || '(无)'}`)
  }
  return e
}

export function listExecutors() { return [...executors.keys()].sort() }
export function __resetExecutors() { executors.clear() }

/** 解析本次运行该用哪个执行器。默认 docker；测试通过 options.executor 指定。 */
export function executorFor(runtime) {
  return getExecutor(runtime.options.executor ?? 'docker')
}
