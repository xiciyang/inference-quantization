/**
 * 量化方法注册表 —— **量化方法插件化的接缝**（题目追加原则）。
 *
 * 核心流程（quantize.mjs）不认识任何具体方法；它只从本注册表取一个 method 对象，
 * 调用它的 `recipe()` / `describe()` / `preflight()`。新增方法 = 新增一个模块并 `register()`，
 * **核心零改动**（与 serving 的后端接缝同构）。
 *
 * 每个方法自报 `level`（在降级链中的档位）与 `bits`（权重位宽），
 * 使降级链与"量化矩阵"能自动获得这些信息，而不是在别处维护一张对照表。
 */
const methods = new Map()

const REQUIRED = ['recipe', 'describe', 'preflight']

export function register(method) {
  if (!method || typeof method.id !== 'string' || method.id === '') {
    throw new Error('register(method): method.id 必须是非空字符串')
  }
  for (const m of REQUIRED) {
    if (typeof method[m] !== 'function') throw new Error(`register(${method.id}): 缺必需方法 ${m}()`)
  }
  if (!Number.isInteger(method.level) || method.level < 1) {
    throw new Error(`register(${method.id}): level 必须是 ≥1 的整数（降级链档位）`)
  }
  if (methods.has(method.id)) {
    throw new Error(`register(${method.id}): 该 id 已注册（重复注册会掩盖加载顺序问题，直接拒绝）`)
  }
  methods.set(method.id, method)
  return method
}

export function unregister(id) { return methods.delete(id) }
export function getMethod(id) { return methods.get(id) ?? null }
export function listMethods() { return [...methods.keys()] }
export function allMethods() { return [...methods.values()].sort((a, b) => a.level - b.level) }

/** 按降级链档位排序的方法序列（level 1..N）。 */
export function chainOrdered() {
  return allMethods().filter((m) => m.inDegradeChain !== false)
}

/** 在给定环境下可用的方法（只问方法自己"你能不能跑"）。 */
export function selectableMethods(env) {
  return allMethods().filter((m) => m.isAvailable(env)).map((m) => m.id)
}

export function __resetMethods() { methods.clear() }
