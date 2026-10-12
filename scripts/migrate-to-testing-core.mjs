#!/usr/bin/env node
/**
 * 迁移脚本：把包内 `@internal` 的 testing-core 过渡实现换成真库 `@ai-infra/testing-core`。
 *
 * 为什么需要它：题面把精度/性能验证划给共享库，但该库当前不存在，我们内置了最小实现。
 * 一旦真库发布，调用方（src/quantize.mjs）不应改一行 —— 所以这里把"要改什么"固化成脚本，
 * 避免将来靠记忆。
 *
 * 用法：
 *   node scripts/migrate-to-testing-core.mjs --check          # 真库是否已可用
 *   node scripts/migrate-to-testing-core.mjs --apply          # 生成 provider 适配层（不删旧实现）
 */
import { promises as fs } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const args = process.argv.slice(2)

async function probe() {
  try {
    const mod = await import('@ai-infra/testing-core')
    return { available: true, exports: Object.keys(mod) }
  } catch (err) {
    return { available: false, error: String(err.message).slice(0, 200) }
  }
}

const ADAPTER = `/**
 * provider 适配层：优先用真库 @ai-infra/testing-core，缺失时回落到包内 @internal 实现。
 *
 * 由 scripts/migrate-to-testing-core.mjs 生成。调用方只 import 本文件，
 * 因此"用真库还是用过渡实现"对上游透明。
 */
let real = null
try {
  real = await import('@ai-infra/testing-core')
} catch {
  real = null
}

const internal = await import('./parity.mjs')
const internalPerf = await import('./perf.mjs')

export const provider = real ? '@ai-infra/testing-core' : 'inference-quantization/internal'
export const usingInternal = real === null

export const parityCheck = real?.parityCheck ?? internal.parityCheck
export const perfValidation = real?.perfValidation ?? internalPerf.perfValidation
export const coarseScreen = internal.coarseScreen   // 粗筛是本插件特有的成本控制，真库不一定提供
export const TESTING_CORE_CONTRACT_VERSION = real?.CONTRACT_VERSION ?? internal.TESTING_CORE_CONTRACT_VERSION
`

async function main() {
  const p = await probe()
  if (args.includes('--check') || args.length === 0) {
    console.log(JSON.stringify({
      realLibraryAvailable: p.available,
      exports: p.exports ?? null,
      error: p.error ?? null,
      currentImplementation: 'inference-quantization/internal (@internal)',
      action: p.available ? '可以执行 --apply 生成适配层' : '真库仍不可用，保持 @internal 实现',
    }, null, 2))
    return 0
  }
  if (args.includes('--apply')) {
    const out = join(root, 'src', 'testing', 'provider.mjs')
    await fs.writeFile(out, ADAPTER)
    console.log(JSON.stringify({
      ok: true, wrote: out, realLibraryAvailable: p.available,
      nextSteps: [
        '把 src/quantize.mjs 里对 ./testing/parity.mjs 与 ./testing/perf.mjs 的 import 改为 ./testing/provider.mjs',
        '重跑 npm test（契约测试会验证替换后行为一致）',
        '确认无误后再删除 parity.mjs / perf.mjs 中的重复实现',
      ],
    }, null, 2))
    return 0
  }
  console.error('用法：--check | --apply')
  return 2
}

process.exit(await main())
