/**
 * @internal 内置的 `@ai-infra/testing-core` 契约实现（**过渡件**）。
 *
 * 为什么标注 @internal：
 *   题面把精度/性能验证划给共享库 `@ai-infra/testing-core`，但**该库在本机不存在**
 *   （Step 0 已实测：npm 无此包、node_modules 无、工作区无）。用户裁决同意"内置最小实现，
 *   接口按契约设计"。因此本文件是**过渡实现**，不是最终归属：
 *     - 对外只暴露契约里的两个函数：`parityCheck` / `perfValidation`
 *     - 一旦 `@ai-infra/testing-core` 发布，只需把 provider 换成真库，调用方（quantize.mjs）零改动
 *     - 迁移脚本见 scripts/migrate-to-testing-core.mjs
 *
 * 契约版本与返回结构在本文件顶部显式声明，并被测试断言 —— 契约漂移会立刻被测出来。
 */
import { executorFor } from '../executors/index.mjs'
import '../executors/docker.mjs'
import { join } from 'node:path'
import { readText, atomicWriteJson, mkdirp } from '../util/fsx.mjs'
import { fail, warn, ERROR_CODES, WARNING_CODES } from '../errors.mjs'

/** 契约版本。与真库对接时按此版本协商。 */
export const TESTING_CORE_CONTRACT_VERSION = '1.0.0'

/** 契约声明（可被 CLI `contract` 子命令打印，也可被审计记录）。 */
export function describeContract() {
  return {
    name: '@ai-infra/testing-core',
    contractVersion: TESTING_CORE_CONTRACT_VERSION,
    implementation: 'inference-quantization/internal',
    internal: true,
    functions: {
      parityCheck: {
        purpose: '对源模型与候选模型跑同一套测试集，给出精度差异',
        inputs: ['runtime', 'testSetPath', 'baselineModelPath', 'candidateModelPath', 'limit?', 'label?'],
        returns: ['ok', 'metric', 'baseline', 'candidate', 'drop', 'samples', 'perItem', 'evidence'],
      },
      perfValidation: {
        purpose: '通过真实服务端点对比基线与候选的性能',
        inputs: ['runtime', 'baselineEndpoint', 'candidateEndpoint', 'workload', 'requests', 'concurrency'],
        returns: ['ok', 'metrics', 'baseline', 'candidate', 'gain', 'evidence'],
      },
    },
  }
}

function scriptPath(runtime, name) {
  const root = runtime.__packageRoot ?? new URL('../../', import.meta.url).pathname
  return join(root, 'docker', 'scripts', name)
}

/**
 * 精度对比。**两个模型跑同一份测试集**，指标为按 token 加权的 mean NLL / perplexity。
 *
 * 刻意返回 `perItem`（逐条 NLL）：调用方可用同一份产物做"抽样粗筛"，
 * 不必为了非最终档位再跑一次完整评测（用户裁决 #7 的成本控制）。
 */
export async function parityCheck(runtime, {
  testSetPath, baselineModelPath, candidateModelPath, limit = 0, label = 'parity', log = null,
} = {}) {
  if (!testSetPath || !baselineModelPath || !candidateModelPath) {
    fail(ERROR_CODES.E_TESTING_CORE_ERROR, 'parityCheck 需要 testSetPath / baselineModelPath / candidateModelPath', {
      remediation: '检查调用参数；这是内部契约错误，通常意味着上游没准备好测试集。',
      details: { testSetPath, baselineModelPath, candidateModelPath },
    })
  }
  const dir = join(runtime.paths.runDir, 'parity')
  await mkdirp(dir)
  const executor = executorFor(runtime)

  const one = async (modelPath, tag) => {
    const out = join(dir, `${label}-${tag}.json`)
    log?.info(`精度评测：${tag}`, { modelPath, limit, executor: executor.id })
    const r = await executor.parity(runtime, { testSetPath, modelPath, outPath: out, limit, label: tag, log })
    if (!r.ok || !r.parsed) {
      fail(ERROR_CODES.E_TESTING_CORE_ERROR, `精度评测未产出结果（${tag}，exit=${r.exitCode ?? '?'}）`, {
        remediation: '检查执行器日志：通常是模型加载失败或显存不足。',
        details: { tag, modelPath, exitCode: r.exitCode ?? null, stderr: r.stderr ?? null, stdout: r.stdout ?? null, error: r.error ?? null },
      })
    }
    const parsed = r.parsed
    if (!parsed.ok) {
      fail(ERROR_CODES.E_TESTING_CORE_ERROR, `精度评测失败（${tag}）：${parsed.stage} — ${parsed.error}`, {
        remediation: '按 details.error 定位：import/load/score 三类。若为 load，多半是量化产出与本镜像的压缩格式库不兼容。',
        details: parsed,
      })
    }
    return parsed
  }

  const base = await one(baselineModelPath, 'baseline')
  const cand = await one(candidateModelPath, 'candidate')

  // 精度损失口径：**相对困惑度退化**（更直观、与人感知一致）
  //   dropPct = (candPpl - basePpl) / basePpl * 100
  // 同时也给出 NLL 绝对差（便于排查"两边都很好但差一点"的情形）
  const basePpl = base.perplexity
  const candPpl = cand.perplexity
  const dropPct = basePpl > 0 ? ((candPpl - basePpl) / basePpl) * 100 : null
  const nllDelta = cand.meanNll - base.meanNll

  return {
    ok: true,
    metric: 'perplexity-relative-drop-pct',
    baseline: { ...measurementSide(base), perplexity: basePpl, samples: base.scoredSamples },
    candidate: { ...measurementSide(cand), perplexity: candPpl, samples: cand.scoredSamples },
    drop: { pct: dropPct, nllDelta },
    samples: Math.min(base.scoredSamples, cand.scoredSamples),
    limit,
    perItem: { baseline: base.perItem, candidate: cand.perItem },
    // ---- 测量成立性证据（供 src/gate.mjs 的 assertMeasurementValid 判）----
    // 刻意**只搬运事实**，不在这里下结论：判定要能被单独穷举测试。
    validity: { baseline: base.validity ?? null, candidate: cand.validity ?? null },
    evidence: {
      testSetPath, baselineModelPath, candidateModelPath,
      contractVersion: TESTING_CORE_CONTRACT_VERSION,
      implementation: 'inference-quantization/internal',
      baselineDurationSec: base.durationSec, candidateDurationSec: cand.durationSec,
      tools: { baseline: base.toolchain ?? null, candidate: cand.toolchain ?? null },
    },
  }
}

/** 把执行端返回的原始载荷凝练成"测量成立性判定"需要的输入。 */
function measurementSide(raw) {
  const scored = (raw.perItem ?? []).filter((p) => p.tokens > 0)
  return {
    meanNll: raw.meanNll,
    perplexity: raw.perplexity,
    vocabSize: raw.vocabSize ?? null,
    scoredSamples: raw.scoredSamples ?? scored.length,
    tokens: raw.totalTokens ?? null,
    validity: raw.validity ?? null,
    weightHealth: raw.weightHealth ?? null,
    itemMeanNlls: scored.map((p) => p.meanNll),
  }
}

/**
 * 抽样粗筛：用**同一个完整评测产物**的逐条数据，取前 N 条算一次。
 *
 * 为什么允许：完整评测在每次降级时重跑成本过高（用户裁决 #7）。
 * 约束：样本量 ≥ coarseScreenMinSamples（默认 200），阈值放宽 coarseScreenRelaxFactor（默认 2×）以避免误杀。
 * 结果写审计（`W_COARSE_SCREEN`），**最终签收必须用完整评测**。
 */
export function coarseScreen(parityResult, { minSamples = 200, relaxFactor = 2.0, take = null } = {}) {
  const n = take ?? Math.max(minSamples, Math.min(parityResult.samples, minSamples))
  const pick = (arr) => arr.filter((p) => p.tokens > 0).slice(0, n)
  const b = pick(parityResult.perItem.baseline)
  const c = pick(parityResult.perItem.candidate)

  const agg = (items) => {
    const tokens = items.reduce((s, p) => s + p.tokens, 0)
    const nll = items.reduce((s, p) => s + (p.nll ?? 0), 0)
    return { tokens, meanNll: tokens ? nll / tokens : null, samples: items.length }
  }
  const ab = agg(b)
  const ac = agg(c)
  if (!ab.meanNll || !ac.meanNll) {
    return { ok: false, reason: '可用样本不足，无法粗筛', usedSamples: Math.min(ab.samples, ac.samples) }
  }
  const basePpl = Math.exp(Math.min(ab.meanNll, 50))
  const candPpl = Math.exp(Math.min(ac.meanNll, 50))
  const dropPct = ((candPpl - basePpl) / basePpl) * 100
  return {
    ok: true,
    usedSamples: Math.min(ab.samples, ac.samples),
    dropPct,
    relaxedDropPct: dropPct,
    relaxFactor,
    baseline: { meanNll: ab.meanNll, perplexity: basePpl },
    candidate: { meanNll: ac.meanNll, perplexity: candPpl },
    note: '粗筛仅用于非最终档位的方向判断；签收必须用完整评测',
  }
}

export { warn, WARNING_CODES, atomicWriteJson }
