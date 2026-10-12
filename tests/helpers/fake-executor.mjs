/**
 * 测试用执行器：在进程内产出**真实的文件与真实的数值**，不需要 GPU / Docker。
 *
 * 它不做假：真的写出量化产出目录（含 config.json 的 quantization_config 与权重文件）、
 * 真的算困惑度（用一个确定性的"伪 perplexity"函数，由产出的量化配置决定数值大小），
 * 真的写数据集文件。这样闭环里的每一步（去泄漏复核、产出校验、门禁判定、矩阵写入）
 * 走的都是生产代码路径。
 *
 * `knobs` 用来精确控制每一步的成败，从而确定性地驱动降级链。
 */
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { registerExecutor } from '../../src/executors/index.mjs'
import { itemHash, listDigest } from '../../src/dataset/leakage.mjs'

export const knobs = {
  quantizeFailLevels: [],       // 哪些 level（方法档位）的量化要失败
  quantizeFailureCode: 'E_QUANTIZE_FAILED',
  /** 每个方法档位对应的"精度损失"（%）：模拟"bit 越少损失越大" */
  dropByMethod: { 'awq-w4a16': 12.0, 'int8-w8a8': 4.5, 'fp8-dynamic': 3.0, 'w8a16': 1.5 },
  /** 每个方法档位对应的"性能收益"（%） */
  gainByMethod: { 'awq-w4a16': 85.0, 'int8-w8a8': 30.0, 'fp8-dynamic': 25.0, 'w8a16': 8.0 },
  degradeOutput: false,          // 产出步骤是否故意破坏（缺 quantization_config）
  prepareDatasetFails: false,
  parityFails: false,
  /** 词表大小。用于让 validity 的"均匀输出"签名可被真实触发：
   *  设成 10 且 dropPct=0 时 meanNll = ln(10) = ln(vocab) → 恰好命中退化签名。 */
  vocabSize: 151936,
  /** 置真 → 逐条 NLL 全为同一常量（模拟"与输入无关"的退化测量） */
  degenerateItemVariance: false,
  /** 置真 → 基线与候选返回逐位相同的 meanNll（模拟"测量不依赖权重"） */
  degenerateIndiscriminate: false,
  calls: [],
}

export function resetKnobs() {
  knobs.quantizeFailLevels = []
  knobs.quantizeFailureCode = 'E_QUANTIZE_FAILED'
  knobs.dropByMethod = { 'awq-w4a16': 12.0, 'int8-w8a8': 4.5, 'fp8-dynamic': 3.0, 'w8a16': 1.5 }
  knobs.gainByMethod = { 'awq-w4a16': 85.0, 'int8-w8a8': 30.0, 'fp8-dynamic': 25.0, 'w8a16': 8.0 }
  knobs.degradeOutput = false
  knobs.prepareDatasetFails = false
  knobs.parityFails = false
  knobs.vocabSize = 151936
  knobs.degenerateItemVariance = false
  knobs.degenerateIndiscriminate = false
  knobs.calls = []
}

const SOURCE_TOKENS = 20000

export const fakeExecutor = {
  id: 'fake',
  displayName: '进程内假执行器（测试用，产出真实文件与确定性数值）',

  async prepareDataset(runtime, { repo, config, split, outDir, seed, calib, test }) {
    knobs.calls.push({ op: 'prepareDataset', repo, config, split, seed, calib, test })
    if (knobs.prepareDatasetFails) return { ok: false, exitCode: 4, stderr: '注入失败：数据集不可用' }
    await fs.mkdir(outDir, { recursive: true })
    // 确定性生成不重复内容（不同 seed/calib 组合 → 不同内容，便于验证切分可复现）
    const total = calib + test
    const items = Array.from({ length: total }, (_, i) => `测试语料-${repo}-${seed}-条目${i}-${'x'.repeat(40)}`)
    const hashes = items.map(itemHash)
    const calibH = hashes.slice(0, calib)
    const testH = hashes.slice(calib)
    const write = async (name, hs) => {
      await fs.writeFile(join(outDir, name), hs.map((h, i) => JSON.stringify({ h, t: items[hs === calibH ? i : calib + i] })).join('\n') + '\n')
    }
    await write('calib.jsonl', calibH)
    await write('test.jsonl', testH)
    await fs.writeFile(join(outDir, 'lineage.json'), JSON.stringify({
      formatVersion: 1,
      source: { repo, config, split, revision: 'fake-rev', mirror: 'https://hf-mirror.com', rowsInSplit: total, textField: 'text' },
      dedup: { minChars: 200, unique: total, dropped: 0 },
      split: {
        seed, algorithm: 'sorted-sha256-keys + Random(seed).shuffle',
        calibration: { count: calibH.length, listDigest: listDigest(calibH), hashes: calibH },
        test: { count: testH.length, listDigest: listDigest(testH), hashes: testH },
      },
      leakage: { criterion: 'content-sha256-intersection-empty', intersection: 0, clean: true },
      generatedAt: new Date().toISOString(),
    }, null, 2))
    return { ok: true, exitCode: 0, stdout: JSON.stringify({ ok: true, calibration: calib, test }) }
  },

  async quantize(runtime, { sourcePath, outputPath, methodId, samples, maxSeqLen }) {
    knobs.calls.push({ op: 'quantize', methodId, outputPath, samples, maxSeqLen })
    const level = { 'awq-w4a16': 1, 'int8-w8a8': 2, 'fp8-dynamic': 3, 'w8a16': 4 }[methodId] ?? 1
    if (knobs.quantizeFailLevels.includes(level)) {
      return { ok: false, durationMs: 5, exitCode: 1, failureCode: knobs.quantizeFailureCode, stderr: `注入失败：level ${level}` }
    }
    await fs.mkdir(outputPath, { recursive: true })
    const cfg = JSON.parse(await fs.readFile(join(sourcePath, 'config.json'), 'utf8'))
    if (!knobs.degradeOutput) {
      cfg.quantization_config = {
        quant_method: 'compressed-tensors', format: 'pack-quantized',
        config_groups: { group_0: { weights: { num_bits: methodId === 'awq-w4a16' ? 4 : 8, strategy: 'group', group_size: 128 }, targets: ['Linear'] } },
        version: '0.19.0', ignore: ['lm_head'],
      }
    }
    await fs.writeFile(join(outputPath, 'config.json'), JSON.stringify(cfg))
    // 产出权重：bit 越少文件越小（模拟真实压缩）
    const bits = methodId === 'awq-w4a16' ? 4 : 8
    await fs.writeFile(join(outputPath, 'model.safetensors'), Buffer.alloc(bits === 4 ? 1024 : 1800, 3))
    // 测试假件内部用：记录方法 id，供 parity 决定这一档的精度损失
    await fs.writeFile(join(outputPath, '.fake-method'), methodId)
    return {
      ok: true, durationMs: 42, exitCode: 0,
      parsed: { ok: true, method: methodId, checks: { hasConfig: true, hasQuantizationConfig: !knobs.degradeOutput, weightFileCount: 1 }, toolchain: { llmcompressor: 'fake' } },
    }
  },

  async parity(runtime, { modelPath, limit, label }) {
    knobs.calls.push({ op: 'parity', modelPath, limit, label })
    if (knobs.parityFails) return { ok: false, exitCode: 12, stderr: '注入失败：精度评测不可用' }
    const cfgRaw = await fs.readFile(join(modelPath, 'config.json'), 'utf8').catch(() => null)
    let qc = null
    if (cfgRaw) { try { qc = JSON.parse(cfgRaw).quantization_config } catch { qc = null } }
    const bits = qc?.config_groups?.group_0?.weights?.num_bits ?? 16
    const isBaseline = qc === null
    // 候选：按方法档位给出确定性的精度损失（由 knobs 控制，便于逐档验证降级链）
    let dropPct = 0
    if (!isBaseline) {
      const methodId = await fs.readFile(join(modelPath, '.fake-method'), 'utf8').catch(() => null)
      dropPct = (methodId && knobs.dropByMethod[methodId] !== undefined)
        ? knobs.dropByMethod[methodId]
        : Math.max(0, (16 - bits) * 2)
    }
    const ppl = 10.0 * (1 + dropPct / 100)
    const targetNll = Math.log(ppl)
    const n = limit > 0 ? limit : 400
    // 逐条 NLL **必须有方差**：真实模型在不同文档上的 loss 不可能逐位相同。
    // 夹具若给常量，就是在喂一个"与输入无关"的测量 —— 那正是 validity 门禁要拦下的东西。
    // 偏移量对基线与候选**完全相同**，因此在 ppl 比值里精确抵消，drop% 仍等于配置值。
    const perItem = Array.from({ length: n }, (_, i) => {
      const v = knobs.degenerateItemVariance
        ? targetNll
        : targetNll + 0.15 * Math.sin(i * 1.7) + 0.05 * Math.cos(i * 0.3)
      return { i, tokens: 50, nll: v * 50, meanNll: v }
    })
    if (knobs.degenerateIndiscriminate) for (const p of perItem) { p.meanNll = 1.0; p.nll = 50.0 }
    const meanNll = perItem.reduce((s, p) => s + p.nll, 0) / (n * 50)
    const vocabSize = knobs.vocabSize
    return {
      ok: true,
      parsed: {
        ok: true, label, model: modelPath, samples: n, scoredSamples: n, totalTokens: n * 50,
        meanNll, perplexity: Math.exp(meanNll), durationSec: 1, limit, maxLen: 2048,
        vocabSize,
        weightHealth: { available: true, tensorCount: 10, zeroTensorCount: 0, nonFiniteTensorCount: 0 },
        validity: { valid: true, reasons: [], uniformTol: 1e-4, checks: { vocabSize, zeroTensorCount: 0 } },
        perItem,
      },
    }
  },

  async perf(runtime, { baselineModelPath, candidateModelPath, workload }) {
    knobs.calls.push({ op: 'perf', baselineModelPath, candidateModelPath, workload })
    const cfgRaw = await fs.readFile(join(candidateModelPath, 'config.json'), 'utf8').catch(() => null)
    void cfgRaw
    const methodId = await fs.readFile(join(candidateModelPath, '.fake-method'), 'utf8').catch(() => null)
    const gain = (methodId && knobs.gainByMethod[methodId] !== undefined) ? knobs.gainByMethod[methodId] : 0
    const baseTps = 40.0
    const make = (tps, tag) => ({
      endpointUrl: `http://127.0.0.1:0/${tag}`, modelName: tag, workload,
      requests: 8, succeeded: 8, concurrency: 2, wallMs: 1000, totalTokens: 512,
      ttftMs: { median: 100, mean: 100 }, latencyMs: { median: 500, mean: 500 },
      decodeTps: { median: tps, mean: tps }, throughputTps: tps * 2,
    })
    const b = make(baseTps, 'baseline')
    const c = make(baseTps * (1 + gain / 100), 'candidate')
    return {
      ok: true,
      metrics: { primary: 'decode-tps-gain-pct', decodeTpsGainPct: gain, throughputGainPct: gain, ttftReductionPct: 0 },
      baseline: b, candidate: c,
      gain: { decodeTpsPct: gain, throughputPct: gain, ttftReductionPct: 0 },
      evidence: { workload, requests: 8, concurrency: 2, implementation: 'fake' },
    }
  },
}

registerExecutor(fakeExecutor)
export default fakeExecutor
export { SOURCE_TOKENS }
