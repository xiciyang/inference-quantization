/**
 * docker 执行器：把四类外部动作落到真实容器上。
 *
 * 所有动作都是**一次性容器**（--rm），并在源模型目录上使用 **:ro** 只读挂载 ——
 * 这是"原模型只读"的实现层保证（逻辑层还有哈希校验兜底）。
 */
import { join } from 'node:path'
import { run } from '../util/proc.mjs'
import { readText, mkdirp } from '../util/fsx.mjs'
import { registerExecutor } from './index.mjs'

function pkgRoot(runtime) {
  return runtime.__packageRoot ?? new URL('../../', import.meta.url).pathname
}

function baseArgs(runtime, purpose) {
  return [
    'run', '--rm',
    '--label', 'inference-quantization.managed=true',
    '--label', `inference-quantization.purpose=${purpose}`,
    '--runtime', runtime.options.containerRuntime, '--gpus', 'all',
  ]
}

export const dockerExecutor = {
  id: 'docker',
  displayName: '真实容器（docker + nvidia runtime）',

  async prepareDataset(runtime, { repo, config, split, outDir, seed, calib, test, log, signal }) {
    const script = join(pkgRoot(runtime), 'docker', 'scripts', 'prepare-dataset.py')
    await mkdirp(outDir)
    // 持久化 HF 缓存：本机到 hf-mirror 的连接**实测会抖**（握手超时），
    // 把缓存挂进来，重试与后续 run 就不必重新下载。
    const hfCache = join(runtime.paths.cacheDir, 'hf')
    await mkdirp(hfCache)
    const args = [
      ...baseArgs(runtime, 'prepare-dataset'),
      '-e', `HF_HOME=${hfCache}`,
      '-e', `HF_DATASETS_CACHE=${join(hfCache, 'datasets')}`,
      '-v', `${hfCache}:${hfCache}`,
      '-v', `${outDir}:/work/out`,
      '-v', `${script}:/scripts/prepare-dataset.py:ro`,
      '--entrypoint', 'python3',
      runtime.options.quantImage,
      '/scripts/prepare-dataset.py',
      '--repo', repo, '--split', split, '--out', '/work/out',
      '--seed', String(seed), '--calib', String(calib), '--test', String(test),
    ]
    if (config) args.splice(args.indexOf('--split'), 0, '--config', config)
    const r = await run('docker', args, { timeoutMs: Math.min(runtime.options.quantTimeoutMs, 1_800_000), signal })
    log?.info('数据集准备容器结束', { exitCode: r.code, ms: r.durationMs })
    return { ok: r.code === 0, exitCode: r.code, stdout: r.stdout.slice(-2000), stderr: r.stderr.slice(-1500) }
  },

  async quantize(runtime, { sourcePath, outputPath, methodId, calibPath, samples, maxSeqLen, seed, log, signal }) {
    const script = join(pkgRoot(runtime), 'docker', 'scripts', 'quantize.py')
    await mkdirp(outputPath)
    const resultPath = join(runtime.paths.runDir, `quant-${methodId}-${Date.now()}.json`)
    await mkdirp(runtime.paths.runDir)
    const args = [
      ...baseArgs(runtime, 'quantize'),
      '--label', `inference-quantization.method=${methodId}`,
      '-v', `${sourcePath}:${sourcePath}:ro`,
      // 校准/测试集目录必须挂进来（迁移到执行器时曾漏挂 → 容器读不到 calib.jsonl，
      // 四级全部 E_QUANTIZE_FAILED 且只耗时 3-6 秒，一眼就能看出不是真在跑）
      '-v', `${runtime.paths.datasetDir}:${runtime.paths.datasetDir}:ro`,
      '-v', `${outputPath}:${outputPath}`,
      '-v', `${runtime.paths.runDir}:${runtime.paths.runDir}`,
      '-v', `${script}:/scripts/quantize.py:ro`,
      '--entrypoint', 'python3',
      runtime.options.quantImage,
      '/scripts/quantize.py',
      '--model', sourcePath, '--calib', calibPath, '--out', outputPath,
      '--method', methodId, '--samples', String(samples), '--max-seq-len', String(maxSeqLen),
      '--seed', String(seed), '--result', resultPath,
    ]
    const started = Date.now()
    const r = await run('docker', args, { timeoutMs: runtime.options.quantTimeoutMs, signal })
    const raw = await readText(resultPath, null)
    let parsed = null
    if (raw) { try { parsed = JSON.parse(raw) } catch { parsed = null } }
    return {
      ok: Boolean(parsed?.ok), durationMs: Date.now() - started, exitCode: r.code,
      timedOut: r.timedOut, parsed,
      failureCode: r.timedOut ? 'E_QUANTIZE_TIMEOUT' : (parsed?.ok ? null : 'E_QUANTIZE_FAILED'),
      stderr: r.stderr.slice(-2000),
    }
  },

  async parity(runtime, { testSetPath, modelPath, outPath, limit, label, log, signal }) {
    const script = join(pkgRoot(runtime), 'docker', 'scripts', 'parity.py')
    await mkdirp(join(outPath, '..'))
    const args = [
      ...baseArgs(runtime, 'parity'),
      '-v', `${testSetPath}:${testSetPath}:ro`,
      '-v', `${modelPath}:${modelPath}:ro`,
      '-v', `${join(outPath, '..')}:${join(outPath, '..')}`,
      '-v', `${script}:/scripts/parity.py:ro`,
      '--entrypoint', 'python3',
      runtime.options.quantImage,
      '/scripts/parity.py',
      '--model', modelPath, '--test', testSetPath, '--out', outPath,
      '--limit', String(limit), '--label', label,
    ]
    const r = await run('docker', args, { timeoutMs: runtime.options.evalTimeoutMs, signal })
    const raw = await readText(outPath, null)
    if (!raw) return { ok: false, exitCode: r.code, stderr: r.stderr.slice(-1500), stdout: r.stdout.slice(-1500) }
    try { return { ok: true, parsed: JSON.parse(raw) } } catch (e) { return { ok: false, error: `结果解析失败：${e.message}` } }
  },

  /**
   * 性能必须走**真实端点**：经 MCP 让 inference-serving 把基线与候选各起一个端点，
   * 再对两个端点发同一份负载。这正是"部署验证依赖 serving"的落点。
   */
  async perf(runtime, { baselineModelPath, candidateModelPath, workload, requests, concurrency, log }) {
    const { withServing } = await import('../serving-client.mjs')
    const { perfValidation } = await import('../testing/perf.mjs')
    const out = await withServing(runtime, async (client) => perfValidation(runtime, {
      client, baselineModelPath, candidateModelPath, workload, requests, concurrency, log,
    }), { timeoutMs: runtime.options.evalTimeoutMs })
    return out
  },
}

registerExecutor(dockerExecutor)
export default dockerExecutor
