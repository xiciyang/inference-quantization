/**
 * 指标（验收标准 5.4.5：部署耗时、资源消耗、降级次数）。
 *
 * 落盘在 `~/.ai-infra/state/metrics.json`；读-改-写走注册表同一把锁，避免并发覆盖。
 * 指标**不是**安全属性：写失败只记警告，不阻断业务。
 */
import { withLock } from './util/lock.mjs'
import { atomicWriteJson, readText } from './util/fsx.mjs'

const EMPTY = {
  version: 1,
  counters: {
    quantize_attempts: 0,
    quantize_success: 0,
    quantize_failed: 0,
    quantize_qat_plan: 0,
    degrade_steps: 0,
    gate_pass: 0,
    gate_fail: 0,
    gate_near_limit: 0,
    rollback: 0,
    leakage_checks: 0,
    leakage_rejections: 0,
    integrity_checks: 0,
    confirmations_granted: 0,
    sandbox_refusals: 0,
    egress_denied: 0,
    tool_calls: 0,
    tool_errors: 0,
  },
  timings: {
    quantize_ms_total: 0,
    quantize_ms_max: 0,
    parity_ms_total: 0,
    perf_ms_total: 0,
  },
  resources: {
    last_method: null,
    last_compression_ratio: null,
    last_output_bytes: null,
    last_image: null,
    last_samples: null,
  },
  degrade: {
    by_step: {},
    by_reason: {},
  },
  updatedAt: null,
}

async function read(runtime) {
  const raw = await readText(runtime.paths.metricsFile, null)
  if (!raw) return structuredClone(EMPTY)
  try {
    const parsed = JSON.parse(raw)
    return {
      ...structuredClone(EMPTY),
      ...parsed,
      counters: { ...EMPTY.counters, ...(parsed.counters ?? {}) },
      timings: { ...EMPTY.timings, ...(parsed.timings ?? {}) },
      resources: { ...EMPTY.resources, ...(parsed.resources ?? {}) },
      degrade: { ...EMPTY.degrade, ...(parsed.degrade ?? {}) },
    }
  } catch {
    return structuredClone(EMPTY)
  }
}

export async function readMetrics(runtime) {
  return read(runtime)
}

/** 以一次事务更新指标。mutator(m) 直接改对象。 */
export async function updateMetrics(runtime, mutator) {
  try {
    return await withLock(runtime.paths.lockDir, 'metrics', async () => {
      const m = await read(runtime)
      mutator(m)
      m.updatedAt = new Date().toISOString()
      await atomicWriteJson(runtime.paths.metricsFile, m)
      return m
    })
  } catch (err) {
    runtime.warnings.push({ code: 'W_METRICS_WRITE_FAILED', message: String(err.message), details: null })
    return null
  }
}

export function bump(runtime, name, by = 1) {
  return updateMetrics(runtime, (m) => { m.counters[name] = (m.counters[name] ?? 0) + by })
}

export function recordQuantizeTiming(runtime, ms, { success }) {
  return updateMetrics(runtime, (m) => {
    m.counters.quantize_attempts += 1
    m.timings.quantize_ms_total += ms
    m.timings.quantize_ms_max = Math.max(m.timings.quantize_ms_max, ms)
    if (success) m.counters.quantize_success += 1
    else m.counters.quantize_failed += 1
  })
}

export function recordGate(runtime, { passed, decision }) {
  return updateMetrics(runtime, (m) => {
    if (passed) m.counters.gate_pass += 1
    else m.counters.gate_fail += 1
    if (decision === 'confirm-near-limit') m.counters.gate_near_limit += 1
  })
}

export function recordLeakageCheck(runtime, { clean }) {
  return updateMetrics(runtime, (m) => {
    m.counters.leakage_checks += 1
    if (!clean) m.counters.leakage_rejections += 1
  })
}

export function recordDegrade(runtime, { step, reason }) {
  return updateMetrics(runtime, (m) => {
    m.counters.degrade_steps += 1
    m.degrade.by_step[String(step)] = (m.degrade.by_step[String(step)] ?? 0) + 1
    const r = reason ?? 'unknown'
    m.degrade.by_reason[r] = (m.degrade.by_reason[r] ?? 0) + 1
  })
}



export function recordResource(runtime, { method, compressionRatio, outputBytes, image, samples }) {
  return updateMetrics(runtime, (m) => {
    if (method !== undefined) m.resources.last_method = method
    if (compressionRatio !== undefined) m.resources.last_compression_ratio = compressionRatio
    if (outputBytes !== undefined) m.resources.last_output_bytes = outputBytes
    if (image !== undefined) m.resources.last_image = image
    if (samples !== undefined) m.resources.last_samples = samples
  })
}

export function avgQuantizeMs(m) {
  const n = m.counters.quantize_attempts || 1
  return Math.round(m.timings.quantize_ms_total / n)
}

export { EMPTY as EMPTY_METRICS }
