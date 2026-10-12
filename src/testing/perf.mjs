/**
 * 性能验证（testing-core 契约的第二半）。
 *
 * **必须走真实服务端点**（这与精度评测相反）：
 *   性能收益来自"量化后权重变小 → 显存带宽压力下降 → 解码更快"，
 *   这个效应只有在真实的推理引擎 + 真实 HTTP 路径上才成立。
 *   直接调 transformers 测不出任何有意义的收益。
 *
 * 所以本模块通过 MCP 让 inference-serving 把两个模型各起一个端点，
 * 再对两个端点发同样的负载。
 *
 * 指标（全部按 token 计，不按请求计 —— 请求长度不同会让后者失真）：
 *   ttftMs        首 token 延迟（中位数）
 *   decodeTps     解码吞吐 = 生成 token 数 / (总时长 − TTFT)，单位 tok/s/请求
 *   throughputTps 端到端聚合吞吐 = 总 token / 总墙钟（含并发增益）
 *   latencyMs     端到端延迟（中位数）
 */
import https from 'node:https'
import http from 'node:http'
import { fail, warn, ERROR_CODES, WARNING_CODES } from '../errors.mjs'
import { callWithConfirm } from '../serving-client.mjs'

/* --------------------------------------------------------------- HTTP/SSE */

function postStream(url, payload, { timeoutMs = 120_000 } = {}) {
  const u = new URL(url)
  const mod = u.protocol === 'https:' ? https : http
  const body = JSON.stringify(payload)
  const startedAt = Date.now()
  return new Promise((resolvePromise) => {
    const req = mod.request({
      protocol: u.protocol, hostname: u.hostname, port: u.port || 80,
      path: `${u.pathname}${u.search}`, method: 'POST',
      headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body), accept: 'text/event-stream' },
    }, (res) => {
      let firstTokenAt = null
      let tokens = 0
      let text = ''
      let buf = ''
      let done = false
      const finish = (err) => {
        if (done) return
        done = true
        clearTimeout(timer)
        resolvePromise({
          ok: !err && res.statusCode === 200,
          status: res.statusCode,
          error: err ? String(err.message ?? err) : null,
          ttftMs: firstTokenAt ? firstTokenAt - startedAt : null,
          totalMs: Date.now() - startedAt,
          tokens,
          text,
        })
      }
      const timer = setTimeout(() => { try { req.destroy() } catch { /* noop */ } ; finish(new Error('ETIMEDOUT')) }, timeoutMs)
      timer.unref?.()

      res.on('data', (chunk) => {
        buf += chunk.toString('utf8')
        let idx
        while ((idx = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, idx).trim()
          buf = buf.slice(idx + 1)
          if (!line.startsWith('data:')) continue
          const data = line.slice(5).trim()
          if (data === '[DONE]') { finish(null); return }
          try {
            const j = JSON.parse(data)
            const ch = j.choices?.[0]
            if (!ch) continue
            const piece = ch.text ?? ''
            if (piece) {
              if (firstTokenAt === null) firstTokenAt = Date.now()
              tokens += 1
              text += piece
            }
          } catch { /* 忽略非 JSON 行 */ }
        }
      })
      res.on('end', () => finish(null))
      res.on('error', (e) => finish(e))
    })
    req.on('error', (e) => { resolvePromise({ ok: false, status: null, error: String(e.message ?? e), ttftMs: null, totalMs: Date.now() - startedAt, tokens: 0, text: '' }) })
    req.write(body)
    req.end()
  })
}

/* ------------------------------------------------------------------- 统计 */

export function median(xs) {
  const a = xs.filter((x) => typeof x === 'number' && Number.isFinite(x)).sort((x, y) => x - y)
  if (a.length === 0) return null
  const m = Math.floor(a.length / 2)
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2
}

export function mean(xs) {
  const a = xs.filter((x) => typeof x === 'number' && Number.isFinite(x))
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null
}

/* -------------------------------------------------------------- 负载执行 */

const WORKLOADS = {
  'chat-short': { prompt: 'Explain what a KV cache is in one short paragraph.', maxTokens: 64 },
  'chat-medium': { prompt: 'Write a 200-word explanation of how quantization reduces inference cost, covering weight-only vs activation quantization.', maxTokens: 160 },
  'decode-heavy': { prompt: 'Count from 1 to 50, one number per line.', maxTokens: 256 },
}

export async function measureEndpoint(endpointUrl, modelName, {
  workload = 'chat-medium', requests = 8, concurrency = 2, warmup = 1, timeoutMs = 120_000, log = null,
} = {}) {
  const spec = WORKLOADS[workload] ?? WORKLOADS['chat-medium']
  const doOne = () => postStream(`${endpointUrl}/v1/completions`, {
    model: modelName, prompt: spec.prompt, max_tokens: spec.maxTokens, temperature: 0, stream: true,
  }, { timeoutMs })

  for (let i = 0; i < warmup; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    await doOne()
  }
  log?.info('性能测量：预热完成，开始计数', { endpointUrl, workload, requests, concurrency })

  const wallStart = Date.now()
  const results = []
  let inflight = 0
  let idx = 0
  await new Promise((resolvePromise) => {
    const pump = () => {
      if (idx >= requests && inflight === 0) { resolvePromise(); return }
      while (inflight < concurrency && idx < requests) {
        idx += 1
        inflight += 1
        doOne().then((r) => {
          results.push(r)
          inflight -= 1
          pump()
        })
      }
    }
    pump()
  })
  const wallMs = Date.now() - wallStart

  const ok = results.filter((r) => r.ok && r.tokens > 0)
  if (ok.length === 0) {
    fail(ERROR_CODES.E_TESTING_CORE_ERROR, `性能测量未得到任何成功响应：${results[0]?.error ?? 'unknown'}`, {
      remediation: '检查端点是否就绪、模型名是否匹配、max_tokens 是否过小。',
      details: { endpointUrl, modelName, firstError: results[0] ?? null },
    })
  }
  const totalTokens = ok.reduce((s, r) => s + r.tokens, 0)
  const decodeTimes = ok.map((r) => Math.max(1, r.totalMs - (r.ttftMs ?? 0)))

  return {
    endpointUrl, modelName, workload,
    requests: results.length,
    succeeded: ok.length,
    concurrency,
    wallMs,
    totalTokens,
    ttftMs: { median: median(ok.map((r) => r.ttftMs).filter((x) => x !== null)), mean: mean(ok.map((r) => r.ttftMs).filter((x) => x !== null)) },
    latencyMs: { median: median(ok.map((r) => r.totalMs)), mean: mean(ok.map((r) => r.totalMs)) },
    decodeTps: { median: median(ok.map((r, i) => (r.tokens / decodeTimes[i]) * 1000)), mean: mean(ok.map((r, i) => (r.tokens / decodeTimes[i]) * 1000)) },
    throughputTps: (totalTokens / wallMs) * 1000,
  }
}

/* ------------------------------------------------------------ 部署 + 对比 */

/** 通过 MCP 让 serving 起一个端点；失败时把 serving 的结构化错误原样抛出。 */
export async function deployViaServing(client, { modelPath, port = null, servedModelName, extra = {}, log = null, ownerToken = null } = {}) {
  const args = {
    model: modelPath,
    ...(port ? { port } : {}),
    ...(servedModelName ? { servedModelName } : {}),
    // ── 端点租约（F-3）────────────────────────────────────────────────
    // serving 按 model 给端点做键：基线与候选是**不同模型**，所以各自独立租；
    // 但同一次运行内如果对同一模型再 deploy（例如重试），必须复用同一个 token，
    // 否则会被 E_ENDPOINT_LEASED 拒绝。
    owner: 'inference-quantization',
    ...(ownerToken ? { ownerToken } : {}),
    ...extra,
  }
  log?.info('通过 MCP 请求 serving 部署', { modelPath, port })
  const out = await callWithConfirm(client, 'inference_serving_deploy', args, { log })
  if (out.isError || out.value?.ok === false) {
    fail(ERROR_CODES.E_SERVING_UNAVAILABLE, `serving 部署失败：${out.value?.code ?? 'unknown'} — ${out.value?.message ?? ''}`, {
      remediation: '按 serving 返回的 remediation 处理；也可先用 inference_serving_doctor 自检。',
      details: out.value ?? { raw: out.raw },
    })
  }
  const ep = out.value.endpoint
  return {
    endpoint: ep.endpoint,
    port: ep.port,
    id: ep.id,
    servedModelName: ep.servedModelName,
    /** 端点所有权凭证：拆解时必须原样带回，否则服务端会（正确地）拒绝 */
    ownerToken: out.value.ownerToken ?? null,
    reused: out.value.reused === true,
    report: out.value.report ?? null,
    raw: out.value,
  }
}

/** 通过 MCP 让 serving 拆掉一个端点。返回结构化结果，**不抛**（拆解失败要能记账）。 */
export async function teardownViaServing(client, { selector, log = null, ownerToken = null } = {}) {
  log?.info('通过 MCP 请求 serving 拆解端点', { selector })
  const out = await callWithConfirm(client, 'inference_serving_teardown', { selector, ...(ownerToken ? { ownerToken } : {}) }, { log })
  const ok = !out.isError && out.value?.ok !== false
  return {
    selector,
    ok,
    code: ok ? null : (out.value?.code ?? 'E_TEARDOWN_FAILED'),
    message: ok ? null : (out.value?.message ?? String(out.raw ?? '').slice(0, 300)),
    residual: out.value?.residual ?? out.value?.details?.residual ?? null,
    raw: out.value ?? null,
  }
}

/**
 * 性能对比：把基线与候选各起一个端点，跑同一份负载，**跑完必须拆掉**。
 *
 * 为什么拆解是这里的一部分（真机补建，2026-09-24）：
 *   原本只 deploy 不 teardown，于是**每跑一次量化就泄漏两个占着显存的 vLLM 容器**。
 *   连着两次运行后 `docker ps` 里都留着 2 个 up 着的 `inference-serving-*`。
 *   在 GB10 这种统一内存机器上，泄漏的容器会一直占着内存，并让下一次运行在端口/显存上撞车
 *   —— 而这恰恰是本插件自己要求别人做到的事（"拆解并验证无残留"）。
 *
 * 拆解失败**不改判性能结论**（数据已经测到手了），但会：
 *   ① 在 `evidence.cleanup` 里如实记录；② 推一条 `W_RESIDUAL_CLEANUP` 警告进报告。
 *   —— 失败就是失败，不静默。
 *
 * @returns {{ok, metrics, baseline, candidate, gain, cleanup, evidence}}
 */
export async function perfValidation(runtime, {
  client, baselineModelPath, candidateModelPath,
  workload = 'chat-medium', requests = 8, concurrency = 2, log = null,
} = {}) {
  if (!client) {
    fail(ERROR_CODES.E_TESTING_CORE_ERROR, 'perfValidation 需要一个已连接的 serving MCP 客户端', {
      remediation: '内部契约错误；调用方应先用 withServing() 打开客户端。', details: null,
    })
  }
  const baseName = 'quant-baseline'
  const candName = 'quant-candidate'

  const base = await deployViaServing(client, { modelPath: baselineModelPath, servedModelName: baseName, log })
  let cand = null
  let mb = null
  let mc = null
  let cleanup = null
  try {
    cand = await deployViaServing(client, { modelPath: candidateModelPath, servedModelName: candName, log })
    mb = await measureEndpoint(base.endpoint, base.servedModelName, { workload, requests, concurrency, log })
    mc = await measureEndpoint(cand.endpoint, cand.servedModelName, { workload, requests, concurrency, log })
  } finally {
    // 无论测量成功与否都要拆 —— 尤其抛异常时，否则泄漏没人管
    const targets = []
    if (base?.id) targets.push({ tag: 'baseline', selector: base.id, ownerToken: base.ownerToken ?? null })
    if (cand?.id) targets.push({ tag: 'candidate', selector: cand.id, ownerToken: cand.ownerToken ?? null })
    const results = []
    for (const t of targets) {
      try {
        // eslint-disable-next-line no-await-in-loop
        results.push({ ...(await teardownViaServing(client, { selector: t.selector, log, ownerToken: t.ownerToken ?? null })), tag: t.tag })
      } catch (err) {
        results.push({ tag: t.tag, selector: t.selector, ok: false, code: 'E_TEARDOWN_FAILED', message: String(err.message).slice(0, 300) })
      }
    }
    const failed = results.filter((r) => !r.ok)
    cleanup = { attempted: results.length, ok: failed.length === 0, results }
    if (failed.length > 0) {
      runtime?.warnings?.push?.(warn(WARNING_CODES.W_RESIDUAL_CLEANUP,
        `性能评测端点未完全拆解（${failed.length}/${results.length} 失败）：${failed.map((f) => `${f.tag}=${f.code}`).join('、')}`
        + ' —— 容器可能仍在占用显存，请手动核查',
        { cleanup }))
      log?.info('性能评测端点拆解未完成', { failed })
    } else {
      log?.info('性能评测端点已全部拆解', { count: results.length })
    }
  }

  const gainDecode = mb.decodeTps.median > 0 ? ((mc.decodeTps.median - mb.decodeTps.median) / mb.decodeTps.median) * 100 : null
  const gainThroughput = mb.throughputTps > 0 ? ((mc.throughputTps - mb.throughputTps) / mb.throughputTps) * 100 : null
  const gainTtft = mb.ttftMs.median > 0 ? ((mb.ttftMs.median - mc.ttftMs.median) / mb.ttftMs.median) * 100 : null

  return {
    ok: true,
    metrics: {
      primary: 'decode-tps-gain-pct',
      decodeTpsGainPct: gainDecode,
      throughputGainPct: gainThroughput,
      ttftReductionPct: gainTtft,
    },
    baseline: mb,
    candidate: mc,
    gain: { decodeTpsPct: gainDecode, throughputPct: gainThroughput, ttftReductionPct: gainTtft },
    cleanup,
    evidence: {
      workload, requests, concurrency,
      baselineEndpoint: base.endpoint, candidateEndpoint: cand.endpoint,
      baselineModel: baselineModelPath, candidateModel: candidateModelPath,
      servingDeployReports: { baseline: base.report, candidate: cand.report },
      cleanup,
      contractVersion: '1.0.0',
      implementation: 'inference-quantization/internal',
    },
  }
}
export { WORKLOADS, warn, WARNING_CODES }
