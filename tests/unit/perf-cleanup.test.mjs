/**
 * 性能评测的**端点拆解** —— 资源卫生回归测试。
 *
 * 真机缺陷（2026-09-24 补建）：`perfValidation` 原本只 `deploy` 不 `teardown`，
 * 于是每跑一次量化就泄漏两个占着显存的 vLLM 容器 —— 我连着两次运行后都在
 * `docker ps` 里看到 2 个 up 着的 `inference-serving-*`。
 *
 * 这组测试用**假 MCP 客户端**断言：
 *   ① 两个端点都被拆（顺序与次数）；
 *   ② 测量中途抛异常时**仍然**拆（finally 语义）；
 *   ③ 拆解失败不改判性能结论，但必须留下 `cleanup.ok=false` 与 `W_RESIDUAL_CLEANUP` 警告。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { perfValidation } from '../../src/testing/perf.mjs'

/** 假 serving MCP 客户端：记录被调用的工具名与参数。 */
function fakeClient({ failTeardownFor = [], port = 8120 } = {}) {
  const calls = []
  const value = (payload) => ({ isError: false, value: payload, raw: JSON.stringify(payload) })
  return {
    calls,
    async callTool(name, args) {
      calls.push({ name, args })
      if (name === 'inference_serving_deploy') {
        const id = args.model.includes('candidate') ? 'ep_candidate' : 'ep_baseline'
        const p = port
        return value({
          ok: true, reused: false,
          endpoint: { id, endpoint: `http://127.0.0.1:${p}`, port: p, servedModelName: args.servedModelName },
          report: { id, status: 'serving' },
        })
      }
      if (name === 'inference_serving_teardown') {
        const sel = args.selector
        if (failTeardownFor.includes(sel)) {
          return value({ ok: false, code: 'E_CLEANUP_INCOMPLETE', message: `注入失败：${sel} 拆不掉`, details: { residual: [`container-${sel}`] } })
        }
        return value({ ok: true, residual: [] })
      }
      throw new Error(`未预期的工具调用：${name}`)
    },
  }
}

/**
 * 起一个**真的本地 HTTP 服务**扮演 vLLM 端点。
 *
 * 刻意不用打桩 fetch：`measureEndpoint` 走的是 `node:http`，桩 fetch 根本拦不住
 * （我第一版就是这么写的，结果测试真的去连了 127.0.0.1:8120）。
 * 起真服务更省事也更诚实 —— 测量代码走的仍是它本来的网络路径。
 */
async function withFakeEndpoint(fn) {
  const { createServer } = await import('node:http')
  const server = createServer((req, res) => {
    let body = ''
    req.on('data', (c) => { body += c })
    req.on('end', () => {
      void body
      res.writeHead(200, { 'content-type': 'text/event-stream' })
      // 让 TTFT 非零，且 token 数可预期
      setTimeout(() => {
        res.write(`data: ${JSON.stringify({ choices: [{ text: 'tok ' }] })}\n\n`)
        let n = 0
        const tick = setInterval(() => {
          n += 1
          res.write(`data: ${JSON.stringify({ choices: [{ text: `t${n} ` }] })}\n\n`)
          if (n >= 5) { clearInterval(tick); res.write('data: [DONE]\n\n'); res.end() }
        }, 2)
      }, 5)
    })
  })
  await new Promise((r) => server.listen(0, '127.0.0.1', r))
  const port = server.address().port
  try {
    return await fn(port)
  } finally {
    await new Promise((r) => server.close(r))
  }
}

const runtime = () => ({ warnings: [] })

test('端点拆解: 两个端点跑完都被拆掉，且 cleanup.ok=true', async () => {
  await withFakeEndpoint(async (port) => {
    const client = fakeClient({ port })
    const rt = runtime()
    const out = await perfValidation(rt, {
      client, baselineModelPath: '/m/baseline', candidateModelPath: '/m/candidate',
      requests: 2, concurrency: 1,
    })
    assert.equal(out.ok, true)
    const torn = client.calls.filter((c) => c.name === 'inference_serving_teardown').map((c) => c.args.selector)
    assert.deepEqual(torn.sort(), ['ep_baseline', 'ep_candidate'], `应拆两个端点，实际 ${JSON.stringify(torn)}`)
    assert.equal(out.cleanup.attempted, 2)
    assert.equal(out.cleanup.ok, true)
    assert.equal(rt.warnings.length, 0)
    // 报告里也要带上
    assert.equal(out.evidence.cleanup.ok, true)
  })
})

test('端点拆解: 测量失败（端点无响应）→ 仍然拆（finally 语义，泄漏不留）', async () => {
  // 指向一个**没有服务**的端口：postStream 全部失败 → measureEndpoint 抛 E_TESTING_CORE_ERROR
  const client = fakeClient({ port: 59999 })
  const rt = runtime()
  await assert.rejects(
    () => perfValidation(rt, {
      client, baselineModelPath: '/m/baseline', candidateModelPath: '/m/candidate',
      requests: 1, concurrency: 1,
    }),
    (e) => e.code === 'E_TESTING_CORE_ERROR',
  )
  const torn = client.calls.filter((c) => c.name === 'inference_serving_teardown').map((c) => c.args.selector)
  assert.deepEqual(torn.sort(), ['ep_baseline', 'ep_candidate'], `异常路径也必须拆干净，实际 ${JSON.stringify(torn)}`)
})

test('端点拆解: 拆解失败不改判性能结论，但如实记账并推 W_RESIDUAL_CLEANUP', async () => {
  await withFakeEndpoint(async (port) => {
    const client = fakeClient({ port, failTeardownFor: ['ep_candidate'] })
    const rt = runtime()
    const out = await perfValidation(rt, {
      client, baselineModelPath: '/m/baseline', candidateModelPath: '/m/candidate',
      requests: 2, concurrency: 1,
    })
    // 数据已经测到了 → 结论照给
    assert.equal(out.ok, true)
    assert.ok(out.metrics.decodeTpsGainPct !== undefined)
    // 但必须说清楚有一个没拆掉
    assert.equal(out.cleanup.ok, false)
    assert.equal(out.cleanup.attempted, 2)
    const bad = out.cleanup.results.find((r) => !r.ok)
    assert.equal(bad.tag, 'candidate')
    assert.equal(bad.code, 'E_CLEANUP_INCOMPLETE')
    assert.deepEqual(bad.residual, ['container-ep_candidate'])
    const codes = rt.warnings.map((w) => (typeof w.toJSON === 'function' ? w.toJSON().code : w.code))
    assert.ok(codes.includes('W_RESIDUAL_CLEANUP'), `应推残留警告，实际 ${JSON.stringify(codes)}`)
  })
})

test('端点拆解: 不泄漏 —— deploy 次数必须等于 teardown 次数', async () => {
  await withFakeEndpoint(async (port) => {
    const client = fakeClient({ port })
    await perfValidation(runtime(), {
      client, baselineModelPath: '/m/baseline', candidateModelPath: '/m/candidate',
      requests: 2, concurrency: 1,
    })
    const deploys = client.calls.filter((c) => c.name === 'inference_serving_deploy').length
    const teardowns = client.calls.filter((c) => c.name === 'inference_serving_teardown').length
    assert.equal(deploys, 2)
    assert.equal(teardowns, deploys, 'deploy 与 teardown 必须一一对应（这是"不泄漏"的形式化表达）')
  })
})
