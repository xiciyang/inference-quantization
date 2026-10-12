/**
 * 集成测试：DSH 插件契约 + API 闸门层（不依赖 GPU/Docker）。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { name, inject, apply, validateConfig } from '../../src/index.mjs'
import { TOOL_SPECS, ALL_TOOL_SPECS } from '../../src/tools.mjs'
import * as api from '../../src/api.mjs'
import { makeSandbox } from '../helpers/sandbox.mjs'
import { resetKnobs } from '../helpers/fake-executor.mjs'
import { ERROR_CODES } from '../../src/errors.mjs'

function mockCtx({ withWebServer = true, failOn = null } = {}) {
  const registered = []; const disposers = []; const effects = []; const routes = []
  return {
    ctx: {
      tools: { register(t) { if (failOn && t.name === failOn) throw new Error(`注入失败：${t.name}`); registered.push(t) } },
      effect(fn, label) { effects.push(label); const d = fn(); if (typeof d === 'function') disposers.push({ label, d }); return d },
      on() {},
      get(k) { return k === 'webServer' ? (withWebServer ? { register: (r) => { routes.push(r); return () => {} } } : undefined) : undefined },
      logger() { return { info() {}, warn() {} } },
    },
    registered, disposers, effects, routes,
  }
}

test('契约: 导出 name / inject / apply，只硬依赖 tools', () => {
  assert.equal(name, 'inference-quantization')
  assert.deepEqual(inject, ['tools'], 'webServer 必须可选（headless profile 也要能挂载）')
  assert.equal(typeof apply, 'function')
})

test('契约: 每个工具一个独立 effect，且工具形状完整（只用 DSH 支持的 schema 子集）', () => {
  const { ctx, registered, effects } = mockCtx()
  apply(ctx, undefined)
  assert.equal(registered.length, ALL_TOOL_SPECS.length)
  for (const spec of ALL_TOOL_SPECS) assert.ok(effects.some((l) => l.includes(spec.name)))
  const ALLOWED = new Set(['type', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const', 'oneOf', 'description'])
  for (const t of registered) {
    assert.equal(typeof t.name, 'string')
    assert.equal(typeof t.description, 'string')
    assert.equal(t.parameters.type, 'object')
    assert.ok(t.output?.schema)
    assert.equal(typeof t.execute, 'function')
    const walk = (node, path) => {
      for (const [k, v] of Object.entries(node)) {
        assert.ok(ALLOWED.has(k), `${t.name}${path}.${k} 不在 DSH 支持的 schema 关键字内`)
        if (k === 'properties') for (const [pk, pv] of Object.entries(v)) walk(pv, `${path}.properties.${pk}`)
        if (k === 'items') walk(v, `${path}.items`)
      }
    }
    walk(t.parameters, '')
  }
})

test('契约: 五个必需 MCP 工具都在（题面指定）', () => {
  const names = TOOL_SPECS.map((t) => t.name)
  for (const n of ['inference_quantization_quantize', 'inference_quantization_get_report',
    'inference_quantization_rollback', 'inference_quantization_list', 'inference_quantization_qat_plan']) {
    assert.ok(names.includes(n), `缺工具 ${n}`)
  }
})

test('契约: 卸载时每个 effect 都被清理', () => {
  const { ctx, disposers } = mockCtx()
  apply(ctx, undefined)
  assert.ok(disposers.length >= 1)
  for (const { d, label } of disposers) {
    assert.equal(typeof d, 'function', `${label} 应返回 disposer`)
    assert.doesNotThrow(() => d())
  }
})

test('契约: 单个工具注册失败被隔离，其余照常', () => {
  const failing = ALL_TOOL_SPECS[1].name
  const { ctx, registered } = mockCtx({ failOn: failing })
  assert.doesNotThrow(() => apply(ctx, undefined))
  assert.equal(registered.length, ALL_TOOL_SPECS.length - 1)
})

test('契约: webServer 缺失时仍能挂载；存在时注册只读路由', async () => {
  const a = mockCtx({ withWebServer: false })
  assert.doesNotThrow(() => apply(a.ctx, undefined))
  assert.equal(a.routes.length, 0)

  const b = mockCtx({ withWebServer: true })
  apply(b.ctx, { root: '/tmp/iq-contract' })
  assert.equal(b.routes.length, 1)
  assert.equal(b.routes[0].path, '/api/inference-quantization/status')
  let body = null
  await b.routes[0].handler({}, { setHeader() {}, set statusCode(v) { void v }, end(x) { body = x } })
  assert.ok(JSON.parse(body).package)
})

test('契约: 非法配置加载期响亮失败', () => {
  assert.throws(() => validateConfig({ nope: 1 }), /未知配置键/)
  assert.throws(() => validateConfig({ maxDegradeSteps: 0 }), /必须是 ≥1 的整数/)
  assert.throws(() => validateConfig({ root: 5 }), /root 必须是字符串/)
  assert.doesNotThrow(() => validateConfig({ root: '/tmp/x', quantImage: 'a:1' }))
})

test('契约: execute 尊重 exec.signal（已取消不做事）', async () => {
  const { ctx, registered } = mockCtx()
  apply(ctx, undefined)
  const tool = registered.find((t) => t.name === 'inference_quantization_quantize')
  const ac = new AbortController(); ac.abort()
  const out = await tool.execute({ model: '/x' }, { signal: ac.signal })
  assert.equal(out.ok, false)
  assert.equal(out.code, 'E_CANCELLED')
})

test('API: 永不抛异常 —— 未知能力/未声明资源都回结构化错误', async () => {
  const sb = await makeSandbox()
  try {
    resetKnobs()
    const r = await api.quantize('/nonexistent/model', { root: sb.root, options: { ...sb.options, executor: 'fake' } })
    assert.equal(r.ok, false)
    assert.ok(r.code)
    assert.ok(r.remediation)
  } finally { await sb.cleanup() }
})

test('API: list / get_report 在空矩阵下正常返回（不抛）', async () => {
  const sb = await makeSandbox()
  try {
    const l = await api.list({}, { root: sb.root, options: sb.options })
    assert.equal(l.ok, true)
    assert.equal(l.count, 0)
    assert.deepEqual(l.capabilities.fiveElements, ['method', 'accuracy', 'performance', 'platform', 'status'])

    const rep = await api.getReport({}, { root: sb.root, options: sb.options })
    assert.equal(rep.ok, true)
    assert.equal(rep.count, 0)

    const missing = await api.getReport({ runId: 'nope' }, { root: sb.root, options: sb.options })
    assert.equal(missing.ok, false)
    assert.equal(missing.code, ERROR_CODES.E_RUN_NOT_FOUND)
    assert.ok(missing.details.known !== undefined)
  } finally { await sb.cleanup() }
})

test('API: rollback 无目标时明确报错（不静默成功）', async () => {
  const sb = await makeSandbox()
  try {
    // rollback 是 R2 → 先过确认闸门（这本身也是一条被验证的契约）
    const first = await api.rollback({ selector: 'nothing' }, { root: sb.root, options: sb.options })
    assert.equal(first.ok, false)
    assert.equal(first.code, ERROR_CODES.E_CONFIRMATION_REQUIRED)
    assert.match(first.details.confirmToken, /^cf_/)
    // 带令牌重试 → 应落到"找不到目标"而不是又被确认拦住
    const r = await api.rollback({ selector: 'nothing', confirm: first.details.confirmToken }, { root: sb.root, options: sb.options })
    assert.equal(r.ok, false)
    assert.equal(r.code, ERROR_CODES.E_NO_QUANTIZATION_TARGET)
    assert.ok(r.remediation)
  } finally { await sb.cleanup() }
})

test('API: doctor 自检只读可用，且暴露方法表、契约、镜像状态', async () => {
  const sb = await makeSandbox()
  try {
    const d = await api.doctor({}, { root: sb.root, options: { ...sb.options, quantImage: 'fake/image:1' } })
    assert.equal(d.ok, true)
    assert.equal(d.methods.length, 4)
    assert.deepEqual(d.methods.map((m) => m.level), [1, 2, 3, 4])
    assert.equal(d.testingCore.internal, true)
    assert.equal(d.testingCore.contractVersion, '1.0.0')
    assert.deepEqual(d.testingCore.functions.parityCheck.inputs.length > 0, true)
    assert.equal(d.image.verified, false)
    assert.ok(d.manifest.notProvided.includes('qat_execute'))
    assert.ok(d.manifest.notProvided.includes('disable_leakage_gate'))
  } finally { await sb.cleanup() }
})

test('API: gates.yaml 只生成空模板（不填默认阈值）', async () => {
  const sb = await makeSandbox()
  try {
    const d = await api.doctor({}, { root: sb.root, options: sb.options })
    assert.equal(d.ok, true)
    const { promises: fs } = await import('node:fs')
    const { join } = await import('node:path')
    const p = join(sb.root, 'config', 'gates.yaml')
    const raw = await fs.readFile(p, 'utf8')
    assert.match(raw, /max_accuracy_drop_pct:\s*null/, '模板里必须是 null，不能有默认值')
    assert.match(raw, /min_perf_gain_pct:\s*null/)
    assert.match(raw, /不猜默认值/)
  } finally { await sb.cleanup() }
})

test('API: qat 计划可从矩阵记录重建（不依赖原报告文件）', async () => {
  const sb = await makeSandbox()
  try {
    // 造一条失败记录
    const { withMatrix } = await import('../../src/state/matrix.mjs')
    const { createRuntime } = await import('../../src/config.mjs')
    const runtime = await createRuntime({ root: sb.root, options: sb.options })
    await withMatrix(runtime, (m) => {
      m.runs.push({
        id: 'qr_x', runKey: 'k', sourceModel: sb.modelPath, method: 'awq-w4a16', level: 1, bits: 4,
        status: 'failed', accuracy: { dropPct: 9.5, thresholdPct: 5, samples: 400, passed: false },
        performance: { decodeTpsGainPct: 80, thresholdPct: 20, passed: true },
      })
      return null
    })
    const opts2 = { ...sb.options, executor: 'fake' }
    const gate = await api.qat({ runId: 'qr_x' }, { root: sb.root, options: opts2 })
    assert.equal(gate.ok, false)
    assert.equal(gate.code, ERROR_CODES.E_CONFIRMATION_REQUIRED, 'qat_plan 是写操作（会写计划到报告），需确认')
    const out = await api.qat({ runId: 'qr_x', confirm: gate.details.confirmToken }, { root: sb.root, options: opts2 })
    assert.equal(out.ok, true, JSON.stringify(out).slice(0, 300))
    assert.equal(out.approves, false, '默认只出计划')
    assert.ok(out.plan.data && out.plan.config && out.plan.expectedGain && out.plan.rollbackPath)
    assert.equal(out.plan.executes, false)
    assert.equal(out.plan.approval.cacheable, false)
  } finally { await sb.cleanup() }
})

test('API: QAT 授权需要匹配令牌 + 显式沙箱降级授权', async () => {
  const sb = await makeSandbox()
  try {
    const { authorizeQat } = await import('../../src/qat.mjs')
    const plan = { planId: 'p1', approval: { token: 'qat_abc' }, config: {}, data: { calibrationSet: {} }, expectedGain: { thresholdPct: 5 } }
    assert.throws(() => authorizeQat({}, { plan, confirmToken: 'wrong', allowSandboxDegrade: true }), (e) => {
      assert.equal(e.code, ERROR_CODES.E_QAT_REQUIRES_APPROVAL); return true
    })
    assert.throws(() => authorizeQat({}, { plan, confirmToken: 'qat_abc', allowSandboxDegrade: false }), (e) => {
      assert.equal(e.code, ERROR_CODES.E_QAT_UNAVAILABLE); return true
    })
    const ok = authorizeQat({}, { plan, confirmToken: 'qat_abc', allowSandboxDegrade: true })
    assert.equal(ok.ok, true)
    assert.equal(ok.executes, false, '授权也不代跑训练')
    assert.ok(ok.commands.length > 0)
  } finally { await sb.cleanup() }
})

test('API: 确认令牌在同一输入上稳定（指纹不得把 confirm 自己算进去）', async () => {
  const sb = await makeSandbox()
  try {
    const base = { root: sb.root, options: { ...sb.options, executor: 'fake' } }
    const t1 = (await api.quantize({ model: sb.modelPath }, base)).details?.confirmToken
    assert.match(t1, /^cf_/)
    // 第二次带同一令牌 → 必须**推进**（不是又被要一次确认）
    const run = await api.quantize({ model: sb.modelPath, confirm: t1 }, base)
    assert.notEqual(run.code, ERROR_CODES.E_CONFIRMATION_REQUIRED,
      `同一 input + 正确令牌必须能推进，否则确认流程永远走不通（实际 ${run.code}）`)
    // 令牌在第二参数里也应等价
    const run2 = await api.quantize({ model: sb.modelPath }, { ...base, confirm: t1 })
    assert.notEqual(run2.code, ERROR_CODES.E_CONFIRMATION_REQUIRED)
  } finally { await sb.cleanup() }
})

test('API: 门禁阈值三种写法都认（snake/camel/options.*）—— 不能被静默忽略', async () => {
  const sb = await makeSandbox()
  try {
    const base = { root: sb.root, options: { ...sb.options, executor: 'fake' } }
    // 都不给 → 先过确认闸门，再明确拒绝（阈值缺失不能被"看起来传了"掩盖）
    const gate = await api.quantize({ model: sb.modelPath }, base)
    assert.equal(gate.code, ERROR_CODES.E_CONFIRMATION_REQUIRED)
    const none = await api.quantize({ model: sb.modelPath, confirm: gate.details.confirmToken }, base)
    assert.equal(none.code, ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, JSON.stringify(none).slice(0, 200))
    // 给在 input 的 camelCase
    const g1 = await api.quantize({ model: sb.modelPath, options: { maxAccuracyDropPct: 1, minPerfGainPct: 1 } }, base)
    const a = await api.quantize({ model: sb.modelPath, options: { maxAccuracyDropPct: 1, minPerfGainPct: 1 }, confirm: g1.details?.confirmToken }, base)
    assert.notEqual(a.code, ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, JSON.stringify(a).slice(0, 200))
    // 给在第二参数 options 的 snake_case（CLI 路径）
    const baseB = { root: sb.root, options: { ...sb.options, executor: 'fake', max_accuracy_drop_pct: 1, min_perf_gain_pct: 1 } }
    const g2 = await api.quantize({ model: sb.modelPath, outputName: 'b' }, baseB)
    const b = await api.quantize({ model: sb.modelPath, outputName: 'b', confirm: g2.details?.confirmToken }, baseB)
    assert.notEqual(b.code, ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, JSON.stringify(b).slice(0, 200))
    // 给在 input 的 snake_case
    const g3 = await api.quantize({ model: sb.modelPath, outputName: 'c', max_accuracy_drop_pct: 1, min_perf_gain_pct: 1 }, base)
    const c = await api.quantize({ model: sb.modelPath, outputName: 'c', max_accuracy_drop_pct: 1, min_perf_gain_pct: 1, confirm: g3.details?.confirmToken }, base)
    assert.notEqual(c.code, ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, JSON.stringify(c).slice(0, 200))
  } finally { await sb.cleanup() }
})
