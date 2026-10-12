import { test } from 'node:test'
import assert from 'node:assert/strict'
import { emptyMatrix, parseMatrix, serializeMatrix, findRun, verifiedRuns, runsForSource, pushHistory, summarizeRun, completeness } from '../../src/state/matrix.mjs'
import { sha256, canonicalJson, shortId } from '../../src/util/hash.mjs'
import { parseArgv } from '../../src/cli.mjs'
import { ERROR_CODES, ServingError, fail, toErrorPayload } from '../../src/errors.mjs'
import { PACKAGE_ROOT, PACKAGE_VERSION, MCP_INTERFACE_VERSION } from '../../src/version.mjs'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parse as parseYaml } from '../../src/util/yaml.mjs'

test('矩阵: 往返序列化（含中文、嵌套证据）', () => {
  const m = emptyMatrix()
  m.runs.push({
    id: 'qr_1', runKey: 'k1', sourceModel: '/data/models/中文 模型', method: 'awq-w4a16', level: 1, bits: 4,
    status: 'verified', outputPath: '/x/out',
    accuracy: { metric: 'perplexity-relative-drop-pct', dropPct: 1.23, passed: true },
    performance: { decodeTpsGainPct: 33.3, passed: true },
    platform: { gpu: 'NVIDIA GB10', unifiedMemory: true },
    leakage: { clean: true, counts: { calibration: 256, test: 200 } },
    sourceIntegrity: { unchanged: true },
    degrade: { steps: 1, chain: 'L1:awq-w4a16(E_ACCURACY_GATE_FAILED) → L2:int8-w8a8' },
    history: [{ at: 't', event: 'quantize-gated', detail: { method: 'int8-w8a8' } }],
  })
  const back = parseMatrix(serializeMatrix(m))
  assert.equal(back.runs.length, 1)
  assert.equal(back.runs[0].sourceModel, '/data/models/中文 模型')
  assert.equal(back.runs[0].accuracy.dropPct, 1.23)
  assert.equal(back.runs[0].leakage.counts.calibration, 256)
})

test('矩阵: 五要素完整性检查（缺哪个就说哪个）', () => {
  const full = { method: 'm', accuracy: {}, performance: {}, platform: {}, status: 'verified' }
  assert.equal(completeness(full).complete, true)
  const partial = { method: 'm', status: 'failed' }
  const c = completeness(partial)
  assert.equal(c.complete, false)
  assert.deepEqual(c.missing.sort(), ['accuracy', 'performance', 'platform'])
})

test('矩阵: summarizeRun 把缺失字段显式暴露为 null（缺失本身是验收要看的东西）', () => {
  const s = summarizeRun({ id: 'x', sourceModel: '/m', status: 'failed' })
  assert.equal(s.method, null)
  assert.equal(s.accuracy, null)
  assert.equal(s.sourceHashUnchanged, null)
  assert.equal(s.leakageFree, null)
})

test('矩阵: findRun 支持多种选择器；拒绝以旧格式覆盖新格式', () => {
  const m = emptyMatrix()
  m.runs.push({ id: 'qr_1', runKey: 'k1', sourceModel: '/data/models/m', outputPath: '/out/x' })
  assert.equal(findRun(m, 'qr_1').id, 'qr_1')
  assert.equal(findRun(m, 'k1').id, 'qr_1')
  assert.equal(findRun(m, '/out/x').id, 'qr_1')
  assert.equal(findRun(m, '/data/models/m').id, 'qr_1')
  assert.equal(findRun(m, 'm').id, 'qr_1', '源模型末段也应命中')
  assert.equal(findRun(m, 'nope'), null)
  assert.throws(() => parseMatrix('version: 99\nruns: []\n'), /高于本插件支持/)
})

test('矩阵: verifiedRuns / runsForSource / pushHistory 上限', () => {
  const m = emptyMatrix()
  m.runs.push({ id: 'a', sourceModel: '/s', status: 'verified', createdAt: '2' })
  m.runs.push({ id: 'b', sourceModel: '/s', status: 'failed', createdAt: '1' })
  m.runs.push({ id: 'c', sourceModel: '/other', status: 'verified', createdAt: '3' })
  assert.deepEqual(verifiedRuns(m).map((r) => r.id), ['a', 'c'])
  assert.deepEqual(runsForSource(m, '/s').map((r) => r.id), ['b', 'a'])
  const r = { }
  for (let i = 0; i < 250; i += 1) pushHistory(r, `e${i}`)
  assert.equal(r.history.length, 200)
  assert.equal(r.history.at(-1).event, 'e249')
})

test('哈希: canonicalJson 键序不敏感（runKey 幂等的前提）', () => {
  assert.equal(canonicalJson({ a: 1, b: [2, { d: 4, c: 3 }] }), canonicalJson({ b: [2, { c: 3, d: 4 }], a: 1 }))
  assert.equal(shortId('x', 12).length, 12)
  assert.equal(sha256('x').length, 64)
})

test('错误: 量化专项错误码齐全且格式统一', () => {
  for (const k of ['E_CALIB_LEAKAGE_DETECTED', 'E_SOURCE_MODEL_MODIFIED', 'E_THRESHOLD_NOT_PROVIDED',
    'E_QAT_REQUIRES_APPROVAL', 'E_IMAGE_NOT_VERIFIED', 'E_VLLM_INCOMPATIBLE', 'E_DEGRADE_EXHAUSTED']) {
    assert.equal(ERROR_CODES[k], k, `缺错误码 ${k}`)
  }
  const e = new ServingError(ERROR_CODES.E_CALIB_LEAKAGE_DETECTED, 'x', { remediation: 'y' })
  assert.equal(e.toJSON().remediation, 'y')
  assert.equal(toErrorPayload(new Error('boom')).code, ERROR_CODES.E_INTERNAL)
  assert.throws(() => fail(ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, 'x'), ServingError)
})

test('版本: package.json / src/version.mjs / manifest.yaml 三处一致', () => {
  const pkg = JSON.parse(readFileSync(join(PACKAGE_ROOT, 'package.json'), 'utf8'))
  assert.equal(pkg.name, 'inference-quantization')
  assert.equal(pkg.version, PACKAGE_VERSION)
  const mf = parseYaml(readFileSync(join(PACKAGE_ROOT, 'manifest.yaml'), 'utf8'))
  assert.equal(mf.version, PACKAGE_VERSION)
  assert.equal(mf.interface.mcp, MCP_INTERFACE_VERSION)
  assert.ok(mf.capabilities.length >= 8)
  assert.ok(mf.not_provided.some((n) => n.id === 'qat_execute'))
  assert.ok(mf.not_provided.some((n) => n.id === 'disable_leakage_gate'))
})

test('工程: 零运行时依赖 + 声明 engines/os/cpu', () => {
  const pkg = JSON.parse(readFileSync(join(PACKAGE_ROOT, 'package.json'), 'utf8'))
  assert.deepEqual(Object.keys(pkg.dependencies ?? {}), [])
  assert.ok(pkg.engines?.node && pkg.os?.length && pkg.cpu?.length)
  assert.equal(pkg.dsh.bundle.patch, './cordis.patch.yml')
})

test('CLI: 参数解析（含带连字符的选项名）', () => {
  const r = parseArgv(['quantize', '/m', '--max-accuracy-drop-pct', '5', '--json', '--method=w8a16'])
  assert.deepEqual(r.positional, ['quantize', '/m'])
  assert.equal(r.options['max-accuracy-drop-pct'], '5')
  assert.equal(r.options.method, 'w8a16')
  assert.ok(r.flags.has('json'))
})
