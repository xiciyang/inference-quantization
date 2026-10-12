/**
 * 源模型结构预检 —— 单元测试。
 *
 * 这一组测试的**起点是真机事故**：2026-09-23 有一份下载中断的模型
 * （model.safetensors 只有前 0.747 GiB 是真数据，其余 2.128 GiB 全为 0x00，
 * 其中包含 model.norm.weight）被完整量化了 436 秒，并被签收为"精度零损失"。
 * 旧流程的目录哈希校验通过了 —— 因为它只回答"内容有没有被改动"。
 *
 * 因此这里的每条用例都对应一个**必须能被检出**的残缺形态。
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { randomBytes } from 'node:crypto'
import { safetensorsBuffer, minimalModelTensors } from '../helpers/safetensors.mjs'
import { scanSafetensors, validateModelDir, assertSourceModelValid, DEFAULT_ZERO_POLICY } from '../../src/model-validate.mjs'

async function tmpdirFor() {
  return fs.mkdtemp(join(tmpdir(), 'iq-mv-'))
}

function fakeRuntime(root, extra = {}) {
  return {
    paths: { modelValidationFile: join(root, 'state', 'model-validation.json') },
    warnings: [],
    __log: { info() {}, warn() {}, phase() {} },
    ...extra,
  }
}

/** 写一个结构自洽的模型目录。 */
async function writeModel(dir, { layers = 2, vocab = 128, hidden = 64, config = {}, extraFiles = {}, transformers = null } = {}) {
  await fs.mkdir(dir, { recursive: true })
  await fs.writeFile(join(dir, 'config.json'), JSON.stringify({
    architectures: ['LlamaForCausalLM'], model_type: 'llama',
    num_hidden_layers: layers, hidden_size: hidden, vocab_size: vocab,
    ...config,
  }))
  const tensors = transformers ?? minimalModelTensors({ layers, hidden, vocab })
  const buf = safetensorsBuffer(tensors)
  await fs.writeFile(join(dir, 'model.safetensors'), buf)
  for (const [name, content] of Object.entries(extraFiles)) {
    await fs.writeFile(join(dir, name), content)
  }
  return { dir, buf, tensors }
}

// 用很小的分块跑，顺便覆盖"跨分块零区合并"的逻辑
const TEST_POLICY = { ...DEFAULT_ZERO_POLICY, chunkBytes: 256, minElements: 64 }

test('预检: 结构完整的模型 → 通过，且给出张量数与零张量数', async () => {
  const base = await tmpdirFor()
  try {
    const { dir } = await writeModel(join(base, 'm'))
    const r = await validateModelDir(dir, { policy: TEST_POLICY })
    assert.equal(r.ok, true, JSON.stringify(r.problems))
    assert.equal(r.problems.length, 0)
    assert.ok(r.evidence.totalTensors >= 6)
    assert.equal(r.evidence.totalZeroTensors, 0)
    assert.equal(r.evidence.totalZeroBytes, 0)
    assert.equal(r.evidence.declaredLayers, 2)
    assert.equal(r.evidence.presentLayers, 2)
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 真实事故复现 —— 后半段整块为零的 safetensors 必须被检出', async () => {
  // 复刻现场形态：文件长度是对的（下载器预分配过），但某个字节偏移之后全是 0x00。
  // 这正是"目录哈希通过、模型却是废的"那一类残缺。
  const base = await tmpdirFor()
  try {
    const { dir, buf } = await writeModel(join(base, 'm'), { layers: 4 })
    // 找到数据区起点（8 字节头长 + 头部 JSON）
    const headerLen = Number(buf.readBigUInt64LE(0))
    const dataStart = 8 + headerLen
    const cut = dataStart + Math.floor((buf.length - dataStart) * 0.5)
    buf.fill(0, cut)
    await fs.writeFile(join(dir, 'model.safetensors'), buf)

    const r = await scanSafetensors(join(dir, 'model.safetensors'), { policy: TEST_POLICY })
    assert.equal(r.ok, false, '必须判失败')
    const kinds = r.problems.map((p) => p.kind)
    assert.ok(kinds.includes('zero-filled'), `应检出 zero-filled，实际 ${JSON.stringify(kinds)}`)
    assert.ok(r.evidence.zeroTensorCount > 0, '应列出整块为零的张量')
    assert.ok(r.evidence.zeroBytes > 0)
    assert.ok(r.evidence.largestZeroRegion && r.evidence.largestZeroRegion.bytes > 0, '应给出最大连续零区')
    // 文件长度本身没变 —— 说明"没被截断"并不等于"是完整模型"
    assert.equal(r.evidence.truncatedBy, 0)
    assert.equal(r.evidence.size, buf.length)
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 零填充会点名到具体张量（可操作性）', async () => {
  const base = await tmpdirFor()
  try {
    const { dir } = await writeModel(join(base, 'm'), {
      transformers: [
        { name: 'model.embed_tokens.weight', shape: [128, 64] },
        { name: 'model.layers.0.self_attn.q_proj.weight', shape: [64, 64] },
        { name: 'model.layers.1.self_attn.q_proj.weight', shape: [64, 64] },
        { name: 'model.norm.weight', shape: [64] },
      ],
    })
    // 把最后一层写成全零
    const buf = safetensorsBuffer([
      { name: 'model.embed_tokens.weight', shape: [128, 64] },
      { name: 'model.layers.0.self_attn.q_proj.weight', shape: [64, 64] },
      { name: 'model.layers.1.self_attn.q_proj.weight', shape: [64, 64] },
      { name: 'model.norm.weight', shape: [64] },
    ], { zero: ['model.layers.1.self_attn.q_proj.weight'] })
    await fs.writeFile(join(dir, 'model.safetensors'), buf)
    const r = await scanSafetensors(join(dir, 'model.safetensors'), { policy: TEST_POLICY })
    assert.equal(r.ok, false)
    const names = r.evidence.zeroTensors.map((t) => t.name)
    assert.deepEqual(names, ['model.layers.1.self_attn.q_proj.weight'])
    assert.equal(r.evidence.zeroTensors[0].zeroRatio, 1)
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 小张量（< minElements）恰好为 0 不算残缺 —— 长度 1 的 bias 为 0 是常见的', async () => {
  const base = await tmpdirFor()
  try {
    const { dir } = await writeModel(join(base, 'm'))
    const buf = safetensorsBuffer([
      { name: 'model.embed_tokens.weight', shape: [128, 64] },
      { name: 'model.layers.0.self_attn.q_proj.bias', shape: [8] },   // 8 < 64
      { name: 'model.layers.0.self_attn.q_proj.weight', shape: [64, 64] },
      { name: 'model.layers.1.self_attn.q_proj.weight', shape: [64, 64] },
    ], { zero: ['model.layers.0.self_attn.q_proj.bias'] })
    await fs.writeFile(join(dir, 'model.safetensors'), buf)
    const r = await scanSafetensors(join(dir, 'model.safetensors'), { policy: TEST_POLICY })
    assert.equal(r.ok, true, `小张量为零不应判失败：${JSON.stringify(r.problems)}`)
    assert.equal(r.evidence.zeroTensorCount, 0)
    assert.ok(r.evidence.zeroBytes > 0, '零字节仍应被如实记录（只是不据此判失败）')
    assert.equal(r.evidence.zeroPolicy.minElements, 64)
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 文件比头部声明短 → truncated（下载未完成）', async () => {
  const base = await tmpdirFor()
  try {
    const { dir, buf } = await writeModel(join(base, 'm'))
    const truncated = buf.subarray(0, buf.length - 700)
    await fs.writeFile(join(dir, 'model.safetensors'), truncated)
    const r = await scanSafetensors(join(dir, 'model.safetensors'), { policy: TEST_POLICY })
    assert.equal(r.ok, false)
    assert.ok(r.problems.some((p) => p.kind === 'truncated'), JSON.stringify(r.problems))
    assert.equal(r.evidence.truncatedBy, 700)
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 头部非法 → header-invalid（不猜、不静默）', async () => {
  const base = await tmpdirFor()
  try {
    const dir = join(base, 'm')
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(join(dir, 'model.safetensors'), Buffer.alloc(64, 7))
    const r = await scanSafetensors(join(dir, 'model.safetensors'), { policy: TEST_POLICY })
    assert.equal(r.ok, false)
    assert.equal(r.kind, 'header-invalid')
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: config 声明的层数与权重里实际的层数不符 → layer-count-mismatch', async () => {
  const base = await tmpdirFor()
  try {
    const { dir } = await writeModel(join(base, 'm'), { layers: 4, config: { num_hidden_layers: 8 } })
    const r = await validateModelDir(dir, { policy: TEST_POLICY })
    assert.equal(r.ok, false)
    assert.ok(r.problems.some((p) => p.kind === 'layer-count-mismatch'), JSON.stringify(r.problems))
    assert.equal(r.evidence.declaredLayers, 8)
    assert.equal(r.evidence.presentLayers, 4)
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: index.json 列了却不存在的分片 → shard-missing', async () => {
  const base = await tmpdirFor()
  try {
    const dir = join(base, 'm')
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(join(dir, 'config.json'), JSON.stringify({ num_hidden_layers: 2 }))
    // 只写出 1/2 个分片 —— 下载中断的另一种形态
    await fs.writeFile(join(dir, 'model-00001-of-00002.safetensors'), safetensorsBuffer(minimalModelTensors()))
    await fs.writeFile(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      metadata: { total_size: 1 },
      weight_map: {
        'model.embed_tokens.weight': 'model-00001-of-00002.safetensors',
        'model.norm.weight': 'model-00002-of-00002.safetensors',
      },
    }))
    const r = await validateModelDir(dir, { policy: TEST_POLICY })
    assert.equal(r.ok, false)
    assert.ok(r.problems.some((p) => p.kind === 'shard-missing'), JSON.stringify(r.problems))
    assert.deepEqual(r.evidence.indexShardsMissing, ['model-00002-of-00002.safetensors'])
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 目录里没有任何权重文件 → E_SOURCE_MODEL_INVALID（不是静默通过）', async () => {
  const base = await tmpdirFor()
  try {
    const dir = join(base, 'm')
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(join(dir, 'config.json'), JSON.stringify({ num_hidden_layers: 2 }))
    const rt = fakeRuntime(base)
    await assert.rejects(
      () => assertSourceModelValid(rt, dir, { policy: TEST_POLICY }),
      (e) => e.code === 'E_SOURCE_MODEL_INVALID' && /没有任何权重文件/.test(e.message),
    )
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 只有 .bin（pickle）→ 诚实声明能力边界，不假装检过', async () => {
  const base = await tmpdirFor()
  try {
    const dir = join(base, 'm')
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(join(dir, 'config.json'), JSON.stringify({ num_hidden_layers: 2 }))
    await fs.writeFile(join(dir, 'pytorch_model.bin'), Buffer.alloc(4096, 5))
    const rt = fakeRuntime(base)
    const rec = await assertSourceModelValid(rt, dir, { policy: TEST_POLICY })
    assert.equal(rec.ok, true)
    assert.equal(rec.unsupported, 'legacy-pickle-bin')
    assert.ok(rt.warnings.some((w) => w.code === 'W_SOURCE_VALIDATION_UNSUPPORTED'),
      '必须留下"未做结构预检"的警告痕迹')
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 残缺模型 → 抛 E_SOURCE_MODEL_INVALID，且 remediation 指向"重新完整下载"', async () => {
  const base = await tmpdirFor()
  try {
    const { dir, buf } = await writeModel(join(base, 'm'), { layers: 4 })
    const headerLen = Number(buf.readBigUInt64LE(0))
    buf.fill(0, 8 + headerLen + 200)
    await fs.writeFile(join(dir, 'model.safetensors'), buf)

    const rt = fakeRuntime(base)
    await assert.rejects(
      () => assertSourceModelValid(rt, dir, { policy: TEST_POLICY }),
      (e) => {
        assert.equal(e.code, 'E_SOURCE_MODEL_INVALID')
        assert.match(e.remediation, /重新完整下载/)
        assert.match(e.remediation, /下载中断/)
        assert.ok(Array.isArray(e.details.problems) && e.details.problems.length > 0)
        assert.ok(e.details.zeroTensorNames.length > 0, 'details 必须点名具体张量')
        return true
      },
    )
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检缓存: 第二次命中缓存（不重扫）；文件一改缓存即失效并重新检出问题', async () => {
  const base = await tmpdirFor()
  try {
    const { dir } = await writeModel(join(base, 'm'))
    const rt = fakeRuntime(base)

    const first = await assertSourceModelValid(rt, dir, { policy: TEST_POLICY })
    assert.equal(first.cached, undefined)
    assert.equal(rt.warnings.length, 0)

    const rt2 = fakeRuntime(base)
    const second = await assertSourceModelValid(rt2, dir, { policy: TEST_POLICY })
    assert.equal(second.cached, true, '应命中缓存')
    assert.ok(rt2.warnings.some((w) => w.code === 'W_SOURCE_VALIDATION_CACHED'))

    // 把模型改坏 → 指纹变化 → 缓存失效 → 必须重新扫出问题
    const p = join(dir, 'model.safetensors')
    const buf = await fs.readFile(p)
    buf.fill(0, 8 + Number(buf.readBigUInt64LE(0)) + 100)
    await fs.writeFile(p, buf)
    await new Promise((r) => setTimeout(r, 12))

    const rt3 = fakeRuntime(base)
    await assert.rejects(
      () => assertSourceModelValid(rt3, dir, { policy: TEST_POLICY }),
      (e) => e.code === 'E_SOURCE_MODEL_INVALID',
    )
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检缓存: 只保留"通过"结论 —— 失败不写缓存，也不会被缓存挡住重扫', async () => {
  const base = await tmpdirFor()
  try {
    const { dir } = await writeModel(join(base, 'm'))
    // 先造一个必然失败的策略：coverage 阈值 0 → 每个大张量都被判"整块为零"
    await assert.rejects(
      () => assertSourceModelValid(fakeRuntime(base), dir, { policy: { ...TEST_POLICY, zeroCoverage: 0 } }),
      (e) => e.code === 'E_SOURCE_MODEL_INVALID',
    )
    // 失败不写缓存 → 用正确策略重扫必须能通过（不能被"看似可用"的负条目挡住）
    const rec = await assertSourceModelValid(fakeRuntime(base), dir, { policy: TEST_POLICY })
    assert.equal(rec.ok, true)
    assert.equal(rec.cached, undefined, '第一次通过应真扫，而不是命中缓存')

    const raw = await fs.readFile(join(base, 'state', 'model-validation.json'), 'utf8')
    const cache = JSON.parse(raw)
    const entries = Object.values(cache.entries)
    assert.ok(entries.length >= 1, '通过结论应落缓存')
    for (const v of entries) assert.equal(v.ok, true, '缓存里不应有失败结论')
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 缺 config.json 也能扫（不因读不到配置就放弃结构检查）', async () => {
  const base = await tmpdirFor()
  try {
    const dir = join(base, 'm')
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(join(dir, 'model.safetensors'), safetensorsBuffer(minimalModelTensors()))
    const r = await validateModelDir(dir, { policy: TEST_POLICY })
    assert.equal(r.ok, true, JSON.stringify(r.problems))
    assert.equal(r.evidence.config, null)
    assert.equal(r.evidence.declaredLayers, undefined)
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})

test('预检: 随机字节不会误报（避免"看到零就报警"的假阳性）', async () => {
  const base = await tmpdirFor()
  try {
    const dir = join(base, 'm')
    await fs.mkdir(dir, { recursive: true })
    const tensors = minimalModelTensors({ layers: 3, hidden: 32, vocab: 64 })
    const buf = safetensorsBuffer(tensors)
    // 数据区用真随机覆盖（保持头部不变）
    const headerLen = Number(buf.readBigUInt64LE(0))
    randomBytes(buf.length - 8 - headerLen).copy(buf, 8 + headerLen)
    await fs.writeFile(join(dir, 'model.safetensors'), buf)
    const r = await scanSafetensors(join(dir, 'model.safetensors'), { policy: TEST_POLICY })
    assert.equal(r.ok, true, JSON.stringify(r.problems))
  } finally { await fs.rm(base, { recursive: true, force: true }) }
})
