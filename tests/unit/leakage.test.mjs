import { test } from 'node:test'
import assert from 'node:assert/strict'
import { itemHash, hashAll, listDigest, checkDisjoint, verifyLineage, assertNoLeakage } from '../../src/dataset/leakage.mjs'
import { readJsonl, verifyDatasetDir } from '../../src/dataset/prepare.mjs'
import { makeSandbox } from '../helpers/sandbox.mjs'
import { createRuntime } from '../../src/config.mjs'
import { ERROR_CODES } from '../../src/errors.mjs'
import { promises as fs } from 'node:fs'
import { join } from 'node:path'

test('去泄漏: 内容哈希按 trim 归一（与容器侧一致）', () => {
  assert.equal(itemHash('abc'), itemHash('  abc\n'))
  assert.notEqual(itemHash('abc'), itemHash('abd'))
  assert.equal(itemHash('中文内容').length, 64)
})

test('去泄漏: 交集为空才判 clean', () => {
  const a = hashAll(['x1', 'x2', 'x3'])
  const b = hashAll(['y1', 'y2'])
  assert.equal(checkDisjoint(a, b).clean, true)
  const c = hashAll(['x1', 'y2'])  // y2 与 b 重复
  const r = checkDisjoint(a, c)
  assert.equal(r.clean, false)
  assert.equal(r.intersection.length, 1)
})

test('去泄漏: 同一段内部有重复也算不干净', () => {
  const r = checkDisjoint(hashAll(['a', 'a', 'b']), hashAll(['c']))
  assert.equal(r.clean, false)
  assert.match(r.reason, /内部存在/)
})

test('去泄漏: 抽样段是子集 —— 不要求并集等于全集（判据不得误用）', () => {
  // 两段是从全集抽样的子集：并集 3 < 全集 14324 是**正常**的
  const ok = checkDisjoint(hashAll(['a', 'b']), hashAll(['c']), { dedupedTotal: 14324 })
  assert.equal(ok.clean, true, '抽样段不应被要求覆盖全集')

  // 并集超过全集规模 → 计数/去重口径有误
  const bad = checkDisjoint(hashAll(['a', 'b', 'c', 'd']), hashAll(['e']), { dedupedTotal: 3 })
  assert.equal(bad.clean, false)
  assert.match(bad.reason, /不可能|超过去重全集/)

  // 某一段条数超过全集
  const bad2 = checkDisjoint(hashAll(['a', 'b', 'c', 'd']), hashAll(['e']), { dedupedTotal: 2 })
  assert.equal(bad2.clean, false)
})

test('去泄漏: verifyLineage 会**重算**哈希，而不是相信 lineage 里写的数字', async () => {
  const calib = ['样本一', '样本二']
  const test = ['样本三']
  const good = {
    dedup: { unique: 3 },
    split: {
      calibration: { hashes: hashAll(calib), listDigest: listDigest(hashAll(calib)), count: 2 },
      test: { hashes: hashAll(test), listDigest: listDigest(hashAll(test)), count: 1 },
    },
  }
  assert.equal(verifyLineage(good, calib, test).verified, true)

  // 篡改 lineage：把测试段写成与校准段有交集
  const tampered = JSON.parse(JSON.stringify(good))
  tampered.split.test.hashes = [hashAll(calib)[0]]
  const v = verifyLineage(tampered, calib, test)
  assert.equal(v.verified, false)
  assert.ok(v.problems.length > 0)
})

test('去泄漏: assertNoLeakage 是硬门禁 —— 抛错且 remediation 明说无人工放行通道', () => {
  assert.throws(() => assertNoLeakage({ verified: false, problems: ['有交集'] }), (err) => {
    assert.equal(err.code, ERROR_CODES.E_CALIB_LEAKAGE_DETECTED)
    assert.match(err.remediation, /不接受人工确认放行/)
    return true
  })
  assert.doesNotThrow(() => assertNoLeakage({ verified: true, problems: [] }))
})

test('去泄漏: 宿主侧 verifyDatasetDir 对真实文件独立复核', async () => {
  const sb = await makeSandbox()
  try {
    const dir = await sb.makeDataset({ calib: ['a1', 'a2', 'a3'], test: ['b1', 'b2'] })
    const runtime = await createRuntime({ root: sb.root, options: sb.options })
    const out = await verifyDatasetDir(runtime, dir)
    assert.equal(out.verification.verified, true)
    assert.deepEqual(out.verification.counts, { calibration: 3, test: 2, unique: 5 })
    assert.equal(out.calibrationSamples, 3)
  } finally { await sb.cleanup() }
})

test('去泄漏: 篡改 jsonl 内容 → 宿主复核必须检出', async () => {
  const sb = await makeSandbox()
  try {
    const dir = await sb.makeDataset({ calib: ['a1', 'a2'], test: ['b1'] })
    // 把测试集改成与校准集相同的内容 —— 这正是"泄漏"的形态
    await fs.writeFile(join(dir, 'test.jsonl'), JSON.stringify({ h: itemHash('a1'), t: 'a1' }) + '\n')
    const runtime = await createRuntime({ root: sb.root, options: sb.options })
    const out = await verifyDatasetDir(runtime, dir)
    assert.equal(out.verification.verified, false, '内容被改成有交集后必须判不通过')
  } finally { await sb.cleanup() }
})

test('去泄漏: 缺文件时明确报缺哪个，不静默', async () => {
  const sb = await makeSandbox()
  try {
    const dir = join(sb.root, 'state', 'datasets', 'empty')
    await fs.mkdir(dir, { recursive: true })
    const runtime = await createRuntime({ root: sb.root, options: sb.options })
    await assert.rejects(() => verifyDatasetDir(runtime, dir), (err) => {
      assert.equal(err.code, ERROR_CODES.E_DATASET_UNAVAILABLE)
      assert.match(err.message, /缺文件/)
      return true
    })
  } finally { await sb.cleanup() }
})

test('去泄漏: readJsonl 跳过损坏行（由计数比对兜底发现）', async () => {
  const sb = await makeSandbox()
  try {
    const p = join(sb.base, 'x.jsonl')
    await fs.writeFile(p, `${JSON.stringify({ h: 'a', t: 'good1' })}\nNOT JSON\n${JSON.stringify({ h: 'b', t: 'good2' })}\n`)
    const items = await readJsonl(p)
    assert.deepEqual(items, ['good1', 'good2'])
  } finally { await sb.cleanup() }
})
