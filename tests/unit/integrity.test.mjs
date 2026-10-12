import { test } from 'node:test'
import assert from 'node:assert/strict'
import { hashDirectory, assertSourceUnchanged, assertOutputDirSafe } from '../../src/integrity.mjs'
import { ERROR_CODES } from '../../src/errors.mjs'
import { makeSandbox } from '../helpers/sandbox.mjs'
import { promises as fs } from 'node:fs'
import { join } from 'node:path'

test('完整性: 目录哈希对内容敏感，对 mtime/权限不敏感', async () => {
  const sb = await makeSandbox()
  try {
    const h1 = await hashDirectory(sb.modelPath)
    // 改 mtime —— 不应改变哈希
    const now = new Date()
    await fs.utimes(join(sb.modelPath, 'config.json'), now, now)
    const h2 = await hashDirectory(sb.modelPath)
    assert.equal(h1.hash, h2.hash, 'touch 不应被判为"模型被改"')

    // 改内容 —— 必须改变
    const cfg = JSON.parse(await fs.readFile(join(sb.modelPath, 'config.json'), 'utf8'))
    cfg.hidden_size = 999
    await fs.writeFile(join(sb.modelPath, 'config.json'), JSON.stringify(cfg))
    const h3 = await hashDirectory(sb.modelPath)
    assert.notEqual(h1.hash, h3.hash, '内容改动必须被检出')
  } finally { await sb.cleanup() }
})

test('完整性: 文件增删与重命名也会改变哈希', async () => {
  const sb = await makeSandbox()
  try {
    const h1 = await hashDirectory(sb.modelPath)
    await fs.writeFile(join(sb.modelPath, 'extra.json'), '{}')
    const h2 = await hashDirectory(sb.modelPath)
    assert.notEqual(h1.hash, h2.hash, '新增文件必须被检出')
    await fs.rm(join(sb.modelPath, 'extra.json'))
    const h3 = await hashDirectory(sb.modelPath)
    assert.equal(h1.hash, h3.hash, '删回原样应恢复同一哈希')
  } finally { await sb.cleanup() }
})

test('完整性: assertSourceUnchanged —— 未改动通过、改动了就中止', async () => {
  const sb = await makeSandbox()
  try {
    const before = await hashDirectory(sb.modelPath)
    const ok = await assertSourceUnchanged(sb.modelPath, before)
    assert.equal(ok.unchanged, true)

    await fs.writeFile(join(sb.modelPath, 'model.safetensors'), Buffer.alloc(16, 1))
    await assert.rejects(() => assertSourceUnchanged(sb.modelPath, before), (err) => {
      assert.equal(err.code, ERROR_CODES.E_SOURCE_MODEL_MODIFIED)
      assert.match(err.remediation, /只读/)
      assert.ok(err.details.beforeHash && err.details.afterHash)
      return true
    })
  } finally { await sb.cleanup() }
})

test('完整性: 输出目录安全检查 —— 等于/内部/祖先 三种都要拦', () => {
  assert.equal(assertOutputDirSafe('/data/models/m', '/data/models/m-awq'), true)
  assert.equal(assertOutputDirSafe('/data/models/m', '/tmp/out'), true)

  for (const [src, out] of [
    ['/data/models/m', '/data/models/m'],
    ['/data/models/m', '/data/models/m/sub'],
    ['/data/models/m', '/data/models'],
  ]) {
    assert.throws(() => assertOutputDirSafe(src, out), (err) => {
      assert.equal(err.code, ERROR_CODES.E_OUTPUT_DIR_UNSAFE)
      assert.ok(err.remediation)
      return true
    }, `${out} 相对 ${src} 应被拦`)
  }
})

test('完整性: 大目录哈希包含文件数与字节数（便于报告）', async () => {
  const sb = await makeSandbox()
  try {
    const h = await hashDirectory(sb.modelPath)
    assert.ok(h.files >= 3)
    assert.ok(h.bytes > 0)
    assert.equal(h.hash.length, 64)
  } finally { await sb.cleanup() }
})

test('完整性: 流式哈希 —— 大文件不受 2 GiB buffer 上限影响', async () => {
  const { hashFileStreaming } = await import('../../src/integrity.mjs')
  const sb = await makeSandbox()
  try {
    // 造一个 3 GiB 的稀疏文件（不真占磁盘），验证不会抛 ERR_FS_FILE_TOO_LARGE
    const big = join(sb.base, 'big.safetensors')
    const fh = await fs.open(big, 'w')
    await fh.truncate(3 * 1024 ** 3)
    await fh.close()
    const r = await hashFileStreaming(big)
    assert.equal(r.size, 3 * 1024 ** 3)
    assert.equal(r.hash.length, 64)

    // 对照：fs.readFile 在这个尺寸上会抛 —— 说明我们确实绕开了它
    await assert.rejects(() => fs.readFile(big), (err) => err.code === 'ERR_FS_FILE_TOO_LARGE')
  } finally { await sb.cleanup() }
})

test('完整性: 目录哈希会走流式路径处理大文件', async () => {
  const sb = await makeSandbox()
  try {
    const big = join(sb.modelPath, 'huge.safetensors')
    const fh = await fs.open(big, 'w')
    await fh.truncate(2 * 1024 ** 3 + 1024)
    await fh.close()
    const h = await hashDirectory(sb.modelPath)
    assert.ok(h.bytes > 2 * 1024 ** 3)
    assert.equal(h.hash.length, 64)
  } finally { await sb.cleanup() }
})
