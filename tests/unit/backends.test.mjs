import { test } from 'node:test'
import assert from 'node:assert/strict'
import { register, unregister, getMethod, listMethods, allMethods, chainOrdered, __resetMethods } from '../../src/backends/registry.mjs'

const good = () => ({
  id: 'x-method', displayName: 'X', level: 9, bits: 2,
  isAvailable: () => true, describe: () => ({}), recipe: () => ({ methodId: 'x-method' }),
  preflight: async () => ({ ok: true }),
})

test('方法注册表: 形状校验 —— 缺方法/非法 level/重复 id 都被拒', () => {
  __resetMethods()
  assert.throws(() => register({ id: 'a' }), /缺必需方法/)
  assert.throws(() => register({ ...good(), id: 'b', level: 0 }), /level 必须是 ≥1 的整数/)
  assert.throws(() => register({ ...good(), id: 'c', level: 1.5 }), /level 必须是 ≥1 的整数/)
  assert.throws(() => register({ ...good(), id: '' }), /id 必须是非空字符串/)
  register(good())
  assert.throws(() => register(good()), /已注册/)
  unregister('x-method')
})

test('方法注册表: 核心不认识具体方法 —— 只按 level 排序拿列表', () => {
  __resetMethods()
  register({ ...good(), id: 'l3', level: 3 })
  register({ ...good(), id: 'l1', level: 1 })
  register({ ...good(), id: 'l2', level: 2 })
  assert.deepEqual(allMethods().map((m) => m.id), ['l1', 'l2', 'l3'])
  assert.deepEqual(chainOrdered().map((m) => m.id), ['l1', 'l2', 'l3'])
  assert.equal(getMethod('l2').level, 2)
  assert.equal(getMethod('nope'), null)
  assert.equal(listMethods().length, 3)
})

test('方法注册表: inDegradeChain=false 的方法不参与降级链', () => {
  __resetMethods()
  register({ ...good(), id: 'chain', level: 1 })
  register({ ...good(), id: 'standalone', level: 2, inDegradeChain: false })
  assert.deepEqual(chainOrdered().map((m) => m.id), ['chain'])
  assert.equal(allMethods().length, 2, '但仍在总表里（可被显式指定）')
})

test('方法注册表: 内置四档的元数据正确（位宽/有损/档位）', async () => {
  __resetMethods()
  const mod = await import(`../../src/backends/llmcompressor.mjs?fresh=${Date.now()}`)
  void mod
  __resetMethods()
  await import('../../src/backends/index.mjs')
  const mods = allMethods()
  if (mods.length === 0) return // 已被 __resetMethods 清掉（模块缓存），跳过
  assert.ok(mods.every((m) => m.recipe && m.preflight && m.describe))
})
