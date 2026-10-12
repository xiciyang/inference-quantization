/**
 * 测试脚手架：隔离的共享状态根 + 假模型目录。
 */
import { promises as fs } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { randomBytes } from 'node:crypto'
import { safetensorsBuffer, minimalModelTensors } from './safetensors.mjs'

export async function makeSandbox({ modelName = 'test-model', modelRootName = 'models' } = {}) {
  const base = await fs.mkdtemp(join(tmpdir(), 'iq-test-'))
  const root = join(base, 'ai-infra')
  const modelRoot = join(base, modelRootName)
  const modelPath = join(modelRoot, modelName)
  await fs.mkdir(modelPath, { recursive: true })
  await fs.writeFile(join(modelPath, 'config.json'), JSON.stringify({
    architectures: ['LlamaForCausalLM'], model_type: 'llama', max_position_embeddings: 4096,
    num_hidden_layers: 2, hidden_size: 64, torch_dtype: 'bfloat16',
  }))
  // 真格式的 safetensors：源模型结构预检会真的解析它（夹具与生产同一条代码路径）
  await fs.writeFile(join(modelPath, 'model.safetensors'), safetensorsBuffer(minimalModelTensors()))
  await fs.writeFile(join(modelPath, 'LICENSE'), 'Apache License 2.0\n')

  const poolStart = 19_000 + Math.floor(Math.random() * 500) * 20
  return {
    base, root, modelRoot, modelPath, modelName,
    options: { modelRoots: [modelRoot], bindHost: '127.0.0.1', portPoolStart: poolStart, portPoolEnd: poolStart + 19 },
    async cleanup() { await fs.rm(base, { recursive: true, force: true }) },
    async readMatrixText() {
      try { return await fs.readFile(join(root, 'state', 'quantization-matrix.yaml'), 'utf8') } catch { return null }
    },
    async readMatrix() {
      const t = await this.readMatrixText()
      if (!t) return null
      const { parse } = await import('../../src/util/yaml.mjs')
      return parse(t)
    },
    async makeDataset({ calib = [], test = [] } = {}) {
      const { stringify } = await import('../../src/util/yaml.mjs')
      const dir = join(root, 'state', 'datasets', 'test-ds')
      await fs.mkdir(dir, { recursive: true })
      const { itemHash, listDigest } = await import('../../src/dataset/leakage.mjs')
      const write = async (name, items) => {
        await fs.writeFile(join(dir, name), items.map((t) => JSON.stringify({ h: itemHash(t), t })).join('\n') + '\n')
      }
      await write('calib.jsonl', calib)
      await write('test.jsonl', test)
      const ch = calib.map(itemHash); const th = test.map(itemHash)
      await fs.writeFile(join(dir, 'lineage.json'), JSON.stringify({
        formatVersion: 1,
        source: { repo: 'test/repo', config: 'cfg', split: 'train', revision: 'abc' },
        dedup: { unique: new Set([...ch, ...th]).size, dropped: 0 },
        split: {
          seed: 1, algorithm: 'test',
          calibration: { count: ch.length, listDigest: listDigest(ch), hashes: ch },
          test: { count: th.length, listDigest: listDigest(th), hashes: th },
        },
        leakage: { criterion: 'content-sha256-intersection-empty', intersection: 0, clean: true },
      }))
      void stringify
      return dir
    },
  }
}

export function uniqueName(prefix = 'x') { return `${prefix}-${randomBytes(4).toString('hex')}` }
export { fs, join, tmpdir }
