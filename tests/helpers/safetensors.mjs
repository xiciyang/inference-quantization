/**
 * 测试用 safetensors 构造器。
 *
 * 为什么测试夹具要产出**真格式**而不是一堆填充字节：
 *   源模型结构预检（src/model-validate.mjs）会真的去解析 safetensors 头部。
 *   夹具若是假的，要么测不出真实路径，要么逼着实现开一个"测试专用绕过开关" ——
 *   而绕过开关一旦存在，就等于"预检可被关闭"，这正是要杜绝的东西。
 *   因此夹具走真格式：**测试与生产跑同一条代码路径**。
 */

const DTYPE_SIZE = { F32: 4, F16: 2, BF16: 2, I8: 1, U8: 1, I32: 4, I64: 8 }

/** 非零且有限的字节模式（小端 f32 = 1.0），避免被"整块为零"规则命中。 */
const PATTERN = [0x00, 0x00, 0x80, 0x3f]

/**
 * 构造一个合法的 safetensors Buffer。
 * @param {{name:string, shape:number[], dtype?:string}[]} tensors
 * @param {{zero?:string[]}} opts `zero` 里的张量会被写成全 0（用于构造"坏模型"夹具）
 */
export function safetensorsBuffer(tensors, { zero = [] } = {}) {
  const header = {}
  const chunks = []
  let offset = 0
  for (const t of tensors) {
    const dtype = t.dtype ?? 'F32'
    const n = t.shape.reduce((a, b) => a * b, 1)
    const size = n * (DTYPE_SIZE[dtype] ?? 1)
    const buf = Buffer.alloc(size)
    if (!zero.includes(t.name)) {
      for (let i = 0; i < size; i++) buf[i] = PATTERN[i % PATTERN.length]
    }
    header[t.name] = { dtype, shape: t.shape, data_offsets: [offset, offset + size] }
    offset += size
    chunks.push(buf)
  }
  const hjson = Buffer.from(JSON.stringify(header), 'utf8')
  const len = Buffer.alloc(8)
  len.writeBigUInt64LE(BigInt(hjson.length))
  return Buffer.concat([len, hjson, ...chunks])
}

/** 一个最小但结构自洽的模型权重集（层号 0..layers-1 与 config 吻合）。 */
export function minimalModelTensors({ layers = 2, hidden = 64, vocab = 128 } = {}) {
  const out = [
    { name: 'model.embed_tokens.weight', shape: [vocab, hidden] },
    { name: 'model.norm.weight', shape: [hidden] },
  ]
  for (let i = 0; i < layers; i++) {
    out.push({ name: `model.layers.${i}.self_attn.q_proj.weight`, shape: [hidden, hidden] })
    out.push({ name: `model.layers.${i}.mlp.down_proj.weight`, shape: [hidden * 2, hidden] })
  }
  return out
}

export { DTYPE_SIZE }
