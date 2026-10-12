/**
 * 源模型可用性预检 —— "文件存在且没被改过"**不等于**"权重是真的"。
 *
 * 为什么必须单独补这一层（真机踩坑后补建，2026-09-23）：
 *   量化专项跑出过一次"精度零损失"的签收。事后定位到根因是**源模型文件本身残缺**：
 *   `model.safetensors` 只有前 0.747 GiB 是真数据，其后 2.128 GiB 全是 0x00
 *   （下载中断留下的空洞，目录里还留着 `model.safetensors.download.aria2`）。
 *   旧流程只做了两件事 —— 文件可读、跑完后目录哈希未变 —— 两件都通过，
 *   于是把一份 271/338 张量全为零的模型量化了 436 秒，并给"精度零损失"签了字。
 *
 *   目录哈希回答的是"内容**有没有被改动**"，**回答不了"内容是不是一个完整的模型"**。
 *   本模块补的正是后者，且刻意排在量化之前：坏输入要**最便宜地**失败。
 *
 * 检查口径（每条都出证据，不猜）：
 *   ① 截断     —— safetensors 头部声明的数据终点 ≠ 文件实际长度。
 *   ② 零填充   —— 单个张量 ≥ `zeroCoverage` 比例的字节为 0，且元素数 ≥ `minElements`。
 *      元素数下限的理由：长度 1 的 bias 恰好为 0 在真实模型里很常见；
 *      而 1536×8960 的权重矩阵全为 0 不可能是有意为之。
 *   ③ 结构     —— config.json 声明的层数 与 权重里实际出现的 `model.layers.N` 是否吻合。
 *   ④ 分片     —— 有 `*.index.json` 时必须列全所有分片且每个都存在。
 *
 * 结果按 (路径, 字节数, mtimeMs) 缓存：同一份模型重复跑不必重复扫 3 GB。
 * 缓存键含 size+mtime，文件被改动即自动失效 —— 不会把旧结论套到新文件上。
 */
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { fail, warn, ERROR_CODES, WARNING_CODES } from './errors.mjs'
import { readText, atomicWriteJson, mkdirp } from './util/fsx.mjs'

export const VALIDATION_SCHEMA = 'inference-quantization/model-validation@1'

/** 零填充判定参数（默认值随报告一起输出，可被 options 覆盖）。 */
export const DEFAULT_ZERO_POLICY = Object.freeze({
  minElements: 1024,
  zeroCoverage: 0.999,
  chunkBytes: 4 * 1024 * 1024,
  maxZeroTensorNames: 32,
})

/** 缓存保留条数上限（LRU 语义按写入顺序截断）。 */
const CACHE_LIMIT = 200

function invalid(kind, message, details) {
  return { ok: false, kind, message, details: details ?? null, problems: [], evidence: null }
}

/**
 * 线性扫描一个 safetensors 文件。
 *
 * 为什么是**线性整读**而不是"每个张量抽样几个窗口"：
 *   抽样的漏检面是"张量内部部分为零"，而下载中断造成的正是**一整段连续零区**；
 *   线性扫描用一次顺序读（NVMe 上 3 GB ≈ 1~2 s）换掉整类漏检，代价可接受，
 *   也让"零区起止字节"这个证据可以直接写进报告。
 */
export async function scanSafetensors(path, { policy = DEFAULT_ZERO_POLICY } = {}) {
  const p = { ...DEFAULT_ZERO_POLICY, ...(policy ?? {}) }
  let st
  try {
    st = await fs.stat(path)
  } catch (err) {
    return invalid('unreadable', `无法 stat：${err.message}`, { path })
  }
  const size = st.size

  const fh = await fs.open(path, 'r')
  try {
    if (size < 16) return invalid('header-invalid', `文件过小（${size} 字节），不可能是 safetensors`, { path, size })

    const head = Buffer.alloc(8)
    await fh.read(head, 0, 8, 0)
    const headerLen = Number(head.readBigUInt64LE(0))
    if (!Number.isSafeInteger(headerLen) || headerLen <= 0 || 8 + headerLen > size) {
      return invalid('header-invalid', `safetensors 头部长度非法：${headerLen}（文件 ${size} 字节）`, { path, size, headerLen })
    }
    const hbuf = Buffer.alloc(headerLen)
    await fh.read(hbuf, 0, headerLen, 8)
    let header
    try {
      header = JSON.parse(hbuf.toString('utf8'))
    } catch (err) {
      return invalid('header-invalid', `safetensors 头部不是合法 JSON：${err.message}`, { path, size, headerLen })
    }
    const metadata = header.__metadata__ ?? null
    delete header.__metadata__

    const entries = Object.entries(header).map(([name, v]) => ({
      name,
      dtype: v?.dtype ?? null,
      shape: Array.isArray(v?.shape) ? v.shape : [],
      start: Number(v?.data_offsets?.[0] ?? 0),
      end: Number(v?.data_offsets?.[1] ?? 0),
    }))
    if (entries.length === 0) return invalid('header-invalid', 'safetensors 头部没有任何张量', { path, size, headerLen })

    const dataStart = 8 + headerLen
    const declaredEnd = dataStart + entries.reduce((m, e) => Math.max(m, e.end), 0)

    // ---- 逐张量精确零覆盖扫描 ----
    //
    // 口径演进（写测试时发现的一个真实精度缺陷）：
    //   最早的实现是"按 chunkSize 线性扫全文件、再把零区与张量求交"。
    //   那会让**小于一个 chunk 的零区彻底隐身** —— 例如 1536×2 字节的 `model.norm.weight`
    //   若单独被写坏，整文件里就不存在 ≥ chunkSize 的连续零区，于是判不出来。
    //   而它恰恰是本次事故里最要命的那一个张量。
    // 现在改成**按张量各自的字节区间读**：总读取量不变（数据区恰好读一遍），
    // 但小于一个 chunk 的张量会被整块读入并逐字节比较 —— 结果是精确的。
    // 大于一个 chunk 的张量按 chunk 切分，零覆盖比例的分辨率是 chunkSize/tensorBytes。
    const sorted = [...entries].filter((e) => e.end > e.start).sort((a, b) => a.start - b.start)
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].start < sorted[i - 1].end) {
        return invalid('header-invalid', `张量数据区重叠：${sorted[i - 1].name} 与 ${sorted[i].name}`, { path, size })
      }
    }

    const zeroBuf = Buffer.alloc(p.chunkBytes)
    const buf = Buffer.allocUnsafe(p.chunkBytes)
    const regions = [] // 连续零区（绝对文件偏移），跨张量合并
    const zeroTensors = []
    let zeroBytes = 0
    for (const e of sorted) {
      const len = e.end - e.start
      const from = dataStart + e.start
      const to = dataStart + e.end
      let tensorZero = 0
      let pos = from
      while (pos < to) {
        const want = Math.min(p.chunkBytes, to - pos)
        // eslint-disable-next-line no-await-in-loop
        const { bytesRead } = await fh.read(buf, 0, want, pos)
        if (bytesRead <= 0) break
        if (buf.compare(zeroBuf, 0, bytesRead, 0, bytesRead) === 0) {
          tensorZero += bytesRead
          const last = regions[regions.length - 1]
          if (last && last.end === pos) last.end = pos + bytesRead
          else regions.push({ start: pos, end: pos + bytesRead })
        }
        pos += bytesRead
      }
      zeroBytes += tensorZero
      const elements = e.shape.reduce((a, b) => a * b, 1)
      if (elements >= p.minElements && tensorZero / len >= p.zeroCoverage) {
        zeroTensors.push({
          name: e.name, shape: e.shape, dtype: e.dtype,
          elements, bytes: len,
          zeroBytes: tensorZero,
          zeroRatio: Number((tensorZero / len).toFixed(6)),
        })
      }
    }

    const largest = regions.reduce((m, r) => (!m || r.end - r.start > m.end - m.start ? r : m), null)
    const truncatedBy = declaredEnd - size

    const evidence = {
      path,
      size,
      mtimeMs: st.mtimeMs,
      headerBytes: headerLen,
      metadata,
      tensorCount: entries.length,
      dataStart,
      declaredEnd,
      // 正数 = 文件比头部声明短（截断）；负数 = 文件尾部有多余字节
      truncatedBy,
      trailingBytes: truncatedBy < 0 ? -truncatedBy : 0,
      zeroBytes,
      zeroRatio: (size - dataStart) > 0 ? Number((zeroBytes / (size - dataStart)).toFixed(6)) : 0,
      largestZeroRegion: largest ? { start: largest.start, end: largest.end, bytes: largest.end - largest.start } : null,
      zeroTensorCount: zeroTensors.length,
      zeroTensors: zeroTensors.slice(0, p.maxZeroTensorNames),
      zeroTensorsTruncated: Math.max(0, zeroTensors.length - p.maxZeroTensorNames),
      zeroPolicy: { minElements: p.minElements, zeroCoverage: p.zeroCoverage, chunkBytes: p.chunkBytes },
    }

    const problems = []
    if (truncatedBy > 0) problems.push({ kind: 'truncated', message: `文件比头部声明短 ${truncatedBy} 字节（缺 ${(truncatedBy / 2 ** 20).toFixed(1)} MiB）` })
    if (zeroTensors.length > 0) {
      problems.push({
        kind: 'zero-filled',
        message: `${zeroTensors.length}/${entries.length} 个张量整块为零（占数据区 ${(evidence.zeroRatio * 100).toFixed(1)}%）`,
      })
    }

    return { ok: problems.length === 0, problems, evidence }
  } finally {
    await fh.close()
  }
}

/** 从 config.json 读结构信息（读不到不算失败 —— 有的模型目录没有 config）。 */
async function readModelConfig(dir) {
  const raw = await readText(join(dir, 'config.json'), null)
  if (!raw) return null
  try {
    return JSON.parse(raw)
  } catch {
    return null
  }
}

/**
 * 校验一个模型目录。
 * @returns {{ok, problems, files, evidence, unsupported}}
 */
export async function validateModelDir(dir, { policy = DEFAULT_ZERO_POLICY, log = null } = {}) {
  let files
  try {
    files = await fs.readdir(dir)
  } catch (err) {
    fail(ERROR_CODES.E_SOURCE_MODEL_UNREADABLE, `无法读取源模型目录：${dir}`, {
      remediation: '确认路径存在且当前用户可读。', details: { dir, error: err.message }, cause: err,
    })
  }

  const shards = files.filter((f) => f.endsWith('.safetensors')).sort()
  const indexFile = files.filter((f) => f.endsWith('.safetensors.index.json')).sort()[0] ?? null
  const legacyBins = files.filter((f) => f.endsWith('.bin')).sort()
  const config = await readModelConfig(dir)

  const problems = []
  const evidence = {
    dir,
    config: config ? { architectures: config.architectures ?? null, numHiddenLayers: config.num_hidden_layers ?? null, vocabSize: config.vocab_size ?? null, tieWordEmbeddings: config.tie_word_embeddings ?? null } : null,
    indexFile,
    shards: [],
    legacyBins,
    unsupported: null,
  }

  // ④ 分片完整性：index.json 列了哪些分片，就得到齐
  if (indexFile) {
    const raw = await readText(join(dir, indexFile), null)
    if (raw) {
      try {
        const idx = JSON.parse(raw)
        const listed = [...new Set(Object.values(idx.weight_map ?? {}).map((v) => String(v).split('/').pop()))].sort()
        const missing = listed.filter((f) => !files.includes(f))
        evidence.indexShardsListed = listed.length
        evidence.indexShardsMissing = missing
        if (missing.length > 0) {
          problems.push({ kind: 'shard-missing', message: `index.json 列了 ${listed.length} 个分片，缺 ${missing.length} 个：${missing.slice(0, 5).join(', ')}` })
        }
      } catch (err) {
        problems.push({ kind: 'index-unparsable', message: `${indexFile} 不是合法 JSON：${err.message}` })
      }
    }
  }

  if (shards.length === 0 && legacyBins.length > 0) {
    // 诚实声明能力边界：不做 pickle 解析（它需要 unsafe 反序列化，本插件零依赖且拒绝该风险）
    evidence.unsupported = 'legacy-pickle-bin'
    log?.info('源模型只有 .bin（PyTorch pickle），本插件不做结构预检', { dir, bins: legacyBins.length })
    return { ok: true, problems, files, evidence, unsupported: 'legacy-pickle-bin' }
  }

  if (shards.length === 0) {
    fail(ERROR_CODES.E_SOURCE_MODEL_INVALID, `源模型目录里没有任何权重文件：${dir}`, {
      remediation: '确认这是完整的模型目录（应有 *.safetensors，或至少 *.bin）。'
        + '目录内容清单见 details.files。',
      details: { dir, files: files.slice(0, 60) },
    })
  }

  const allTensorNames = new Set()
  let totalZeroTensors = 0
  let totalTensors = 0
  let totalZeroBytes = 0
  let totalDataBytes = 0
  for (const s of shards) {
    log?.info('源模型结构预检', { file: s })
    // eslint-disable-next-line no-await-in-loop
    const r = await scanSafetensors(join(dir, s), { policy })
    evidence.shards.push(r.evidence ? { file: s, ...r.evidence } : { file: s, error: r.message })
    if (!r.evidence) {
      problems.push({ kind: r.kind, message: `${s}：${r.message}` })
      continue
    }
    for (const t of r.evidence.zeroTensors) allTensorNames.add(t.name)
    totalZeroTensors += r.evidence.zeroTensorCount
    totalTensors += r.evidence.tensorCount
    totalZeroBytes += r.evidence.zeroBytes
    totalDataBytes += r.evidence.size - r.evidence.dataStart
    for (const pb of r.problems) problems.push({ kind: pb.kind, file: s, message: `${s}：${pb.message}` })
  }
  evidence.totalTensors = totalTensors
  evidence.totalZeroTensors = totalZeroTensors
  evidence.totalZeroBytes = totalZeroBytes
  evidence.totalDataBytes = totalDataBytes
  evidence.zeroTensorNames = [...allTensorNames].slice(0, DEFAULT_ZERO_POLICY.maxZeroTensorNames)

  // ③ 层数吻合（只在能读出 config 且权重确实是 HF 命名时判）
  const layers = new Set()
  for (const s of evidence.shards) {
    for (const t of s.zeroTensors ?? []) {
      const m = /(?:^|\.)(?:model\.)?layers\.(\d+)\./.exec(t.name)
      if (m) layers.add(Number(m[1]))
    }
  }
  const declared = evidence.config?.numHiddenLayers ?? null
  if (declared && evidence.shards.length > 0) {
    // 用首分片的头部张量名判层：零张量不足以代表全集，故再扫一次名字
    const names = new Set()
    for (const s of shards) {
      // eslint-disable-next-line no-await-in-loop
      const raw = await readSafetensorsNames(join(dir, s))
      for (const n of raw) names.add(n)
    }
    const present = new Set()
    for (const n of names) {
      const m = /(?:^|\.)(?:model\.)?(?:layers|h)\.(\d+)\./.exec(n)
      if (m) present.add(Number(m[1]))
    }
    evidence.declaredLayers = declared
    evidence.presentLayers = present.size
    evidence.layerIndicesMax = present.size ? Math.max(...present) : null
    if (present.size > 0 && present.size !== declared) {
      problems.push({
        kind: 'layer-count-mismatch',
        message: `config.json 声明 ${declared} 层，权重里只有 ${present.size} 层（最大层号 ${evidence.layerIndicesMax}）`,
      })
    }
  }

  return { ok: problems.length === 0, problems, files, evidence, unsupported: null }
}

/** 只读 safetensors 头部的张量名列表（不读数据区，快）。 */
export async function readSafetensorsNames(path) {
  const fh = await fs.open(path, 'r')
  try {
    const head = Buffer.alloc(8)
    await fh.read(head, 0, 8, 0)
    const headerLen = Number(head.readBigUInt64LE(0))
    if (!Number.isSafeInteger(headerLen) || headerLen <= 0) return []
    const hbuf = Buffer.alloc(headerLen)
    await fh.read(hbuf, 0, headerLen, 8)
    const header = JSON.parse(hbuf.toString('utf8'))
    delete header.__metadata__
    return Object.keys(header)
  } catch {
    return []
  } finally {
    await fh.close()
  }
}

// ---------------------------------------------------------------- 缓存

async function loadCache(file) {
  const raw = await readText(file, null)
  if (!raw) return {}
  try {
    const j = JSON.parse(raw)
    return j && typeof j.entries === 'object' && j.entries ? j.entries : {}
  } catch {
    return {}
  }
}

async function saveCache(file, entries) {
  const keys = Object.keys(entries)
  const trimmed = keys.length > CACHE_LIMIT ? Object.fromEntries(keys.slice(-CACHE_LIMIT).map((k) => [k, entries[k]])) : entries
  await mkdirp(join(file, '..'))
  await atomicWriteJson(file, { schema: `${VALIDATION_SCHEMA}#cache`, updatedAt: new Date().toISOString(), entries: trimmed })
}

/** 目录指纹：以各权重文件的 (名字, 字节数, mtime) 组成，内容一变缓存即失效。 */
async function dirFingerprint(dir, files) {
  const wanted = files.filter((f) => /\.(safetensors|bin)$/.test(f) || f.endsWith('.index.json')).sort()
  const parts = []
  for (const f of wanted) {
    // eslint-disable-next-line no-await-in-loop
    const st = await fs.stat(join(dir, f)).catch(() => null)
    parts.push(`${f}:${st ? st.size : 'missing'}:${st ? Math.round(st.mtimeMs) : 0}`)
  }
  return parts.join('|')
}

/**
 * 断言源模型可用。不可用 → `E_SOURCE_MODEL_INVALID`（终态失败，换档位没用）。
 *
 * @param {object} runtime
 * @param {string} dir
 * @param {{policy?, useCache?, force?: boolean}} opts
 */
export async function assertSourceModelValid(runtime, dir, { policy = DEFAULT_ZERO_POLICY, useCache = true, force = false } = {}) {
  const cacheFile = runtime?.paths?.modelValidationFile ?? null
  let files = []
  try {
    files = await fs.readdir(dir)
  } catch (err) {
    fail(ERROR_CODES.E_SOURCE_MODEL_UNREADABLE, `无法读取源模型目录：${dir}`, {
      remediation: '确认路径存在且当前用户可读。', details: { dir, error: err.message }, cause: err,
    })
  }
  const fingerprint = await dirFingerprint(dir, files)
  const key = `${dir}\u0000${fingerprint}`

  if (useCache && cacheFile && !force) {
    const cache = await loadCache(cacheFile)
    const hit = cache[key]
    if (hit && hit.ok === true) {
      runtime?.warnings?.push?.(warn(WARNING_CODES.W_SOURCE_VALIDATION_CACHED,
        `源模型结构预检命中缓存（${hit.validatedAt}）：未重复扫描 ${((hit.evidence?.totalDataBytes ?? 0) / 2 ** 20).toFixed(0)} MiB`,
        { dir, validatedAt: hit.validatedAt }))
      runtime?.__log?.info?.('源模型结构预检：命中缓存', { dir, validatedAt: hit.validatedAt })
      return { ...hit, cached: true }
    }
  }

  const result = await validateModelDir(dir, { policy, log: runtime?.__log ?? null })
  const record = {
    schema: VALIDATION_SCHEMA,
    ok: result.ok,
    validatedAt: new Date().toISOString(),
    unsupported: result.unsupported,
    problems: result.problems,
    evidence: {
      dir: result.evidence.dir,
      config: result.evidence.config,
      totalTensors: result.evidence.totalTensors,
      totalZeroTensors: result.evidence.totalZeroTensors,
      totalZeroBytes: result.evidence.totalZeroBytes,
      totalDataBytes: result.evidence.totalDataBytes,
      zeroTensorNames: result.evidence.zeroTensorNames,
      declaredLayers: result.evidence.declaredLayers ?? null,
      presentLayers: result.evidence.presentLayers ?? null,
      shards: (result.evidence.shards ?? []).map((s) => ({
        file: s.file, size: s.size, tensorCount: s.tensorCount, declaredEnd: s.declaredEnd,
        truncatedBy: s.truncatedBy, zeroBytes: s.zeroBytes, zeroRatio: s.zeroRatio,
        largestZeroRegion: s.largestZeroRegion, zeroTensorCount: s.zeroTensorCount,
      })),
    },
  }

  if (!result.ok) {
    const zero = result.problems.find((p) => p.kind === 'zero-filled')
    const trunc = result.problems.find((p) => p.kind === 'truncated')
    const hint = zero
      ? '权重里有整块为零的张量 —— 典型成因是**下载中断**（目录里常留有 *.aria2 / *.part 残留文件）。'
      : trunc
        ? '权重文件比 safetensors 头部声明短 —— 下载未完成。'
        : '源模型结构不完整。'
    fail(ERROR_CODES.E_SOURCE_MODEL_INVALID, `源模型不可用（${result.problems.map((p) => p.kind).join('、')}）：${dir}`, {
      remediation: `${hint}请重新完整下载该模型（例如 modelscope download / hf download --resume），`
        + '或用 options.source_model 指向另一份**已验证完整**的副本后重试。'
        + '本插件不会在一份结构残缺的模型上做量化 —— 那会产出一个看似"零精度损失"、'
        + '实际毫无意义的签收。证据见 details 与报告里的 sourceValidation 字段。',
      details: {
        dir,
        problems: result.problems,
        zeroTensorNames: record.evidence.zeroTensorNames,
        shards: record.evidence.shards,
        hint,
      },
    })
  }

  // **只有通过才写缓存**：负结论不入缓存。
  // （缓存键虽含 size+mtime、文件一改即失效，但不留一个"看起来可用"的失败条目更干净；
  //  读取侧也只接受 ok === true 的条目，两道都保守。）
  if (cacheFile) {
    try {
      const cache = await loadCache(cacheFile)
      cache[key] = record
      await saveCache(cacheFile, cache)
    } catch (err) {
      runtime?.__log?.warn?.('源模型预检缓存写入失败（不影响结论）', { error: String(err.message) })
    }
  }

  if (result.unsupported) {
    runtime?.warnings?.push?.(warn(WARNING_CODES.W_SOURCE_VALIDATION_UNSUPPORTED,
      `源模型只有 .bin（PyTorch pickle），跳过结构预检；可用性改由**加载期实测**兜底（parity 的权重健康度断言）`,
      { dir, kind: result.unsupported }))
  }

  runtime?.__log?.info?.('源模型结构预检通过', {
    dir, tensors: record.evidence.totalTensors, zeroTensors: record.evidence.totalZeroTensors,
    mb: Number((record.evidence.totalDataBytes / 2 ** 20).toFixed(1)),
  })
  return record
}

export { warn, WARNING_CODES }
