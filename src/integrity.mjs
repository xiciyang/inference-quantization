/**
 * 源模型完整性（题目追加原则："原始模型只读：输出到独立目录，哈希校验原模型未改"）。
 *
 * 哈希口径刻意**不含 mtime / 权限 / 路径**：只对"内容"负责。
 * 若把 mtime 算进去，一次 `touch` 就会误报"模型被改了"；
 * 而真正要防的是**内容被改写**（量化脚本误写回源目录、权重被覆盖）。
 *
 * 算法：对目录内**排序后的相对路径 + 每个文件内容的 sha256** 再求一次 sha256。
 * 这样既能检出内容改动，也能检出文件增删与重命名。
 */
import { promises as fs } from 'node:fs'
import { join, relative, sep } from 'node:path'
import { createReadStream } from 'node:fs'
import { createHash } from 'node:crypto'
import { sha256 } from './util/hash.mjs'
import { fail, ERROR_CODES } from './errors.mjs'

const SKIP = new Set(['.git', '__pycache__', 'node_modules', '.cache'])

/** 递归收集文件（排序，跳过缓存类目录）。 */
async function collect(dir, base = dir, acc = []) {
  let entries
  try {
    entries = await fs.readdir(dir, { withFileTypes: true })
  } catch (err) {
    fail(ERROR_CODES.E_SOURCE_MODEL_UNREADABLE, `无法读取目录：${dir}`, {
      remediation: '确认路径存在且当前用户可读。', details: { dir, error: String(err.message) }, cause: err,
    })
  }
  for (const e of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (SKIP.has(e.name)) continue
    const full = join(dir, e.name)
    if (e.isDirectory()) await collect(full, base, acc)
    else if (e.isFile()) acc.push(full)
    // 软链：跟随一层（模型目录里常用软链指向快照）
    else if (e.isSymbolicLink()) {
      try {
        const st = await fs.stat(full)
        if (st.isDirectory()) await collect(full, base, acc)
        else if (st.isFile()) acc.push(full)
      } catch { /* 断链跳过 */ }
    }
  }
  return acc
}

/** 流式算单文件 sha256（不受 2 GiB buffer 上限影响）。 */
export function hashFileStreaming(path) {
  return new Promise((resolvePromise, reject) => {
    const h = createHash('sha256')
    let size = 0
    const stream = createReadStream(path, { highWaterMark: 4 * 1024 * 1024 })
    stream.on('data', (chunk) => { size += chunk.length; h.update(chunk) })
    stream.on('error', reject)
    stream.on('end', () => resolvePromise({ size, hash: h.digest('hex') }))
  })
}

/**
 * 计算目录内容哈希。
 * @returns {{hash, files, bytes, fileCount}}
 */
export async function hashDirectory(dir, { log = null } = {}) {
  const files = await collect(dir)
  const outer = []
  let bytes = 0
  for (const f of files.sort()) {
    // **必须流式**：fs.readFile 在单文件 > 2 GiB 时会直接抛
    // `ERR_FS_FILE_TOO_LARGE`（Node 的 buffer 上限），而 1.5B 模型的
    // 单个 safetensors 就有 3.09 GB —— 真机跑出来的坑。
    // eslint-disable-next-line no-await-in-loop
    const { size, hash } = await hashFileStreaming(f)
    bytes += size
    const rel = relative(dir, f).split(sep).join('/')
    outer.push(`${rel}\u0000${hash}`)
    if (log && outer.length % 200 === 0) log.info(`哈希进度 ${outer.length}/${files.length}`, { dir })
  }
  return { hash: sha256(outer.join('\n')), files: files.length, fileCount: files.length, bytes }
}

/** 量化前后各算一次并比对。 */
export async function assertSourceUnchanged(dir, before, { log = null } = {}) {
  const after = await hashDirectory(dir, { log })
  const unchanged = after.hash === before.hash
  if (!unchanged) {
    fail(ERROR_CODES.E_SOURCE_MODEL_MODIFIED, `源模型在量化过程中被改动：${dir}`, {
      remediation: '这是硬失败（题目要求原始模型只读）。请检查量化容器是否误写了源目录（应只读挂载），'
        + '以及输出目录是否被误设为源目录内部。本插件不会在源模型被改动的情况下签收任何结果。',
      details: { dir, beforeHash: before.hash, afterHash: after.hash, beforeBytes: before.bytes, afterBytes: after.bytes },
    })
  }
  log?.info('源模型哈希校验通过（内容未变）', { dir, hash: after.hash.slice(0, 16), files: after.files })
  return { unchanged, before, after }
}

/**
 * 输出目录安全检查：不得等于源目录，也不得位于源目录内部。
 * 这条是"原模型只读"的**第一道防线**（第二道是哈希校验）。
 */
export function assertOutputDirSafe(sourceDir, outputDir) {
  const s = sourceDir.endsWith(sep) ? sourceDir : sourceDir + sep
  const o = outputDir.endsWith(sep) ? outputDir : outputDir + sep
  if (o === s) {
    fail(ERROR_CODES.E_OUTPUT_DIR_UNSAFE, `输出目录不得等于源模型目录：${outputDir}`, {
      remediation: '给量化产出指定一个独立目录（默认在 ~/.ai-infra/quantized/ 下）。',
      details: { sourceDir, outputDir },
    })
  }
  if (o.startsWith(s)) {
    fail(ERROR_CODES.E_OUTPUT_DIR_UNSAFE, `输出目录位于源模型目录内部：${outputDir} ⊂ ${sourceDir}`, {
      remediation: '输出必须放到源目录之外，否则会污染源模型（且哈希校验会失败）。',
      details: { sourceDir, outputDir },
    })
  }
  if (s.startsWith(o)) {
    fail(ERROR_CODES.E_OUTPUT_DIR_UNSAFE, `输出目录是源模型目录的父目录：${outputDir} ⊃ ${sourceDir}`, {
      remediation: '输出目录不能是源目录的祖先，否则清理产出时会连带删掉源模型。',
      details: { sourceDir, outputDir },
    })
  }
  return true
}

export { collect }
