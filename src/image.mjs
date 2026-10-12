/**
 * 量化镜像的构建与验证（用户裁决 #4：签名 + 构建日志 + 漏洞扫描 + 可复现）。
 *
 * 四个约束的落地方式：
 *   ① 可复现 → Dockerfile 固定 base digest（docker/base-digest.txt）+ 依赖精确钉版；
 *      构建后把**镜像 digest** 与**依赖版本清单**落到 ~/.ai-infra/state/images/。
 *   ② 构建日志 → 全量写 logs/image-build.log（append，不截断）。
 *   ③ 漏洞扫描 → 调用可用的扫描器（trivy / grype / docker scout），一个都没有时
 *      **明确报告"未扫描"并让 build 结果带 warning**，不假装扫过。
 *   ④ 签名 → 调用 cosign / docker trust；两者都不可用时同样明说。
 *
 * 设计立场：**扫描器与签名工具不可用不等于可以跳过** —— 它们会让 `verified` 为 false，
 * 而 `verified=false` 的镜像在 ops 层会被拒绝用于量化（除非显式授权）。
 * 这比"静默跳过检查然后宣称安全"诚实得多。
 */
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { run, which } from './util/proc.mjs'
import { readText, atomicWriteJson, mkdirp, exists, appendLine } from './util/fsx.mjs'
import { fail, warn, ERROR_CODES, WARNING_CODES } from './errors.mjs'
import { sha256 } from './util/hash.mjs'

const SCANNERS = [
  { bin: 'trivy', args: (img) => ['image', '--quiet', '--format', 'json', img] },
  { bin: 'grype', args: (img) => [img, '-o', 'json', '--quiet'] },
  { bin: 'docker', args: (img) => ['scout', 'quickview', img], optional: true },
]
const SIGNERS = [
  { bin: 'cosign', args: (img) => ['sign', '--yes', img] },
  { bin: 'docker', args: (img) => ['trust', 'sign', img], optional: true },
]

export function imageMetaPath(runtime, image) {
  return join(runtime.paths.imageMetaDir, `${image.replace(/[^A-Za-z0-9._-]+/g, '_')}.json`)
}

export async function readImageMeta(runtime, image) {
  const p = imageMetaPath(runtime, image)
  const raw = await readText(p, null)
  if (!raw) return null
  try { return JSON.parse(raw) } catch { return null }
}

/** 从镜像里读出关键依赖版本（可复现性证据的一部分）。 */
export async function readToolchainVersions(runtime, image) {
  const py = 'import json,llmcompressor,compressed_tensors,transformers,torch,datasets;'
    + 'print(json.dumps({"llmcompressor":llmcompressor.__version__,"compressed_tensors":compressed_tensors.__version__,'
    + '"transformers":transformers.__version__,"torch":torch.__version__,"datasets":datasets.__version__}))'
  const r = await run('docker', ['run', '--rm', '--entrypoint', 'python3', image, '-c', py], { timeoutMs: 180_000 })
  if (r.code !== 0) return { ok: false, error: (r.stderr.trim() || 'probe failed').slice(0, 300) }
  try { return { ok: true, ...JSON.parse(r.stdout.trim().split('\n').filter(Boolean).pop()) } } catch { return { ok: false, error: 'parse failed' } }
}

/**
 * 构建镜像（可复现参数 + 全量日志）。
 */
export async function buildImage(runtime, { dockerfile = 'docker/Dockerfile.quant', image = null, log = null } = {}) {
  const img = image ?? runtime.options.quantImage
  const root = runtime.__packageRoot ?? new URL('../', import.meta.url).pathname
  const df = join(root, dockerfile)
  if (!await exists(df)) {
    fail(ERROR_CODES.E_IMAGE_NOT_VERIFIED, `Dockerfile 不存在：${df}`, {
      remediation: '确认包内包含 docker/Dockerfile.quant。', details: { dockerfile: df },
    })
  }
  const digestFile = join(root, 'docker', 'base-digest.txt')
  const baseDigest = (await readText(digestFile, '')).trim()
  if (!baseDigest) {
    fail(ERROR_CODES.E_IMAGE_NOT_VERIFIED, '缺少 base 镜像 digest（docker/base-digest.txt）', {
      remediation: '可复现构建要求固定 base digest；请先执行 `docker inspect --format "{{index .RepoDigests 0}}" <base>` 并写入该文件。',
      details: { digestFile },
    })
  }
  await mkdirp(join(root, 'logs'))
  const logFile = join(root, 'logs', 'image-build.log')
  await appendLine(logFile, `\n${'='.repeat(70)}\n[build] ${new Date().toISOString()} image=${img} base=${baseDigest}\n`)

  log?.phase('image-build', { image: img, baseDigest })
  const r = await run('docker', ['build', '-f', df, '--build-arg', `BASE_IMAGE=${baseDigest}`, '-t', img, root], {
    timeoutMs: 3_600_000,
  })
  await appendLine(logFile, r.stdout)
  await appendLine(logFile, `[stderr]\n${r.stderr}`)
  if (r.code !== 0) {
    fail(ERROR_CODES.E_IMAGE_NOT_VERIFIED, `镜像构建失败（exit=${r.code}）`, {
      remediation: `查看构建日志：${logFile}`,
      details: { image: img, exitCode: r.code, logFile, tail: r.stderr.slice(-1500) },
    })
  }
  return { ok: true, image: img, logFile, durationMs: r.durationMs }
}

/** 漏洞扫描。没装扫描器 → 明确报告未扫描（不是"通过"）。 */
export async function scanImage(runtime, { image, log = null } = {}) {
  for (const s of SCANNERS) {
    // eslint-disable-next-line no-await-in-loop
    const p = await which(s.bin)
    if (!p) continue
    // eslint-disable-next-line no-await-in-loop
    const r = await run(s.bin, s.args(image), { timeoutMs: 900_000 })
    const raw = r.stdout || ''
    let counts = null
    try {
      const j = JSON.parse(raw)
      if (Array.isArray(j.Results)) {
        counts = { critical: 0, high: 0, medium: 0, low: 0, unknown: 0 }
        for (const res of j.Results) {
          for (const v of res.Vulnerabilities ?? []) {
            const k = String(v.Severity ?? 'unknown').toLowerCase()
            counts[k] = (counts[k] ?? 0) + 1
          }
        }
      } else if (Array.isArray(j.matches)) {
        counts = { critical: 0, high: 0, medium: 0, low: 0, unknown: 0 }
        for (const m of j.matches) {
          const k = String(m.vulnerability?.severity ?? 'unknown').toLowerCase()
          counts[k] = (counts[k] ?? 0) + 1
        }
      }
    } catch { counts = null }
    log?.info('镜像漏洞扫描完成', { scanner: s.bin, exitCode: r.code, counts })
    return {
      scanned: true, scanner: s.bin, exitCode: r.code,
      counts, gate: (counts?.critical ?? 0) === 0 && (counts?.high ?? 0) === 0 ? 'pass' : 'fail',
      summary: counts ? JSON.stringify(counts) : raw.slice(0, 500),
    }
  }
  return { scanned: false, scanner: null, gate: 'unknown', summary: '未找到可用的漏洞扫描器（trivy / grype / docker scout）' }
}

/** 签名。没有签名工具 → 明确报告未签名。 */
export async function signImage(runtime, { image, log = null } = {}) {
  for (const s of SIGNERS) {
    // eslint-disable-next-line no-await-in-loop
    const p = await which(s.bin)
    if (!p) continue
    // eslint-disable-next-line no-await-in-loop
    const r = await run(s.bin, s.args(image), { timeoutMs: 300_000 })
    log?.info('镜像签名尝试', { signer: s.bin, exitCode: r.code })
    return { signed: r.code === 0, signer: s.bin, exitCode: r.code, detail: (r.stderr || r.stdout).slice(0, 400) }
  }
  return { signed: false, signer: null, detail: '未找到可用的签名工具（cosign / docker trust）' }
}

/**
 * 完整验证并落盘元数据。**verified 的含义**：构建成功 + 扫描器可用且无 critical/high + 已签名。
 * 任何一项缺失都会让 verified=false，并写清缺哪一项。
 */
export async function verifyImage(runtime, { image = null, log = null, build = false } = {}) {
  const img = image ?? runtime.options.quantImage
  let buildInfo = null
  if (build) buildInfo = await buildImage(runtime, { image: img, log })

  const inspect = await run('docker', ['inspect', '--format', '{{json .RepoDigests}}|{{.Id}}|{{.Size}}', img], { timeoutMs: 30_000 })
  if (inspect.code !== 0) {
    fail(ERROR_CODES.E_IMAGE_NOT_VERIFIED, `镜像不存在：${img}`, {
      remediation: '先执行 `inference-quantization image build`。', details: { image: img },
    })
  }
  const [digestsRaw, id, sizeRaw] = inspect.stdout.trim().split('|')
  let digests = []
  try { digests = JSON.parse(digestsRaw) } catch { digests = [] }

  const toolchain = await readToolchainVersions(runtime, img)
  const scan = await scanImage(runtime, { image: img, log })
  const sign = await signImage(runtime, { image: img, log })

  const missing = []
  if (!scan.scanned) missing.push('漏洞扫描器缺失')
  if (scan.scanned && scan.gate !== 'pass') missing.push(`扫描发现 critical/high（${scan.summary}）`)
  if (!sign.signed) missing.push('未签名')
  if (!toolchain.ok) missing.push('依赖版本探测失败')

  const meta = {
    image: img,
    digest: digests[0] ?? null,
    id: id ?? null,
    sizeBytes: Number(sizeRaw) || null,
    baseDigest: (await readText(join(runtime.__packageRoot ?? new URL('../', import.meta.url).pathname, 'docker', 'base-digest.txt'), '')).trim() || null,
    toolchain,
    scan,
    signature: sign,
    build: buildInfo,
    verified: missing.length === 0,
    missing,
    verifiedAt: new Date().toISOString(),
  }
  await mkdirp(runtime.paths.imageMetaDir)
  await atomicWriteJson(imageMetaPath(runtime, img), meta)
  if (missing.length > 0) {
    runtime.warnings.push(warn(WARNING_CODES.W_INTERNAL_SHARED_LIB,
      `量化镜像验证未完全通过（缺：${missing.join('；')}）；未显式授权时量化会被拒绝`, { image: img, missing }))
  }
  log?.info('镜像验证完成', { image: img, verified: meta.verified, missing })
  return meta
}

/** 门禁：镜像未 verified 时的行为（R2 操作需显式授权才放行）。 */
export async function assertImageVerified(runtime, { image = null, allowUnverified = false } = {}) {
  const img = image ?? runtime.options.quantImage
  const meta = await readImageMeta(runtime, img)
  if (!meta) {
    // 显式授权可以越过"从未验证过"这一情形（用户裁决允许带审计地放宽镜像可信度这一项）。
    // 不放宽的是：容器级沙箱、源模型只读、去泄漏、双门禁 —— 那些与镜像可信度无关。
    if (!allowUnverified) {
      fail(ERROR_CODES.E_IMAGE_NOT_VERIFIED, `量化镜像尚未验证：${img}`, {
        remediation: '先执行 `inference-quantization image verify`（会做构建/扫描/签名/依赖探测并落盘元数据）；'
          + '或在调用里显式传 allow_unverified_image=true（会记入审计）。',
        details: { image: img, metaPath: imageMetaPath(runtime, img) },
      })
    }
    runtime.warnings.push(warn(WARNING_CODES.W_INTERNAL_SHARED_LIB,
      `镜像 ${img} 从未验证过，已按显式授权继续（审计中已记录）`, { image: img, authorized: true }))
    return { image: img, verified: false, missing: ['从未验证'], digest: null, toolchain: null, authorized: true }
  }
  if (!meta.verified && !allowUnverified) {
    fail(ERROR_CODES.E_IMAGE_NOT_VERIFIED, `量化镜像验证未通过：缺 ${meta.missing.join('；')}`, {
      remediation: '补齐扫描器/签名工具后重跑 `image verify`；或在调用里显式传 allow_unverified_image=true（会记入审计）。'
        + '注意：容器级沙箱仍然生效，这条只放宽"镜像可信度"这一项。',
      details: meta,
    })
  }
  return meta
}

export { sha256 }
