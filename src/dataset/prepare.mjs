/**
 * 校准/测试集获取与切分编排。
 *
 * 分工刻意做成两层：
 *   - **容器内**跑 `prepare-dataset.py`（宿主没有 datasets，也不该装）；
 *   - **宿主侧**用 `leakage.mjs` 的同一套纯函数**独立复核**容器的产出。
 *
 * 也就是说：容器说"我没泄漏"不算数，宿主必须自己算出同样的结论。
 * 这是"去泄漏门禁可复核"的落点。
 */
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { run } from '../util/proc.mjs'
import { readText, mkdirp, atomicWriteJson, exists } from '../util/fsx.mjs'
import { fail, warn, ERROR_CODES, WARNING_CODES } from '../errors.mjs'
import { verifyLineage, assertNoLeakage, listDigest, hashAll } from './leakage.mjs'
import { LINEAGE_FORMAT_VERSION } from '../version.mjs'

/** 读 jsonl（每行 {"h","t"}）。 */
export async function readJsonl(path) {
  const raw = await readText(path, '')
  const out = []
  for (const line of raw.split('\n')) {
    const t = line.trim()
    if (!t) continue
    try {
      const o = JSON.parse(t)
      if (typeof o.t === 'string') out.push(o.t)
    } catch { /* 跳过损坏行，由后续计数比对发现 */ }
  }
  return out
}

/**
 * 准备数据集：容器内下载切分 → 宿主独立复核。
 * @returns {{ok, dir, lineage, verification, reused}}
 */
export async function prepareDataset(runtime, {
  repo = null, config = null, split = null, seed = null, calibrationSamples = null, testSamples = 400,
  log = null, force = false,
} = {}) {
  const opts = runtime.options
  const repoId = repo ?? opts.datasetRepo
  const cfgName = config ?? opts.datasetConfig
  const splitName = split ?? opts.datasetSplit
  const useSeed = seed ?? opts.calibrationSeed
  const calibN = calibrationSamples ?? opts.calibrationSamples

  // 目录名把全部切分参数编码进去 —— 换参数就换目录，绝不复用旧数据（复用=隐性泄漏源）
  const tag = `${repoId.replace(/[^A-Za-z0-9]+/g, '-')}__${cfgName ?? 'default'}__${splitName}__s${useSeed}__c${calibN}`
  const dir = join(runtime.paths.datasetDir, tag)
  const lineagePath = join(dir, 'lineage.json')

  if (!force && await exists(lineagePath)) {
    log?.info('数据集已存在，复核后复用', { dir })
    const reuse = await verifyDatasetDir(runtime, dir, { log })
    if (reuse.verification.verified) return { ok: true, dir, reused: true, ...reuse }
    log?.warn('已存在的数据集复核未通过，将重新生成', { dir, problems: reuse.verification.problems })
  }

  await mkdirp(dir)
  log?.phase('dataset-fetch', { repo: repoId, config: cfgName, split: splitName, seed: useSeed, calibrationSamples: calibN })

  const { executorFor } = await import('../executors/index.mjs')
  await import('../executors/docker.mjs')
  const executor = executorFor(runtime)
  // **有界重试**：本机到 hf-mirror 的连接实测会抖（`handshake operation timed out`）。
  // 这是可恢复的瞬时故障，不该让整次量化作废。最多 3 次，每次前清掉半成品目录，
  // 且因为挂了持久化 HF 缓存，重试通常不必重下。
  const attempts = Number.isInteger(opts.datasetRetries) ? opts.datasetRetries : 3
  let r = null
  let stdoutTail = ''
  let stderrTail = ''
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    // eslint-disable-next-line no-await-in-loop
    r = await executor.prepareDataset(runtime, {
      repo: repoId, config: cfgName, split: splitName, outDir: dir,
      seed: useSeed, calib: calibN, test: testSamples, log,
    })
    stdoutTail = (r.stdout ?? '').slice(-3000)
    stderrTail = (r.stderr ?? '').slice(-1500)
    log?.info('数据集准备结束', { executor: executor.id, attempt, attempts, exitCode: r.exitCode, ok: r.ok })
    if (r.ok) break
    const transient = /timeout|timed out|Connection|ECONN|handshake|Temporary failure/i.test(`${stdoutTail}\n${stderrTail}`)
    if (!transient || attempt === attempts) break
    log?.warn(`数据集下载疑似瞬时失败（第 ${attempt}/${attempts} 次），清掉半成品后重试`, { tail: stdoutTail.slice(-200) })
    // 清掉半成品，避免"目录存在但内容不全"被后续误判为可复用
    const { promises: fsp } = await import('node:fs')
    await fsp.rm(join(dir, 'calib.jsonl'), { force: true }).catch(() => {})
    await fsp.rm(join(dir, 'test.jsonl'), { force: true }).catch(() => {})
    await fsp.rm(join(dir, 'lineage.json'), { force: true }).catch(() => {})
    // eslint-disable-next-line no-await-in-loop
    await new Promise((res) => { setTimeout(res, 3000 * attempt) })
  }

  if (!r?.ok) {
    fail(ERROR_CODES.E_DATASET_UNAVAILABLE, `数据集准备失败（exit=${r?.exitCode}，已尝试 ${attempts} 次）`, {
      remediation: '检查 hf-mirror.com 可达性（本机实测会抖，可重跑）与 --repo/--config 是否正确；容器内输出见 details。',
      details: { repo: repoId, config: cfgName, split: splitName, exitCode: r?.exitCode, stdoutTail, stderrTail },
    })
  }

  const verify = await verifyDatasetDir(runtime, dir, { log })
  assertNoLeakage(verify.verification)

  // 生成宿主侧的血缘副本（含复核结论）—— 容器写的 lineage 保留原样，便于对照
  const hostLineage = {
    ...verify.lineage,
    formatVersion: LINEAGE_FORMAT_VERSION,
    hostVerification: {
      verified: verify.verification.verified,
      problems: verify.verification.problems,
      counts: verify.verification.counts,
      calibListDigest: verify.verification.calibListDigest,
      testListDigest: verify.verification.testListDigest,
      criterion: verify.verification.criterion,
      verifiedAt: new Date().toISOString(),
    },
  }
  await atomicWriteJson(join(runtime.paths.lineageDir, `${tag}.json`), hostLineage)
  return { ok: true, dir, reused: false, tag, ...verify, hostLineage }
}

/** 复核一个数据集目录（宿主独立计算）。 */
export async function verifyDatasetDir(runtime, dir, { log = null } = {}) {
  const calibPath = join(dir, 'calib.jsonl')
  const testPath = join(dir, 'test.jsonl')
  const lineagePath = join(dir, 'lineage.json')
  for (const [p, what] of [[calibPath, '校准集'], [testPath, '测试集'], [lineagePath, '血缘']]) {
    if (!await exists(p)) {
      fail(ERROR_CODES.E_DATASET_UNAVAILABLE, `数据集目录缺文件：${what}（${p}）`, {
        remediation: '数据集准备未完成或已被破坏；用 force=true 重新生成。',
        details: { dir, missing: p },
      })
    }
  }
  const [calibItems, testItems, lineageRaw] = await Promise.all([
    readJsonl(calibPath), readJsonl(testPath), readText(lineagePath, '{}'),
  ])
  let lineage = {}
  try { lineage = JSON.parse(lineageRaw) } catch { lineage = {} }
  const verification = verifyLineage(lineage, calibItems, testItems)
  log?.info('去泄漏复核（宿主独立计算）', {
    verified: verification.verified,
    counts: verification.counts,
    problems: verification.problems,
  })
  return { dir, lineage, verification, calibPath, testPath, lineagePath, calibrationSamples: calibItems.length, testSamples: testItems.length }
}

/** 便捷：给定数据集目录，返回校准/测试内容。 */
export async function loadDatasetSplit(dir) {
  const [calib, test] = await Promise.all([readJsonl(join(dir, 'calib.jsonl')), readJsonl(join(dir, 'test.jsonl'))])
  return { calib, test, calibHashes: hashAll(calib), testHashes: hashAll(test), calibListDigest: listDigest(hashAll(calib)) }
}

export { warn }
