/**
 * 去泄漏门禁 —— **本插件最核心的安全属性**（用户裁决：内容哈希交集为空）。
 *
 * 为什么不是"索引不重叠"：
 *   同一段文本在数据集的 train / test 两个 split 里同时出现是很常见的
 *   （wikitext、C4 等都有）。按索引判断不会发现这种泄漏，而这恰恰是最脏的一种：
 *   校准时"见过"的样本又出现在测试集里，精度数字会偏乐观。
 *
 * 为什么按**内容**：
 *   内容哈希是**可证伪**的 —— 交集为空可以逐条复核。索引、行号、随机种子都做不到这点。
 *
 * 本模块是**纯函数**，容器的产出必须由宿主用同一套函数**独立复核**一遍
 * （不信任容器自己的自检结论）。
 */
import { sha256 } from '../util/hash.mjs'
import { fail, warn, ERROR_CODES, WARNING_CODES } from '../errors.mjs'

/** 单条样本的内容哈希。归一化必须与容器侧一致（见 docker/scripts/prepare-dataset.py）。 */
export function itemHash(text) {
  return sha256(String(text ?? '').trim())
}

export function hashAll(items) {
  return items.map((t) => itemHash(t))
}

/** 有序哈希清单的摘要（与容器侧 list_digest 同算法：按换行连接后 sha256）。 */
export function listDigest(hashes) {
  return sha256(hashes.join('\n'))
}

/**
 * 校验两段哈希清单无交集，且并集等于去重全集。
 * @returns {{clean:boolean, intersection:string[], unionCount:number, reason:string|null}}
 */
export function checkDisjoint(calibHashes, testHashes, { dedupedTotal = null } = {}) {
  const c = new Set(calibHashes)
  const t = new Set(testHashes)
  const intersection = [...c].filter((h) => t.has(h))
  const union = new Set([...c, ...t])
  const duplicateWithin = calibHashes.length - c.size + (testHashes.length - t.size)

  let reason = null
  if (intersection.length > 0) reason = `校准集与测试集有 ${intersection.length} 条内容重复（去泄漏判据：内容 sha256 交集必须为空）`
  else if (duplicateWithin > 0) reason = `同一段内部存在 ${duplicateWithin} 条重复`
  // 注意：**不能**要求"两段并集 == 去重全集"。
  // 两段是从去重后的语料里**抽样**出来的子集（例如 128 + 400 ≪ 14324），
  // 要求它们覆盖全集在逻辑上就是错的 —— 真机跑出来才发现的判据误用。
  // 正确的不变量是：并集规模不得超过全集（下界合理性），且两段互不相交（上面已判）。
  else if (dedupedTotal !== null && union.size > dedupedTotal) {
    reason = `两段并集 ${union.size} 条 > 去重全集 ${dedupedTotal} 条（不可能，说明计数或去重口径有误）`
  } else if (dedupedTotal !== null && (calibHashes.length > dedupedTotal || testHashes.length > dedupedTotal)) {
    reason = `某一段的条数超过去重全集（校准 ${calibHashes.length} / 测试 ${testHashes.length} vs 全集 ${dedupedTotal}）`
  }

  return { clean: reason === null, intersection, unionCount: union.size, reason }
}

/**
 * 宿主侧独立复核：给定两份 jsonl 原文 + 容器产出的 lineage，
 * **重新计算**哈希并逐条比对，而不是相信 lineage 里写的数字。
 */
export function verifyLineage(lineage, calibItems, testItems) {
  const problems = []
  const calibHashes = hashAll(calibItems)
  const testHashes = hashAll(testItems)

  const disjoint = checkDisjoint(calibHashes, testHashes, { dedupedTotal: lineage?.dedup?.unique ?? null })
  if (!disjoint.clean) problems.push(disjoint.reason)

  // 逐条比对：容器写下的每条哈希，必须与宿主重算的一致
  const recordedCalib = lineage?.split?.calibration?.hashes ?? []
  const recordedTest = lineage?.split?.test?.hashes ?? []
  if (recordedCalib.length !== calibHashes.length) {
    problems.push(`lineage 记录的校准条目数 ${recordedCalib.length} ≠ 实际 ${calibHashes.length}`)
  }
  if (recordedTest.length !== testHashes.length) {
    problems.push(`lineage 记录的测试条目数 ${recordedTest.length} ≠ 实际 ${testHashes.length}`)
  }
  const sortEq = (a, b) => a.length === b.length && [...a].sort().join() === [...b].sort().join()
  if (!sortEq(recordedCalib, calibHashes)) problems.push('校准段哈希清单与宿主重算结果不一致（内容被改动或计数有误）')
  if (!sortEq(recordedTest, testHashes)) problems.push('测试段哈希清单与宿主重算结果不一致')
  if (lineage?.split?.calibration?.listDigest && lineage.split.calibration.listDigest !== listDigest(calibHashes)) {
    problems.push('校准段 listDigest 与宿主重算不一致')
  }

  return {
    verified: problems.length === 0,
    problems,
    counts: {
      calibration: calibHashes.length,
      test: testHashes.length,
      unique: new Set([...calibHashes, ...testHashes]).size,
    },
    calibListDigest: listDigest(calibHashes),
    testListDigest: listDigest(testHashes),
    criterion: 'content-sha256-intersection-empty',
  }
}

/** 门禁：不干净就中止。这是硬门禁，没有"确认后继续"这条路径（用户裁决：R6 语义）。 */
export function assertNoLeakage(verification) {
  if (verification.verified) return verification
  fail(ERROR_CODES.E_CALIB_LEAKAGE_DETECTED, `去泄漏门禁未通过：${verification.problems.join('；')}`, {
    remediation: '校准集与测试集必须内容不相交。请更换切分种子、扩大去重范围，或改用另一个数据集；'
      + '本门禁不接受人工确认放行（泄漏会让精度数字失去意义）。',
    details: verification,
  })
}

export { warn, WARNING_CODES }
