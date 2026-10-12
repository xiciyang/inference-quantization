/**
 * 双门禁判定（精度 + 性能）—— 题目核心闭环的第 6 步。
 *
 * 两条硬约束来自用户裁决：
 *   ① **精度阈值必须调用方传入**（MCP 的 `max_accuracy_drop_pct`）；
 *      未传 → 从 `~/.ai-infra/config/gates.yaml` 读；两者都无 → **拒绝执行，不猜默认值**。
 *      理由：题面要求"精度门禁独立：量化不能自己签收精度"。若插件内有一套默认标准，
 *      "独立"就名存实亡 —— 量化自己定阈值、自己判达标。
 *   ② **精度损失接近容忍上限要停下确认**（默认 ≥90% 阈值即视为接近）。
 *
 * 门禁是**纯函数**：给数值与阈值，出判定。不发请求、不读文件、不写状态 ——
 * 这样它可以被穷举测试，也保证"判定"与"测量"彻底分离。
 */
import { fail, warn, ERROR_CODES, WARNING_CODES } from './errors.mjs'

export const GATE_FIELDS = ['max_accuracy_drop_pct', 'min_perf_gain_pct']

/**
 * 解析阈值：显式参数 > gates.yaml > 拒绝。
 * @returns {{maxAccuracyDropPct, minPerfGainPct, source}}
 */
export function resolveThresholds(runtime, { maxAccuracyDropPct = null, minPerfGainPct = null } = {}) {
  const file = runtime?.policy?.gates ?? null
  const pick = (explicit, key) => {
    if (explicit !== null && explicit !== undefined && Number.isFinite(Number(explicit))) {
      return { value: Number(explicit), from: 'caller' }
    }
    const v = file?.[key]
    if (v !== undefined && v !== null && Number.isFinite(Number(v))) return { value: Number(v), from: 'gates.yaml' }
    return { value: null, from: null }
  }
  const acc = pick(maxAccuracyDropPct, 'max_accuracy_drop_pct')
  const perf = pick(minPerfGainPct, 'min_perf_gain_pct')

  const missing = []
  if (acc.value === null) missing.push('max_accuracy_drop_pct（精度容忍）')
  if (perf.value === null) missing.push('min_perf_gain_pct（性能收益目标）')
  if (missing.length > 0) {
    fail(ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, `缺少门禁阈值：${missing.join('、')}`, {
      remediation: '在调用里传 options.max_accuracy_drop_pct / min_perf_gain_pct，'
        + `或把它们写进 ${runtime?.paths?.gatesFile ?? '~/.ai-infra/config/gates.yaml'}。`
        + '本插件**不会**替你猜一个默认阈值 —— 那会让"精度门禁独立"失效。',
      details: { missing, gatesFile: runtime?.paths?.gatesFile ?? null, policySource: runtime?.policy?.sources?.gates ?? null },
    })
  }
  if (acc.value < 0) {
    fail(ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, `max_accuracy_drop_pct 不能为负：${acc.value}`, {
      remediation: '精度损失容忍应为非负数（0 表示不容忍任何下降）。', details: { value: acc.value },
    })
  }
  return {
    maxAccuracyDropPct: acc.value,
    minPerfGainPct: perf.value,
    source: { maxAccuracyDropPct: acc.from, minPerfGainPct: perf.from },
  }
}

/** 精度门禁。近上限判定用**相对阈值的比例**，避免不同量纲下的魔数。 */
export function evaluateAccuracyGate({ dropPct, thresholdPct, nearLimitFactor = 0.9 }) {
  if (dropPct === null || dropPct === undefined) return { passed: false, undetermined: true, reason: '精度损失未测得' }
  const passed = dropPct <= thresholdPct
  const ratio = thresholdPct > 0 ? dropPct / thresholdPct : (dropPct <= 0 ? 0 : Infinity)
  const nearLimit = passed && ratio >= nearLimitFactor
  return {
    passed,
    dropPct,
    thresholdPct,
    marginPct: thresholdPct - dropPct,
    ratioOfThreshold: Number.isFinite(ratio) ? Number(ratio.toFixed(4)) : null,
    nearLimit,
    nearLimitFactor,
    reason: passed
      ? (nearLimit ? `精度损失 ${dropPct.toFixed(3)}% 已达阈值的 ${(ratio * 100).toFixed(0)}%（接近上限）` : `精度损失 ${dropPct.toFixed(3)}% ≤ 阈值 ${thresholdPct}%`)
      : `精度损失 ${dropPct.toFixed(3)}% > 阈值 ${thresholdPct}%`,
  }
}

/** 性能门禁。收益为负（变慢）同样记为不通过。 */
export function evaluatePerfGate({ gainPct, thresholdPct }) {
  if (gainPct === null || gainPct === undefined) return { passed: false, undetermined: true, reason: '性能收益未测得' }
  const passed = gainPct >= thresholdPct
  return {
    passed,
    gainPct,
    thresholdPct,
    marginPct: gainPct - thresholdPct,
    reason: passed
      ? `性能收益 ${gainPct.toFixed(2)}% ≥ 目标 ${thresholdPct}%`
      : `性能收益 ${gainPct.toFixed(2)}% < 目标 ${thresholdPct}%`,
  }
}

/**
 * **测量成立性**判定 —— 与"数值是否达标"是两件独立的事。
 *
 * 用户裁决（2026-09-24）：门禁不仅要判"数值是否达标"，还要判"测量是否成立"。
 * 真机事故证明了这个要求的必要性：一份下载中断的源模型跑出
 * meanNll = ln(vocab_size)、perplexity = 151935.97（= 词表大小），
 * 基线与候选**逐位相同** → dropPct = 0 → 旧门禁判"精度零损失，签收"。
 * 数值没错，测量没有意义。
 *
 * 五条互相独立的否证（任一条成立即判"测量不成立"）：
 *   ① 任一侧没评到分（scoredSamples = 0）
 *   ② 任一侧的 sidecar 自报 validity.valid = false（parity 的权重/输出健康度断言）
 *   ③ **均匀输出签名**：|meanNll − ln(vocab_size)| ≤ toleranceNats
 *   ④ **无区分力**：基线与候选的 meanNll 逐位相同（权重不同却给出同一个数，
 *      说明这个数不依赖权重）
 *   ⑤ **与输入无关**：逐条 meanNll 的标准差为 0
 *
 * 纯函数：不发请求、不读文件、不写状态。
 *
 * @returns {{valid:boolean, reasons:object[], checks:object}}
 */
export function assertMeasurementValid({ baseline, candidate, toleranceNats = 1e-4 } = {}) {
  const reasons = []
  const checks = { toleranceNats }

  const side = (name, r) => {
    if (!r) {
      reasons.push({ code: 'missing-side', message: `${name} 侧没有测量结果` })
      return
    }
    const scored = r.scoredSamples ?? null
    if (scored !== null && scored <= 0) {
      reasons.push({ code: 'no-scored-samples', message: `${name} 侧没有任何样本评分成功` })
    }
    if (r.validity && r.validity.valid === false) {
      for (const x of r.validity.reasons ?? []) {
        reasons.push({ code: `sidecar:${x.code ?? 'invalid'}`, message: `${name} 侧自报测量不成立：${x.detail ?? x.code ?? ''}` })
      }
    }
    const vocab = r.vocabSize ?? null
    const meanNll = Number(r.meanNll)
    if (vocab && Number.isFinite(meanNll)) {
      const uniformLoss = Math.log(vocab)
      const gap = Math.abs(meanNll - uniformLoss)
      checks[`${name}UniformGap`] = gap
      checks[`${name}VocabSize`] = vocab
      checks[`${name}UniformLoss`] = uniformLoss
      if (gap <= toleranceNats) {
        reasons.push({
          code: 'uniform-output',
          message: `${name} 侧 meanNll=${meanNll} 与 ln(vocab_size)=${uniformLoss} 相差 ${gap.toExponential(2)} ≤ ${toleranceNats}`
            + ' —— 模型输出近似均匀分布，此时"精度损失"没有区分力',
        })
      }
    } else {
      checks[`${name}UniformGap`] = null
    }
    // 逐条方差：用**极差**而不是标准差。
    // 理由：标准差要先把 N 个数求和再相除，浮点误差会让"全都相同"算出 ~1e-15 而不是 0，
    // 于是必须引入一个容差魔数。极差（max − min）是精确比较，不需要任何阈值：
    // 所有样本逐位相同 ⇒ 极差恰好为 0。
    const items = r.itemMeanNlls ?? null
    if (Array.isArray(items) && items.length > 1) {
      let min = Infinity
      let max = -Infinity
      for (const x of items) {
        if (typeof x !== 'number' || !Number.isFinite(x)) continue
        if (x < min) min = x
        if (x > max) max = x
      }
      const range = max - min
      checks[`${name}ItemMeanNllRange`] = Number.isFinite(range) ? range : null
      checks[`${name}ItemMeanNllMin`] = Number.isFinite(min) ? min : null
      checks[`${name}ItemMeanNllMax`] = Number.isFinite(max) ? max : null
      if (range === 0) {
        reasons.push({
          code: 'no-item-variance',
          message: `${name} 侧 ${items.length} 条样本的逐条 meanNll 逐位相同（极差 = 0）`
            + ' —— 该数与输入无关，不构成一次测量',
        })
      }
    }
  }

  side('baseline', baseline)
  side('candidate', candidate)

  if (baseline && candidate) {
    const b = Number(baseline.meanNll)
    const c = Number(candidate.meanNll)
    const identical = Number.isFinite(b) && Number.isFinite(c) && b === c
    checks.bitIdenticalMeanNll = identical
    if (identical) {
      reasons.push({
        code: 'indiscriminate',
        message: `基线与候选的 meanNll 逐位相同（${b}）：两个不同的权重集合给出同一个数，`
          + '说明该测量不依赖权重 —— 退化的"精度损失 0%"不能作为无损证据',
      })
    }
  }

  return { valid: reasons.length === 0, reasons, checks }
}

/**
 * 双门禁合并判定。
 * @param {{accuracy, perf, thresholds, measurement?}} p
 *   `measurement` = assertMeasurementValid 的结果。传入且 valid=false 时，
 *   判定直接落到 `undetermined`、**不可签收** —— 数值再好也不签。
 *   刻意不传（undefined）时保持旧行为，便于既有单测与"只判数值"的纯函数测试。
 * @returns {{passed, decision:'sign'|'confirm-near-limit'|'degrade'|'undetermined', accuracy, perf, reasons}}
 */
export function evaluateGates({ accuracy, perf, thresholds, measurement = null }) {
  const acc = evaluateAccuracyGate({ dropPct: accuracy?.dropPct ?? null, thresholdPct: thresholds.maxAccuracyDropPct })
  const pf = evaluatePerfGate({ gainPct: perf?.gainPct ?? null, thresholdPct: thresholds.minPerfGainPct })

  const reasons = []
  if (measurement && measurement.valid === false) {
    for (const r of measurement.reasons ?? []) reasons.push(`测量：${r.message}`)
  }
  if (acc.undetermined) reasons.push(`精度：${acc.reason}`)
  else if (!acc.passed) reasons.push(`精度：${acc.reason}`)
  if (pf.undetermined) reasons.push(`性能：${pf.reason}`)
  else if (!pf.passed) reasons.push(`性能：${pf.reason}`)

  const measurementInvalid = Boolean(measurement && measurement.valid === false)

  let decision
  if (measurementInvalid || acc.undetermined || pf.undetermined) decision = 'undetermined'
  else if (acc.passed && pf.passed) decision = acc.nearLimit ? 'confirm-near-limit' : 'sign'
  else decision = 'degrade'

  return {
    passed: !measurementInvalid && acc.passed === true && pf.passed === true,
    decision,
    signable: decision === 'sign',
    needsConfirmation: decision === 'confirm-near-limit',
    measurementValid: measurement ? measurement.valid !== false : null,
    measurement: measurement ?? null,
    accuracy: acc,
    perf: pf,
    thresholds,
    reasons: reasons.length ? reasons : [acc.reason, pf.reason],
  }
}

/** 接近上限时的确认令牌（与 serving 的令牌语义一致：绑定具体目标）。 */
export function nearLimitTokenFor({ sourceModel, method, dropPct }) {
  return `nlq_${Buffer.from(`${sourceModel}|${method}|${Number(dropPct).toFixed(3)}`).toString('hex').slice(0, 20)}`
}

export { warn, WARNING_CODES }
