/**
 * QAT 计划（**只产出计划，不执行训练**）。
 *
 * 依据用户裁决 #1：
 *   "QAT 是训练（写新权重、小时级），落 R3 → 需 VM 沙箱 → 本机没有 → 按规则必然拒绝。
 *    补充：QAT 计划必须包含数据、配置、预期收益、回退路径，让用户能直接决策。"
 *
 * 所以本模块做的事是：把"要跑 QAT 需要什么"**具体化到可执行的程度**，
 * 然后停下等裁决 —— 而不是含糊地说一句"建议做 QAT"。
 *
 * 若用户显式授权降级到容器级，`authorizeQat()` 会产出一条**每次确认、不缓存**的授权记录，
 * 但仍然只产出可执行的命令，不在这里替用户按下开始键。
 */
import { warn, WARNING_CODES, fail, ERROR_CODES } from './errors.mjs'
import { shortId } from './util/hash.mjs'

/** QAT 授权令牌：与目标绑定，且**不落盘缓存**（每次都要重新确认）。 */
export function qatApprovalToken({ sourceModel, method, planId }) {
  return `qat_${shortId(`${sourceModel}|${method}|${planId}`, 20)}`
}

export const QAT_APPROVAL_TOKEN = 'qat-approve'

/**
 * 构建 QAT 计划。
 * @returns {object} 可直接给人做决策的计划（含数据/配置/预期收益/回退路径/风险）
 */
export async function buildQatPlan(runtime, {
  standard, env, attempts = [], thresholds, lastParity = null, lastPerf = null, lastFailure = null, log = null,
} = {}) {
  const planId = `qat_${shortId(`${standard.source.path}|${Date.now()}`, 10)}`
  const bestDrop = attempts
    .map((a) => a.parity?.dropPct ?? a.parityCoarse?.dropPct)
    .filter((x) => typeof x === 'number')
    .sort((a, b) => a - b)[0] ?? null
  const gapPct = bestDrop !== null ? Number((bestDrop - thresholds.maxAccuracyDropPct).toFixed(4)) : null

  const plan = {
    schema: 'inference-quantization/qat-plan@1',
    planId,
    generatedAt: new Date().toISOString(),
    status: 'awaiting-approval',
    executes: false,
    why: `所有可执行 PTQ 档位均未同时通过双门禁。最好的一档精度损失 ${bestDrop ?? 'n/a'}%，`
      + `超出容忍 ${thresholds.maxAccuracyDropPct}%${gapPct !== null ? `（差 ${gapPct} 个百分点）` : ''}。`,

    // --- 数据 ---
    data: {
      calibrationSet: {
        path: standard.dataset ? `${runtime.paths.datasetDir}` : null,
        repo: standard.dataset?.repo, config: standard.dataset?.config, split: standard.dataset?.split,
        samples: standard.calibrationSamples, seed: standard.seed,
        leakageGate: 'content-sha256-intersection-empty（QAT 同样受此门禁约束）',
      },
      testSet: { repo: standard.dataset?.repo, config: standard.dataset?.config, note: '与校准集内容不相交（已由去泄漏门禁保证）' },
      note: 'QAT 需要比 PTQ 更大且更接近目标分布的数据；当前校准集规模对 QAT 偏小，建议扩容到 5k–50k 条并重新过一遍去泄漏门禁。',
    },

    // --- 配置 ---
    config: {
      baseModel: standard.source.path,
      startFrom: attempts.at(-1)?.method ?? standard.method,
      epochsSuggested: [1, 3],
      learningRateSuggested: [1e-5, 5e-6],
      batchSizeSuggested: [1, 2],
      sequenceLength: standard.maxSeqLen ?? 2048,
      quantizationScheme: 'W4A16（与 PTQ 最优档一致，便于对比）',
      frameworkHint: 'llmcompressor 的 QAT 路径（本机镜像已含其依赖），或 torchao/peft 的手写 QAT',
      image: runtime.options.quantImage,
      gpu: env?.gpu?.gpus?.[0]?.name ?? null,
      availableMemoryBytes: env?.capability?.availableMemoryBytes ?? null,
    },

    // --- 预期收益（必须诚实：这是估计，不是承诺） ---
    expectedGain: {
      basis: 'QAT 通常能把同 bit 宽的精度损失压到 PTQ 的 1/3 ~ 1/2（文献经验值，非本机实测）',
      bestPtqDropPct: bestDrop,
      thresholdPct: thresholds.maxAccuracyDropPct,
      gapToClosePct: gapPct,
      caveat: '本数值是经验估计，**本机从未跑过 QAT**；实际能否达标须以跑完之后的完整评测为准。',
    },

    // --- 风险 ---
    risks: [
      { id: 'sandbox', severity: 'high', description: 'QAT 属训练，按已确认的风险分级落 R3（需 VM 级沙箱）；本机无 VM 沙箱 → 默认拒绝。要跑必须显式授权降级到容器级（每次确认，不缓存）。' },
      { id: 'duration', severity: 'medium', description: '小时级到天级，期间占用 GPU；与既有 qwen3-embedding 服务共存需留意内存。' },
      { id: 'no-guarantee', severity: 'medium', description: 'QAT 不保证一定达标；可能需要多轮调参。' },
      { id: 'weights', severity: 'low', description: 'QAT 会写出新权重，但输出到独立目录、源模型只读挂载并做哈希校验，不会污染源模型。' },
    ],

    // --- 回退路径 ---
    rollbackPath: {
      steps: [
        '① 不签收：量化矩阵里本 run 保持 failed，不产生 verified 记录',
        '② 删除产出目录（只删本插件登记的路径，源模型不受影响）',
        `③ 源模型哈希校验：量化中已记录 before/after，回退不涉及源模型写入`,
        '④ 若已部署过候选端点：通过 MCP 调 inference_serving_teardown 拆掉',
      ],
      command: `inference-quantization rollback ${planId}`,
      note: '任何时刻都可以选择"回退 = 不做量化"，成本只是删掉产出目录。',
    },

    // --- 授权链 ---
    approval: {
      required: true,
      token: qatApprovalToken({ sourceModel: standard.source.path, method: standard.method, planId }),
      cacheable: false,
      note: '授权**每次都要重新确认**（用户裁决 #6）；授权只放宽沙箱层级这一项，源模型只读/独立目录/资源上限/可回退不变。',
    },

    // --- 全部尝试记录（让人能自己判断） ---
    attemptSummary: attempts.map((a) => ({
      level: a.level, method: a.method, failureCode: a.failureCode,
      accuracyDropPct: a.parity?.dropPct ?? a.parityCoarse?.dropPct ?? null,
      perfGainPct: a.perf?.gainDecodePct ?? null,
    })),
    lastFailure,
    lastParity,
    lastPerf,
  }

  runtime.warnings.push(warn(WARNING_CODES.W_QAT_PLAN_ONLY,
    `已产出 QAT 计划（${planId}）但**不会自动执行**：QAT 属训练（R3，需 VM 沙箱，本机不可用）`,
    { planId, approvalToken: plan.approval.token }))

  log?.phase('qat-plan', { planId, required: true })
  return plan
}

/**
 * 显式授权 QAT 时调用：产出**可执行但不自动执行**的命令与约束清单。
 * 每次调用都必须携带与计划绑定的令牌；令牌不落盘、不进缓存。
 */
export function authorizeQat(runtime, { plan, confirmToken = null, allowSandboxDegrade = false }) {
  const expected = plan?.approval?.token
  if (!expected || confirmToken !== expected) {
    fail(ERROR_CODES.E_QAT_REQUIRES_APPROVAL, 'QAT 授权令牌缺失或不匹配', {
      remediation: `必须带 confirm="${expected ?? '<plan token>'}" 重新调用；该令牌与具体模型+方法绑定，且每次都要重新确认（不缓存）。`,
      details: { expected, provided: confirmToken, planId: plan?.planId ?? null },
    })
  }
  if (!allowSandboxDegrade) {
    fail(ERROR_CODES.E_QAT_UNAVAILABLE, 'QAT 需要 VM 级沙箱（R3），本机不可用；未获显式降级授权', {
      remediation: '若要执行，需同时传 allowSandboxDegrade=true（会记入审计并在报告中标注"沙箱降级"）。'
        + '本插件不会替你按下开始键 —— 授权后仍需你自行执行计划里的命令。',
      details: { sandboxRequired: 'vm', available: runtime.policy?.risk?.sandbox?.R3 ?? 'vm' },
    })
  }
  return {
    ok: true,
    planId: plan.planId,
    sandboxDegradeAuthorized: true,
    cacheable: false,
    executes: false,
    note: '授权已记录。本插件**不自动执行** QAT；请按计划里的 config 与 data 自行执行，'
      + '完成后用 inference_quantization_quantize 对产出做门禁验收（可复用同一套阈值与测试集）。',
    commands: [
      `# 1) 扩容校准集并重新过去泄漏门禁`,
      `inference-quantization dataset prepare --repo ${plan.data.calibrationSet.repo} --config ${plan.data.calibrationSet.config} --seed ${plan.data.calibrationSet.seed} --calib 5000`,
      `# 2) 在量化镜像内执行 QAT（参数见 plan.config；本插件不代跑）`,
      `docker run --rm --runtime nvidia --gpus all -v ${plan.config.baseModel}:${plan.config.baseModel}:ro -v <outputDir>:<outputDir> --entrypoint python3 ${plan.config.image} /scripts/qat.py ...`,
      `# 3) 用同一套阈值与测试集做门禁验收`,
      `inference-quantization quantize <outputDir> --max-accuracy-drop-pct ${plan.expectedGain.thresholdPct} --min-perf-gain-pct <target>`,
    ],
  }
}
