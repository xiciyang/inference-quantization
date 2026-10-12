/**
 * 量化主流程（题目核心闭环）。
 *
 *   模型 + 精度容忍 + 性能目标
 *     → 选方法（默认 AWQ+INT4）
 *     → PTQ 校准（内容哈希去泄漏）
 *     → 精度对比（testing-core parity-check）
 *     → 性能对比（testing-core perf-validation）
 *     → 双门禁
 *     → 通过 → 写矩阵标记 verified
 *     → 失败 → 降级链 INT4→INT8→FP8→W8A16→QAT计划→回退
 *     → 输出量化模型 + 报告
 *
 * 状态迁移与不变量已由 logicprobe 穷举验证（docs/step0/model-degradation-fsm.json）：
 * 有界（level ≤ 5）、无死锁、守卫完备、**签收必经 acc_pass 与 perf_pass**、
 * **进入量化前必经 calib_ok（去泄漏）**；最长路径 15 步 → 超时预算按 16 留余量。
 */
import { join } from 'node:path'
import { promises as fs } from 'node:fs'
import { shortId, sha256, canonicalJson } from './util/hash.mjs'
import { readText, atomicWriteJson, mkdirp, exists } from './util/fsx.mjs'
import { fail, warn, ERROR_CODES, WARNING_CODES } from './errors.mjs'
import { runInputPipeline } from './input/pipeline.mjs'
import { allMethods, getMethod } from './backends/index.mjs'
import { hashDirectory, assertSourceUnchanged, assertOutputDirSafe } from './integrity.mjs'
import { assertSourceModelValid } from './model-validate.mjs'
import { prepareDataset, loadDatasetSplit } from './dataset/prepare.mjs'
import { assertNoLeakage, verifyLineage } from './dataset/leakage.mjs'
import { parityCheck, coarseScreen, TESTING_CORE_CONTRACT_VERSION } from './testing/parity.mjs'
import { perfValidation } from './testing/perf.mjs'
import { resolveThresholds, evaluateGates, assertMeasurementValid, nearLimitTokenFor } from './gate.mjs'
import { planNextMethod, describeChain, isDegradable } from './degrade.mjs'
import { withServing, callWithConfirm } from './serving-client.mjs'
import { assertImageVerified } from './image.mjs'
import { withMatrix, findRun, pushHistory, summarizeRun, markRunInvalid } from './state/matrix.mjs'
import { writeProvenance } from './state/provenance.mjs'
import { recordQuantizeTiming, recordDegrade, recordGate, bump } from './metrics.mjs'
import { assertSandbox } from './sandbox.mjs'
import { buildQatPlan, QAT_APPROVAL_TOKEN } from './qat.mjs'
import { assertCapability } from './capability.mjs'
import { executorFor } from './executors/index.mjs'
import './executors/docker.mjs'

/** 单档量化：交给执行器（生产是容器，测试是进程内假件）。 */
export async function runQuantizeInContainer(runtime, opts) {
  const executor = executorFor(runtime)
  const started = Date.now()
  runtime.__log?.phase?.('quantize-execute', { method: opts.methodId, executor: executor.id })
  const r = await executor.quantize(runtime, opts)
  return { ...r, durationMs: r.durationMs ?? (Date.now() - started) }
}

/** 校验量化产出：有 quantization_config、有权重、字节数合理。 */
export async function verifyQuantOutput(outputPath, sourceBytes) {
  const cfgRaw = await readText(join(outputPath, 'config.json'), null)
  if (!cfgRaw) return { ok: false, reason: '产出缺 config.json' }
  let cfg
  try { cfg = JSON.parse(cfgRaw) } catch (e) { return { ok: false, reason: `config.json 解析失败：${e.message}` } }
  const qc = cfg.quantization_config
  if (!qc) return { ok: false, reason: 'config.json 里没有 quantization_config（说明没真的量化）' }
  const files = await fs.readdir(outputPath)
  const weights = files.filter((f) => /\.(safetensors|bin)$/.test(f))
  if (weights.length === 0) return { ok: false, reason: '产出目录没有权重文件' }
  let bytes = 0
  for (const w of weights) bytes += (await fs.stat(join(outputPath, w))).size
  return {
    ok: true,
    quantMethod: qc.quant_method ?? null,
    format: qc.format ?? null,
    version: qc.version ?? null,
    scheme: qc.config_groups?.group_0?.weights ? `${qc.config_groups.group_0.weights.num_bits}bit/${qc.config_groups.group_0.weights.strategy}` : null,
    weightFiles: weights.length,
    bytes,
    compressionRatio: sourceBytes && bytes ? Number((sourceBytes / bytes).toFixed(3)) : null,
    quantizationConfig: qc,
  }
}

/**
 * 主流程。
 * @returns {{ok, run, report}}
 */
export async function quantizeOperation(runtime, { raw, log = null, signal = null, onPhase = null } = {}) {
  const startedAt = Date.now()
  const attempts = []
  const phase = (name, data) => { log?.phase?.(name, data); onPhase?.({ name, ...(data ?? {}), at: Date.now() - startedAt }) }

  // ---------- 0. 输入 ----------
  phase('probe')
  const { standard, env } = await runInputPipeline(runtime, raw, { log })

  // ---------- 1. 沙箱（R2 容器级） ----------
  phase('sandbox')
  const sandbox = await assertSandbox(runtime, runtime.policy.risk.sandbox?.R2 ?? 'container')

  // ---------- 2. 阈值（不猜默认） ----------
  // 刻意排在镜像检查之前：**配置类错误应最先失败**（不需要碰环境、更便宜、更可操作）。
  // 若阈值缺失却先去查镜像，用户会先收到一个与真正问题无关的错误。
  phase('thresholds')
  const thresholds = resolveThresholds(runtime, {
    maxAccuracyDropPct: standard.thresholds.maxAccuracyDropPct,
    minPerfGainPct: standard.thresholds.minPerfGainPct,
  })
  log?.info('门禁阈值已解析', thresholds)

  // ---------- 3. 镜像可信度 ----------
  phase('image')
  const imageMeta = await assertImageVerified(runtime, {
    image: runtime.options.quantImage,
    allowUnverified: standard.allowUnverifiedImage === true,
  })

  // ---------- 4. 输出目录安全 ----------
  assertOutputDirSafe(standard.source.path, standard.outputPath)

  // ---------- 4b. 源模型结构预检（"是完整模型"≠"文件没被改过"） ----------
  // 刻意排在数据集准备之前：坏输入要在**最便宜**的地方失败。
  // 真机教训：一份下载中断的模型（271/338 张量全为零）被量化了 436 秒并签收"精度零损失"。
  // 目录哈希只能回答"内容有没有被改动"，回答不了"内容是不是一个完整模型"。
  phase('source-validate')
  const sourceValidation = await assertSourceModelValid(runtime, standard.source.path, {
    useCache: standard.validateSourceCache !== false,
  })

  // ---------- 5. 数据集 + 去泄漏门禁 ----------
  phase('dataset')
  const ds = await prepareDataset(runtime, {
    repo: standard.dataset.repo, config: standard.dataset.config, split: standard.dataset.split,
    seed: standard.seed, calibrationSamples: standard.calibrationSamples, testSamples: standard.dataset.testSamples,
    log,
  })
  // 双重把关：prepareDataset 内部已 assertNoLeakage，这里再独立复核一次（不信任上游）
  assertNoLeakage(ds.verification)
  const calibPath = join(ds.dir, 'calib.jsonl')
  const testPath = join(ds.dir, 'test.jsonl')

  const runId = `qr_${shortId(`${standard.source.path}|${standard.method}|${Date.now()}`, 12)}`
  const runKey = sha256(canonicalJson({
    source: standard.source.path, method: standard.method,
    calibration: { seed: standard.seed, samples: standard.calibrationSamples, listDigest: ds.verification.calibListDigest },
  })).slice(0, 16)

  // ---------- 6. 降级链主循环 ----------
  const methods = allMethods()
  const triedLevels = []
  const triedMethods = []
  let finalCandidate = null
  let finalGates = null
  let finalParity = null
  let finalPerf = null
  let outcome = null
  let lastFailure = null
  let qatPlan = null
  let method = getMethod(standard.method)

  const maxSteps = Math.min(runtime.options.maxDegradeSteps + 1, 5) // 4 个可执行档 + QAT 档

  while (true) {
    const level = method?.level ?? 1
    const attempt = {
      level, method: method?.id ?? standard.method, methodSpec: method?.describe(env) ?? null,
      startedAt: new Date().toISOString(), preflight: null, quantize: null, parity: null,
      measurement: null, gates: null, failureCode: null, ms: 0,
    }
    let measurement = null
    const aStart = Date.now()
    const outputPath = triedLevels.length === 0
      ? standard.outputPath
      : `${standard.outputPath}-L${level}`

    // 6a) 预检
    const pre = await method.preflight({ env, sourceBytes: standard.sourceIntegrity?.before?.bytes ?? null })
    attempt.preflight = { ok: pre.ok, code: pre.code ?? null, message: pre.message ?? null, details: pre.details ?? null }
    let failureCode = pre.ok ? null : pre.code

    // 6b) 量化
    if (pre.ok) {
      assertOutputDirSafe(standard.source.path, outputPath)
      const q = await runQuantizeInContainer(runtime, {
        sourcePath: standard.source.path, outputPath, methodId: method.id,
        calibPath, samples: standard.calibrationSamples, maxSeqLen: standard.maxSeqLen ?? 2048,
        seed: standard.seed, log, signal,
      })
      attempt.quantize = {
        ok: q.ok, durationMs: q.durationMs, exitCode: q.exitCode ?? null,
        failureCode: q.failureCode ?? null, checks: q.parsed?.checks ?? null,
        toolchain: q.parsed?.toolchain ?? null, stderr: q.ok ? null : (q.stderr ?? '').slice(-800),
      }
      if (!q.ok) failureCode = q.failureCode
    }

    // 6c) 产出校验
    let outputCheck = null
    if (!failureCode) {
      outputCheck = await verifyQuantOutput(outputPath, standard.sourceIntegrity?.before?.bytes ?? null)
      attempt.outputCheck = outputCheck
      if (!outputCheck.ok) failureCode = 'E_QUANTIZE_FAILED'
    }

    // 6d) 精度（先粗筛；通过后再做完整评测）
    let parity = null
    if (!failureCode) {
      phase('parity-coarse', { level, method: method.id })
      const coarse = await parityCheck(runtime, {
        testSetPath: testPath, baselineModelPath: standard.source.path, candidateModelPath: outputPath,
        limit: Math.max(runtime.options.coarseScreenMinSamples, 200), label: `L${level}-coarse`, log,
      })
      const screen = coarseScreen(coarse, {
        minSamples: runtime.options.coarseScreenMinSamples,
        relaxFactor: runtime.options.coarseScreenRelaxFactor,
      })
      attempt.parityCoarse = { dropPct: screen.dropPct, usedSamples: screen.usedSamples }
      runtime.warnings.push(warn(WARNING_CODES.W_COARSE_SCREEN,
        `L${level} 粗筛：精度损失 ${screen.dropPct?.toFixed(3)}%（样本 ${screen.usedSamples}，阈值放宽 ${runtime.options.coarseScreenRelaxFactor}×）`,
        { level, method: method.id, dropPct: screen.dropPct, usedSamples: screen.usedSamples }))

      // **先判测量是否成立，再判数值是否达标** —— 顺序不能反。
      // 测量不成立时数值没有意义（真机事故：均匀输出 → dropPct = 0 → 误判"无损"）。
      // 放在粗筛位置是为了在最便宜的观测点上就失败（不白跑完整评测 + 性能评测）。
      const coarseMeasurement = assertMeasurementValid({
        baseline: coarse.baseline, candidate: coarse.candidate,
      })
      attempt.measurement = coarseMeasurement
      measurement = coarseMeasurement
      if (!coarseMeasurement.valid) {
        failureCode = 'E_MEASUREMENT_INVALID'
        attempt.measurementRejected = true
        runtime.warnings.push(warn(WARNING_CODES.W_MEASUREMENT_VALIDITY,
          `L${level} 测量不成立，拒绝继续：${coarseMeasurement.reasons.map((r) => r.code).join('、')}`,
          { level, method: method.id, reasons: coarseMeasurement.reasons }))
        log?.info('测量不成立，停止本档位（不进入完整评测）', {
          level, codes: coarseMeasurement.reasons.map((r) => r.code),
        })
      } else {
        const relaxedLimit = thresholds.maxAccuracyDropPct * runtime.options.coarseScreenRelaxFactor
        if (screen.ok && screen.dropPct !== undefined && screen.dropPct > relaxedLimit) {
          failureCode = 'E_ACCURACY_GATE_FAILED'
          attempt.parityCoarse.rejected = true
          log?.info('粗筛已判定精度不达标，跳过完整评测', { level, dropPct: screen.dropPct, relaxedLimit })
        } else {
          phase('parity-full', { level, method: method.id })
          parity = await parityCheck(runtime, {
            testSetPath: testPath, baselineModelPath: standard.source.path, candidateModelPath: outputPath,
            limit: 0, label: `L${level}-full`, log,
          })
          attempt.parity = { dropPct: parity.drop.pct, samples: parity.samples, baselinePpl: parity.baseline.perplexity, candidatePpl: parity.candidate.perplexity }
          // 完整评测再独立复核一次（不信任粗筛的结论，也不信任上游）
          measurement = assertMeasurementValid({ baseline: parity.baseline, candidate: parity.candidate })
          attempt.measurement = measurement
          if (!measurement.valid) {
            failureCode = 'E_MEASUREMENT_INVALID'
            attempt.measurementRejected = true
            log?.info('完整评测判定测量不成立', { level, codes: measurement.reasons.map((r) => r.code) })
          } else if (parity.drop.pct === null) {
            failureCode = 'E_ACCURACY_GATE_MISSING'
          }
        }
      }
    }

    // 6e) 双门禁（性能只在精度通过后测 —— 精度不及格测性能是浪费）
    let perf = null
    let gates = null
    if (!failureCode) {
      phase('perf', { level, method: method.id })
      perf = await executorFor(runtime).perf(runtime, {
        baselineModelPath: standard.source.path, candidateModelPath: outputPath,
        workload: standard.perf.workload, requests: standard.perf.requests, concurrency: standard.perf.concurrency, log,
      })
      if (!perf?.ok) {
        fail(ERROR_CODES.E_SERVING_UNAVAILABLE, `性能评测失败：${perf?.error ?? perf?.code ?? 'unknown'}`, {
          remediation: '性能评测需要 inference-serving 起真实端点。先跑 inference-serving doctor 自检，或用 options.servingPath 指定其路径。',
          details: perf ?? null,
        })
      }
      attempt.perf = {
        gainDecodePct: perf.gain.decodeTpsPct, gainThroughputPct: perf.gain.throughputPct,
        ttftReductionPct: perf.gain.ttftReductionPct,
        // 端点拆解结果进报告：资源卫生与性能数据同等重要（真机曾泄漏过 2 个容器）
        cleanupOk: perf.cleanup?.ok ?? null,
      }

      gates = evaluateGates({
        accuracy: { dropPct: parity.drop.pct },
        perf: { gainPct: perf.gain.decodeTpsPct },
        thresholds: { maxAccuracyDropPct: thresholds.maxAccuracyDropPct, minPerfGainPct: thresholds.minPerfGainPct },
        // 测量不成立 → 判定落到 undetermined、不可签收（数值再好也不签）
        measurement,
      })
      attempt.gates = { passed: gates.passed, decision: gates.decision, measurementValid: gates.measurementValid, reasons: gates.reasons }
      phase('gates', { level, decision: gates.decision, passed: gates.passed })
    }

    attempt.ms = Date.now() - aStart
    attempt.failureCode = failureCode
    attempts.push(attempt)
    triedLevels.push(level)
    triedMethods.push({ level, method: method.id, failureCode })
    await recordQuantizeTiming(runtime, attempt.ms, { success: !failureCode })

    if (!failureCode && gates) {
      finalCandidate = outputPath
      finalGates = gates
      finalParity = parity
      finalPerf = perf
      // ── 产物溯源边车（F-4）──────────────────────────────────────────
      //
      // 真机事故：量化产物目录里**只有量化配方**（recipe.yaml），没有来源模型信息，
      // 而 serving 只按产物路径查 license → 判 `unknown` → 要求人工重新确认。
      // 结果：headless 下每个派生模型都会撞 E_HEADLESS_NEEDS_CONFIRMATION，链路卡死
      // （非 headless 下也要人再点一次，虽然基座的 license 早已确认过）。
      //
      // 溯源信息本来就是现成的（矩阵里有 outputPath → sourceModel），只是没有落到**产物**里。
      // 现在落一份机读边车，serving 据此把基座的 license 确认沿链继承。
      try {
        await writeProvenance(runtime, {
          outputPath, runId, standard, method, level, sourceRunKey: runKey,
        })
      } catch (err) {
        // 边车写失败不能把一次成功的量化搞失败 —— 但要留痕
        runtime.warnings.push(warn(WARNING_CODES.W_POLICY_DEFAULTED,
          `溯源边车写入失败（不影响本次产出，但下游无法自动继承 license）：${String(err.message).slice(0, 200)}`,
          { outputPath }))
      }
      if (gates.decision === 'sign' || gates.decision === 'confirm-near-limit') { outcome = 'gated'; break }
      // 双门禁未过（passed=false）→ 降级
      failureCode = !gates.accuracy.passed ? 'E_ACCURACY_GATE_FAILED' : 'E_PERF_GATE_FAILED'
      attempt.failureCode = failureCode
    }
    lastFailure = failureCode

    // 6f) 降级决策
    const plan = planNextMethod({ failureCode, triedLevels, methods, maxSteps })
    attempt.degradeDecision = plan
    if (plan.terminal) { outcome = 'aborted-terminal'; break }
    if (plan.exhausted) { outcome = 'exhausted'; break }
    if (plan.qatPlan) {
      qatPlan = await buildQatPlan(runtime, {
        standard, env, attempts, thresholds,
        lastParity: parity, lastPerf: perf, lastFailure: failureCode, log,
      })
      outcome = 'qat-plan'
      break
    }
    await recordDegrade(runtime, { step: triedLevels.length, reason: failureCode })
    phase('degrade', { to: plan.level, method: plan.method.id, from: failureCode })
    method = plan.method
  }

  // ---------- 7. 收尾：源模型完整性 ----------
  phase('integrity')
  let integrity = null
  if (standard.sourceIntegrity) {
    integrity = await assertSourceUnchanged(standard.source.path, standard.sourceIntegrity.before, { log })
  }

  // ---------- 8. 结果裁定 ----------
  const baseReport = {
    schema: 'inference-quantization/quantize-report@1',
    runId, runKey,
    generatedAt: new Date().toISOString(),
    durationMs: Date.now() - startedAt,
    source: {
      input: standard.source.input, path: standard.source.path,
      hashBefore: standard.sourceIntegrity?.before?.hash ?? null,
      hashAfter: integrity?.after?.hash ?? null,
      unchanged: integrity?.unchanged ?? null,
      // "内容没被改动"回答不了"内容是不是一个完整模型" —— 两条证据并列给出
      validation: sourceValidation ?? null,
    },
    output: finalCandidate,
    method: triedMethods.at(-1)?.method ?? standard.method,
    attempts: attempts.map((a) => ({
      level: a.level, method: a.method, preflightOk: a.preflight?.ok ?? null, failureCode: a.failureCode,
      quantizeOk: a.quantize?.ok ?? null, quantizeMs: a.quantize?.durationMs ?? null,
      outputBytes: a.outputCheck?.bytes ?? null, compressionRatio: a.outputCheck?.compressionRatio ?? null,
      parityDropPct: a.parity?.dropPct ?? a.parityCoarse?.dropPct ?? null,
      perfGainPct: a.perf?.gainDecodePct ?? null,
      gatesDecision: a.gates?.decision ?? null, ms: a.ms,
      measurementValid: a.measurement?.valid ?? null,
      measurementReasons: a.measurement?.reasons?.map((r) => r.code) ?? null,
      degrade: a.degradeDecision?.level ? { to: a.degradeDecision.level, method: a.degradeDecision.method?.id } : (a.degradeDecision?.qatPlan ? { qatPlan: true } : null),
    })),
    degraded: triedLevels.length - 1,
    chain: describeChain(triedMethods),
    thresholds,
    thresholdSource: thresholds.source,
    gates: finalGates,
    // 测量成立性（与阈值判定并列的独立结论）—— 签收必须两者都过
    measurementValidity: {
      // 测量在粗筛阶段就被判不成立时，门禁根本没跑到 → 结论取自该档位自己的测量记录
      valid: finalGates?.measurementValid ?? (attempts.at(-1)?.measurement?.valid ?? null),
      perAttempt: attempts.map((a) => ({ level: a.level, method: a.method, valid: a.measurement?.valid ?? null, reasons: (a.measurement?.reasons ?? []).map((r) => r.code), checks: a.measurement?.checks ?? null })),
      sourceValidation: sourceValidation ? { ok: sourceValidation.ok, tensors: sourceValidation.evidence?.totalTensors ?? null, zeroTensors: sourceValidation.evidence?.totalZeroTensors ?? null, cached: sourceValidation.cached === true } : null,
    },
    parity: finalParity ? { metric: finalParity.metric, dropPct: finalParity.drop.pct, baseline: finalParity.baseline, candidate: finalParity.candidate, samples: finalParity.samples, validity: finalParity.validity, evidence: finalParity.evidence } : null,
    perf: finalPerf ? {
      metrics: finalPerf.metrics,
      baseline: { decodeTps: finalPerf.baseline.decodeTps.median, ttftMs: finalPerf.baseline.ttftMs.median },
      candidate: { decodeTps: finalPerf.candidate.decodeTps.median, ttftMs: finalPerf.candidate.ttftMs.median },
      // 性能端点拆解记录（evidence 里也有一份）
      cleanup: finalPerf.cleanup ?? null,
      evidence: finalPerf.evidence,
    } : null,
    leakage: { verified: ds.verification.verified, criterion: ds.verification.criterion, counts: ds.verification.counts, calibListDigest: ds.verification.calibListDigest, testListDigest: ds.verification.testListDigest, problems: ds.verification.problems },
    dataset: { dir: ds.dir, reused: ds.reused, lineage: ds.lineage?.source ?? null },
    image: { image: runtime.options.quantImage, digest: imageMeta.digest, verified: imageMeta.verified, missing: imageMeta.missing, toolchain: imageMeta.toolchain },
    sandbox: { tier: sandbox.tier, enforced: sandbox.enforced },
    testingCore: { contractVersion: TESTING_CORE_CONTRACT_VERSION, implementation: 'inference-quantization/internal (@internal 过渡实现)' },
    qatPlan,
    outcome,
    assumptions: standard.assumptions,
    warnings: runtime.warnings.map((w) => (typeof w.toJSON === 'function' ? w.toJSON() : w)),
  }

  // 写入量化矩阵
  const matrixRecord = await withMatrix(runtime, (m) => {
    let rec = findRun(m, runId)
    if (!rec) {
      rec = {
        id: runId, runKey,
        sourceModel: standard.source.path,
        createdAt: new Date().toISOString(),
        history: [],
      }
      m.runs.push(rec)
    }
    rec.method = baseReport.method
    rec.level = triedLevels.at(-1) ?? 1
    rec.bits = method?.bits ?? null
    // verified 需要**三条同时成立**：数值达标 + 决策为 sign + 测量成立。
    // 少任何一条都不算签收 —— 这正是"门禁不仅判数值、还要判测量"的落点。
    rec.status = outcome === 'gated' && finalGates?.decision === 'sign' && finalGates?.measurementValid !== false ? 'verified'
      : outcome === 'qat-plan' ? 'qat-plan-required'
        : 'failed'
    rec.outputPath = finalCandidate
    rec.accuracy = finalParity ? {
      metric: finalParity.metric, dropPct: finalParity.drop.pct,
      baselinePpl: finalParity.baseline.perplexity, candidatePpl: finalParity.candidate.perplexity,
      thresholdPct: thresholds.maxAccuracyDropPct, passed: finalGates?.accuracy?.passed ?? null,
      samples: finalParity.samples,
      // 测量成立性证据随精度结论一起落矩阵：只看到 dropPct=0 的人必须同时看到"这次测量成不成立"
      measurementValid: finalGates?.measurementValid ?? null,
      baselineMeanNll: finalParity.baseline.meanNll ?? null,
      candidateMeanNll: finalParity.candidate.meanNll ?? null,
      vocabSize: finalParity.baseline.vocabSize ?? null,
    } : null
    rec.performance = finalPerf ? { metric: finalPerf.metrics.primary, decodeTpsGainPct: finalPerf.gain.decodeTpsPct, throughputGainPct: finalPerf.gain.throughputPct, ttftReductionPct: finalPerf.gain.ttftReductionPct, thresholdPct: thresholds.minPerfGainPct, passed: finalGates?.perf?.passed ?? null } : null
    rec.platform = {
      gpu: standard.env.gpu, unifiedMemory: standard.env.unifiedMemory,
      container: standard.env.docker, nvidiaRuntime: standard.env.nvidiaRuntime,
      image: runtime.options.quantImage, imageDigest: imageMeta.digest,
      sandboxTier: sandbox.tier,
    }
    rec.leakage = { clean: ds.verification.verified, criterion: ds.verification.criterion, calibListDigest: ds.verification.calibListDigest, testListDigest: ds.verification.testListDigest, counts: ds.verification.counts }
    rec.sourceIntegrity = { unchanged: integrity?.unchanged ?? null, beforeHash: standard.sourceIntegrity?.before?.hash ?? null, afterHash: integrity?.after?.hash ?? null }
    rec.degrade = { steps: triedLevels.length - 1, chain: baseReport.chain }
    if (rec.status === 'verified') rec.verifiedAt = new Date().toISOString()

    // 测量不成立 → 落**机读**失效标记（F-5）。
    // 只写在报告散文里不够：下游读的是状态文件。真机教训见 matrix.mjs 的 markRunInvalid。
    if (finalGates?.measurementValid === false) {
      markRunInvalid(rec, {
        code: 'E_MEASUREMENT_INVALID',
        reason: `门禁判测量不成立：${(finalGates.measurementReasons ?? []).join('；') || '见 evidence'}`,
        evidence: {
          measurementReasons: finalGates.measurementReasons ?? null,
          accuracy: rec.accuracy ?? null,
          note: '这条 run 的数字（含"精度没掉"与"性能提升"）不得作为验收证据',
        },
      })
    }
    rec.reportPath = null
    pushHistory(rec, `quantize-${outcome}`, { method: rec.method, level: rec.level, gates: finalGates?.decision ?? null })
    return JSON.parse(JSON.stringify(rec))
  })

  // 报告落盘
  const reportPath = join(runtime.paths.reportsDir, `${runId}.json`)
  await mkdirp(runtime.paths.reportsDir)
  await atomicWriteJson(reportPath, baseReport)
  await withMatrix(runtime, (m) => {
    const rec = findRun(m, runId)
    if (rec) rec.reportPath = reportPath
    return null
  })
  baseReport.reportPath = reportPath

  if (finalGates) {
    await recordGate(runtime, { passed: finalGates.passed, decision: finalGates.decision })
  }

  if (outcome === 'gated' && finalGates.decision === 'confirm-near-limit') {
    const token = nearLimitTokenFor({ sourceModel: standard.source.path, method: baseReport.method, dropPct: finalParity.drop.pct })
    const provided = (raw?.options?.confirm ?? raw?.confirm ?? null)
    const tokens = Array.isArray(provided) ? provided : (provided ? [provided] : [])
    runtime.warnings.push(warn(WARNING_CODES.W_GRADIENT_NEAR_LIMIT, `精度损失接近上限：${finalGates.accuracy.reason}`, finalGates.accuracy))
    if (!tokens.includes(token)) {
      return {
        ok: false, code: ERROR_CODES.E_GRADIENT_NEAR_LIMIT,
        message: `双门禁均通过，但精度损失接近容忍上限，需要确认后签收：${finalGates.accuracy.reason}`,
        remediation: `确认可接受后带 confirm="${token}" 重新调用即可签收（该令牌与此模型+方法+精度值绑定）。`,
        details: { confirmToken: token, gates: finalGates, parity: baseReport.parity, runId },
        report: baseReport, run: matrixRecord,
      }
    }
  }

  if (outcome === 'gated' && finalGates.decision === 'sign') {
    await bump(runtime, 'quantize_success')
    return { ok: true, signed: true, run: matrixRecord, report: baseReport }
  }
  if (outcome === 'qat-plan') {
    await bump(runtime, 'quantize_qat_plan')
    return {
      ok: false, code: ERROR_CODES.E_QAT_REQUIRES_APPROVAL,
      message: `所有可执行量化档位均未通过双门禁（${describeChain(triedMethods)}）；已产出 QAT 计划，需你裁决`,
      remediation: '选择：① 批准 QAT（需显式授权从 R3 降级到容器级，且每次确认）；② 放宽精度容忍；③ 回退（不做量化）。'
        + 'QAT 计划含数据、配置、预期收益与回退路径，见 details.qatPlan。',
      details: { qatPlan, attempts: baseReport.attempts, gates: finalGates },
      report: baseReport, run: matrixRecord,
    }
  }
  await bump(runtime, 'quantize_failed')
  const code = outcome === 'aborted-terminal' ? (lastFailure ?? ERROR_CODES.E_QUANTIZE_FAILED) : ERROR_CODES.E_DEGRADE_EXHAUSTED
  fail(code,
    outcome === 'aborted-terminal'
      ? `量化中止：${lastFailure}（该失败无法通过更换量化方法解决）`
      : `降级链耗尽（共尝试 ${attempts.length} 档：${describeChain(triedMethods)}），未得到达标产出`, {
      remediation: outcome === 'aborted-terminal'
        ? '这是环境/输入类失败，换方法无用。按 details 里的 remediation 修复后重试。'
        : '可选：放宽精度容忍、降低性能目标、换更小的模型，或走 QAT（需批准）。完整尝试记录见 details.attempts。',
      details: { attempts: baseReport.attempts, tried: triedMethods, lastFailure, reportPath },
    })
}

export { isDegradable, withServing, callWithConfirm, loadDatasetSplit, verifyLineage, exists }
