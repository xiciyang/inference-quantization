/**
 * 一次性历史回溯：给已被判定**不可作为验收证据**的 run 补上机读失效标记（F-5）。
 *
 * ## 为什么需要这个脚本
 *
 * `qr_25b26474a21b` 产生于一个 223/338 张量全零的残缺模型（`model.norm.weight` 整块为零），
 * 它记录的 `accuracy.dropPct = 0` / `ppl = 151935.97`（= vocab_size，均匀输出签名）/
 * `status = verified` **全部无效**。作废这件事此前只写在 `DEPLOYMENT_REPORT.md` 与
 * `ACCEPTANCE_MATRIX.md` 的**散文**里。
 *
 * 集成测试的结论是：下游（serving / tuning / 编排器 / RSI）读的是**状态文件**，
 * 散文里的作废声明对它们**不存在**。而 serving 确实据此上线过端点 `ep_4d4a641dc2`。
 *
 * 本脚本把那份散文判定**回填**成机读标记。**幂等**：已标记的不会重复写。
 *
 * 用法：
 *   node scripts/retro-mark-invalid.mjs            # 只报告（零写入）
 *   node scripts/retro-mark-invalid.mjs --apply    # 真写
 */
import os from 'node:os'
import { createRuntime } from '../src/config.mjs'
import { withMatrix, findRun, markRunInvalid, isRunInvalid } from '../src/state/matrix.mjs'

/**
 * 历史失效清单。**每一条都要有可复核的依据**，不接受"感觉不对"。
 */
export const KNOWN_INVALID_RUNS = Object.freeze([
  {
    id: 'qr_25b26474a21b',
    code: 'E_MEASUREMENT_INVALID',
    reason: '源模型残缺（ms_cache 副本 338 张量中 223 个整块为零，含 model.norm.weight），'
      + '输出退化为均匀分布；其 accuracy.dropPct=0 / ppl=151935.97(=vocab_size) / status=verified 均不可作为验收证据',
    evidence: {
      sourceModel: '<MODELS_DIR>/ms_cache/.../Qwen2.5-1.5B-Instruct/snapshots/master',
      uniformOutputSignature: 'ppl == vocab_size == 151936（exp(11.93121445743442)）',
      replacedBy: 'qr_45e79f22f441（完好源模型上重跑，dropPct=19.571%，measurementValid=true）',
      voidDeclaredIn: ['DEPLOYMENT_REPORT.md §五', 'ACCEPTANCE_MATRIX.md FIX-3'],
      detectedBy: 'inference-quantization/src/model-validate.mjs（FIX-0，2.6 s 拦下）',
    },
  },
])

export async function retroMark({ root = process.env.AI_INFRA_ROOT || os.homedir() + '/.ai-infra', apply = false } = {}) {
  const runtime = await createRuntime({ root, options: {} })
  const report = { apply, marked: [], alreadyMarked: [], notFound: [], runsScanned: 0 }

  await withMatrix(runtime, (m) => {
    report.runsScanned = m.runs.length
    for (const spec of KNOWN_INVALID_RUNS) {
      const run = findRun(m, spec.id)
      if (!run) { report.notFound.push(spec.id); continue }
      if (isRunInvalid(run)) { report.alreadyMarked.push(spec.id); continue }
      if (!apply) { report.marked.push({ id: spec.id, code: spec.code, dryRun: true }); continue }
      markRunInvalid(run, { code: spec.code, reason: spec.reason, evidence: spec.evidence })
      report.marked.push({ id: spec.id, code: spec.code })
    }
    return null
  }, { backup: apply })

  return report
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`
if (isMain) {
  const apply = process.argv.includes('--apply')
  const r = await retroMark({ apply })
  process.stdout.write(`${JSON.stringify(r, null, 2)}\n`)
  if (!apply) process.stdout.write('\n（这是预演；加 --apply 才真写）\n')
}
