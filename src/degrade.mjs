/**
 * 降级链（题面：INT4→INT8→FP8→W8A16→QAT→回退）。
 *
 * 与 serving 的降级链同构，但有两点不同（均来自已确认的裁决）：
 *   ① **有界到 5 级**：4 个可执行档 + 第 5 档"QAT 计划"。第 5 档**不执行训练**，
 *      只产出计划并停下等裁决（E_QAT_REQUIRES_APPROVAL）。超过 5 级 → 回退并标记。
 *   ② **不可降级的失败立即中止**：源模型被改、校准泄漏、阈值缺失、serving 不可用、
 *      镜像未验证 —— 这些换档位也没用，重试只是假装在努力。
 *
 * 状态机最长路径 15 步（logicprobe 实测），因此**超时预算按 16 留余量**。
 */
import { METHODS_ORDER_FALLBACK } from './policy-defaults.mjs'

/** 换档位解决不了的失败。 */
export const TERMINAL_FAILURES = new Set([
  'E_SOURCE_MODEL_MODIFIED',
  'E_SOURCE_MODEL_UNREADABLE',
  // 源模型结构残缺 → 换量化档位解决不了（换哪一档都是在坏权重上做量化）
  'E_SOURCE_MODEL_INVALID',
  // 测量不成立 → 换档位同样解决不了。若根因在候选产出，重跑同一档位也不会变好；
  // 这类问题需要人先修输入/环境，不是"再试一次"。
  'E_MEASUREMENT_INVALID',
  'E_OUTPUT_DIR_UNSAFE',
  'E_CALIB_LEAKAGE_DETECTED',
  'E_DATASET_UNAVAILABLE',
  'E_THRESHOLD_NOT_PROVIDED',
  'E_ACCURACY_GATE_MISSING',
  'E_SERVING_UNAVAILABLE',
  'E_IMAGE_NOT_VERIFIED',
  'E_IMAGE_SCAN_FAILED',
  'E_METHOD_UNKNOWN',
  'E_SANDBOX_UNAVAILABLE',
  'E_CAPABILITY_DENIED',
  'E_RESOURCE_NOT_DECLARED',
  'E_EGRESS_DENIED',
  'E_INPUT_INVALID',
  'E_INPUT_CONFLICT',
  'E_INPUT_AMBIGUOUS',
  'E_HEADLESS_NEEDS_CONFIRMATION',
  'E_CONFIRMATION_REQUIRED',
  'E_CANCELLED',
])

export function isDegradable(code) {
  return !TERMINAL_FAILURES.has(code)
}

/** 降级链的档位序列（可执行部分）。 */
export function chainMethods(availableMethods) {
  const ordered = [...availableMethods].sort((a, b) => a.level - b.level)
  return ordered.length ? ordered : METHODS_ORDER_FALLBACK.map((id, i) => ({ id, level: i + 1 }))
}

/**
 * 选下一档。
 * @param {object} p
 * @param {string} p.failureCode       本次失败码
 * @param {number[]} p.triedLevels     已试过的档位
 * @param {object[]} p.methods         本次运行可用的方法（已按 level 排序）
 * @param {number} p.maxSteps          最大档位数（默认 5，含 QAT 档）
 * @returns {{level, method}|{exhausted:true, terminal?:boolean, qatPlan?:boolean, reason:string}}
 */
export function planNextMethod({ failureCode, triedLevels = [], methods = [], maxSteps = 5 }) {
  if (!isDegradable(failureCode)) {
    return { exhausted: true, terminal: true, reason: `失败原因 ${failureCode} 无法通过更换量化方法解决` }
  }
  const tried = new Set(triedLevels)
  const next = methods.find((m) => !tried.has(m.level))
  if (next) return { level: next.level, method: next }

  // 可执行档位用尽 → 第 5 档：QAT 计划（不执行）
  if (!tried.has(5) && maxSteps >= 5) {
    return { exhausted: false, qatPlan: true, level: 5, reason: '可执行档位已用尽；进入第 5 档：产出 QAT 计划并停下等裁决' }
  }
  return { exhausted: true, qatPlan: false, reason: `已达到最大降级档位 ${maxSteps}` }
}

/** 人类可读的降级轨迹（进报告与审计）。 */
export function describeChain(tried) {
  return tried.map((t) => `L${t.level}:${t.method}${t.failureCode ? `(${t.failureCode})` : ''}`).join(' → ')
}

export { METHODS_ORDER_FALLBACK }
