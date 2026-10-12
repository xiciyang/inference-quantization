/**
 * 稳定错误码表 + 结构化错误类型。
 *
 * 设计要点（对应验收标准 5.2.5「错误信息可操作，指出具体问题和修正方式」）：
 *   每个错误码都必须带 remediation —— 不是"出了错"，而是"出了什么错、怎么修"。
 *   错误码是接口的一部分：一旦发布即冻结，破坏性变更升 MCP major（题目原则 3.1.5）。
 */

export const ERROR_CODES = Object.freeze({
  // --- 输入处理（7 类输入场景）---
  E_INPUT_INVALID: 'E_INPUT_INVALID',
  E_INPUT_CONFLICT: 'E_INPUT_CONFLICT',
  E_INPUT_AMBIGUOUS: 'E_INPUT_AMBIGUOUS',
  E_INPUT_MISSING: 'E_INPUT_MISSING',
  E_MODEL_UNRESOLVED: 'E_MODEL_UNRESOLVED',
  E_LICENSE_UNRESOLVED: 'E_LICENSE_UNRESOLVED',

  // --- 权限与能力清单 ---
  E_CAPABILITY_DENIED: 'E_CAPABILITY_DENIED',
  E_RESOURCE_NOT_DECLARED: 'E_RESOURCE_NOT_DECLARED',

  // --- 沙箱 ---
  E_SANDBOX_UNAVAILABLE: 'E_SANDBOX_UNAVAILABLE',
  E_SANDBOX_PROBE_FAILED: 'E_SANDBOX_PROBE_FAILED',
  E_HOST_SANDBOX_MISMATCH: 'E_HOST_SANDBOX_MISMATCH',

  // --- 确认与交互 ---
  E_CONFIRMATION_REQUIRED: 'E_CONFIRMATION_REQUIRED',
  E_HEADLESS_NEEDS_CONFIRMATION: 'E_HEADLESS_NEEDS_CONFIRMATION',

  // --- 网络出口 ---
  E_EGRESS_DENIED: 'E_EGRESS_DENIED',

  // --- 部署与降级 ---
  E_BACKEND_UNKNOWN: 'E_BACKEND_UNKNOWN',
  E_BACKEND_UNAVAILABLE: 'E_BACKEND_UNAVAILABLE',
  E_PORT_UNAVAILABLE: 'E_PORT_UNAVAILABLE',
  E_INSUFFICIENT_MEMORY: 'E_INSUFFICIENT_MEMORY',
  E_MEMORY_CONFIG_INFEASIBLE: 'E_MEMORY_CONFIG_INFEASIBLE',
  E_DEPLOY_FAILED: 'E_DEPLOY_FAILED',
  E_DEPLOY_TIMEOUT: 'E_DEPLOY_TIMEOUT',
  E_DEGRADE_EXHAUSTED: 'E_DEGRADE_EXHAUSTED',
  E_HEALTH_FAILED: 'E_HEALTH_FAILED',
  E_CANCELLED: 'E_CANCELLED',

  // --- 状态与生命周期 ---
  E_STATE_CONFLICT: 'E_STATE_CONFLICT',
  E_ENDPOINT_NOT_FOUND: 'E_ENDPOINT_NOT_FOUND',
  E_NO_ROLLBACK_TARGET: 'E_NO_ROLLBACK_TARGET',
  E_TEARDOWN_FAILED: 'E_TEARDOWN_FAILED',
  E_CLEANUP_INCOMPLETE: 'E_CLEANUP_INCOMPLETE',

  // --- 量化特有 ---
  E_ACCURACY_GATE_MISSING: 'E_ACCURACY_GATE_MISSING',
  E_THRESHOLD_NOT_PROVIDED: 'E_THRESHOLD_NOT_PROVIDED',
  E_CALIB_LEAKAGE_DETECTED: 'E_CALIB_LEAKAGE_DETECTED',
  E_DATASET_UNAVAILABLE: 'E_DATASET_UNAVAILABLE',
  E_SOURCE_MODEL_MODIFIED: 'E_SOURCE_MODEL_MODIFIED',
  E_SOURCE_MODEL_UNREADABLE: 'E_SOURCE_MODEL_UNREADABLE',
  // 结构残缺（截断 / 零填充 / 层数不符 / 分片缺失）—— 与"被改动"是两回事：
  // 前者是"内容不是一个完整模型"，后者是"内容被改过"。目录哈希只能回答后者。
  E_SOURCE_MODEL_INVALID: 'E_SOURCE_MODEL_INVALID',
  // **测量不成立**（与"测量值没达标"是两回事）：模型输出均匀分布、
  // 基线与候选逐位相同、逐条无方差…… 这些情形下"精度损失"这个数没有意义，
  // 门禁必须拒绝签收，而不是把 0% 当成"无损"。
  E_MEASUREMENT_INVALID: 'E_MEASUREMENT_INVALID',
  E_OUTPUT_DIR_UNSAFE: 'E_OUTPUT_DIR_UNSAFE',
  E_METHOD_UNKNOWN: 'E_METHOD_UNKNOWN',
  E_METHOD_UNAVAILABLE: 'E_METHOD_UNAVAILABLE',
  E_QUANTIZE_FAILED: 'E_QUANTIZE_FAILED',
  E_QUANTIZE_TIMEOUT: 'E_QUANTIZE_TIMEOUT',
  E_ACCURACY_GATE_FAILED: 'E_ACCURACY_GATE_FAILED',
  E_PERF_GATE_FAILED: 'E_PERF_GATE_FAILED',
  E_GRADIENT_NEAR_LIMIT: 'E_GRADIENT_NEAR_LIMIT',
  E_DEGRADE_EXHAUSTED: 'E_DEGRADE_EXHAUSTED',
  E_QAT_REQUIRES_APPROVAL: 'E_QAT_REQUIRES_APPROVAL',
  E_QAT_UNAVAILABLE: 'E_QAT_UNAVAILABLE',
  E_RUN_NOT_FOUND: 'E_RUN_NOT_FOUND',
  E_NO_QUANTIZATION_TARGET: 'E_NO_QUANTIZATION_TARGET',
  E_IMAGE_NOT_VERIFIED: 'E_IMAGE_NOT_VERIFIED',
  E_IMAGE_SCAN_FAILED: 'E_IMAGE_SCAN_FAILED',
  E_SERVING_UNAVAILABLE: 'E_SERVING_UNAVAILABLE',
  E_VLLM_INCOMPATIBLE: 'E_VLLM_INCOMPATIBLE',
  E_TESTING_CORE_ERROR: 'E_TESTING_CORE_ERROR',
  // --- 内部 ---
  E_INTERNAL: 'E_INTERNAL',
})

export const WARNING_CODES = Object.freeze({
  W_PROXY_DEAD: 'W_PROXY_DEAD',
  W_ASSUMPTION_FILLED: 'W_ASSUMPTION_FILLED',
  W_POLICY_DEFAULTED: 'W_POLICY_DEFAULTED',
  W_AMBIGUITY_RESOLVED: 'W_AMBIGUITY_RESOLVED',
  W_SYMLINK_MODEL_ROOT: 'W_SYMLINK_MODEL_ROOT',
  W_VM_SANDBOX_ABSENT: 'W_VM_SANDBOX_ABSENT',
  W_RESIDUAL_CLEANUP: 'W_RESIDUAL_CLEANUP',
  W_HIGH_MEMORY_REQUEST_ON_UNIFIED: 'W_HIGH_MEMORY_REQUEST_ON_UNIFIED',
  W_IMAGE_PROBE_FALLBACK: 'W_IMAGE_PROBE_FALLBACK',
  W_COARSE_SCREEN: 'W_COARSE_SCREEN',
  W_GRADIENT_NEAR_LIMIT: 'W_GRADIENT_NEAR_LIMIT',
  W_SOURCE_SYMLINK: 'W_SOURCE_SYMLINK',
  W_QAT_PLAN_ONLY: 'W_QAT_PLAN_ONLY',
  W_LEAKAGE_DEDUP: 'W_LEAKAGE_DEDUP',
  W_INTERNAL_SHARED_LIB: 'W_INTERNAL_SHARED_LIB',
  W_SOURCE_VALIDATION_CACHED: 'W_SOURCE_VALIDATION_CACHED',
  W_SOURCE_VALIDATION_UNSUPPORTED: 'W_SOURCE_VALIDATION_UNSUPPORTED',
  W_MEASUREMENT_VALIDITY: 'W_MEASUREMENT_VALIDITY',
  W_CONVERSION_APPLIED: 'W_CONVERSION_APPLIED',
  W_DEGRADE_LOSSY: 'W_DEGRADE_LOSSY',
})

/** 结构化错误。`code` 取自 ERROR_CODES，`remediation` 必须可操作。 */
export class ServingError extends Error {
  constructor(code, message, { remediation, details, cause, retriable = false, capability } = {}) {
    super(message, cause ? { cause } : undefined)
    this.name = 'ServingError'
    this.code = code
    this.remediation = remediation ?? null
    this.details = details ?? null
    this.retriable = retriable
    this.capability = capability ?? null
  }

  toJSON() {
    return {
      ok: false,
      code: this.code,
      message: this.message,
      remediation: this.remediation,
      details: this.details,
      retriable: this.retriable,
    }
  }
}

/** 构造并抛出。`fail(code, msg, opts)`。 */
export function fail(code, message, opts) {
  throw new ServingError(code, message, opts)
}

export function isServingError(e) {
  return e instanceof ServingError
}

/** 把任意异常规范化为可序列化载荷 —— 保证故障隔离：调用方永远拿到结构化结果。 */
export function toErrorPayload(err, { capability } = {}) {
  if (isServingError(err)) {
    const p = err.toJSON()
    if (capability && !p.capability) p.capability = capability
    return p
  }
  return {
    ok: false,
    code: ERROR_CODES.E_INTERNAL,
    message: err && err.message ? String(err.message) : String(err),
    remediation: '这是未预期的内部错误；请附上 code 与本次调用的 auditId 提交问题。',
    details: err && err.stack ? { stack: String(err.stack).split('\n').slice(0, 8) } : null,
    retriable: false,
    capability: capability ?? null,
  }
}

/** 带 code 的警告（非致命），用于把自动决策记录进报告。 */
export class ServingWarning {
  constructor(code, message, details) {
    this.code = code
    this.message = message
    this.details = details ?? null
  }

  toJSON() {
    return { code: this.code, message: this.message, details: this.details }
  }
}

export function warn(code, message, details) {
  return new ServingWarning(code, message, details)
}
