/**
 * 工具定义（MCP 工具 + DSH 原生工具**共用同一份规格**）—— 与 serving 同构。
 */
import * as api from './api.mjs'

const ROOT = { type: 'string', description: '共享状态根目录。默认 ~/.ai-infra，也可用 INFERENCE_QUANTIZATION_ROOT 覆盖。' }
const CONFIRM = {
  type: 'string',
  description: '确认令牌。首次调用若需确认会返回 E_* 并在 details.confirmToken 给出该值；把原值回传即可继续。'
    + '本插件的确认令牌分三类且互不相同：风险确认 cf_*、精度接近上限 nlq_*、QAT 授权 qat_*。',
}

function passthrough() {
  return {
    schema: { type: 'object', additionalProperties: true },
    render(_a, v) { return [{ type: 'text', text: JSON.stringify(v ?? null, null, 2) }] },
  }
}

const ACCURACY_DROP = {
  type: 'number',
  description: '【必填，除非 gates.yaml 已设置】精度容忍：允许的最大精度损失（按困惑度相对退化的百分比）。'
    + '本插件**不猜默认值** —— 不给就拒绝执行（E_THRESHOLD_NOT_PROVIDED），因为量化不能自己定阈值、自己签收精度。',
}
const PERF_GAIN = {
  type: 'number',
  description: '【必填，除非 gates.yaml 已设置】性能收益目标：候选相对基线的解码吞吐至少要提升的百分比。',
}

export const TOOL_SPECS = [
  {
    name: 'inference_quantization_quantize',
    description:
      '对模型 checkpoint 做量化并**证明它达标**：选方法（默认 AWQ+INT4）→ PTQ 校准（校准集与测试集内容哈希必须不相交，否则中止）'
      + '→ 精度对比（同一测试集上算困惑度退化）→ 性能对比（通过 inference-serving 起真实端点测吞吐）→ 双门禁判定'
      + '→ 通过则写入量化矩阵标记 verified；失败自动降级 INT4→INT8→FP8→W8A16→（QAT 计划）→ 回退。'
      + '源模型只读，产出写独立目录，量化前后做哈希校验。',
    parameters: {
      type: 'object', additionalProperties: false,
      properties: {
        model: { type: 'string', description: '模型：本地路径（/path/to/models/<model>）、模型 ID、或环境变量 env:VAR。量化需要本地 checkpoint。' },
        path: { type: 'string', description: '显式指定本地模型目录。' },
        method: { type: 'string', description: '量化方法 id。默认 awq-w4a16；可选 int8-w8a8 / fp8-dynamic / w8a16。' },
        max_accuracy_drop_pct: ACCURACY_DROP,
        min_perf_gain_pct: PERF_GAIN,
        outputName: { type: 'string', description: '产出目录名（默认 <源模型名>-<方法id>）。必须与源不同名且在源目录之外。' },
        outputRoot: { type: 'string', description: '产出根目录（默认 ~/.ai-infra/quantized）。' },
        calibrationSamples: { type: 'integer', description: '校准样本数，默认 512。' },
        calibrationSeed: { type: 'integer', description: '切分种子，默认固定值（保证可复现）。' },
        datasetRepo: { type: 'string', description: '校准数据集 repo，默认 Salesforce/wikitext。' },
        datasetConfig: { type: 'string', description: '数据集 config，默认 wikitext-2-raw-v1。' },
        testSamples: { type: 'integer', description: '测试集样本数，默认 400。' },
        workload: { type: 'string', enum: ['chat-short', 'chat-medium', 'decode-heavy'], description: '性能评测负载，默认 chat-medium。' },
        requests: { type: 'integer', description: '性能评测请求数，默认 8。' },
        concurrency: { type: 'integer', description: '性能评测并发，默认 2。' },
        allowUnverifiedImage: { type: 'boolean', description: '允许使用未通过"扫描+签名"验证的量化镜像（会记入审计）。' },
        confirm: CONFIRM,
        headless: { type: 'boolean', description: '强制 headless（需确认的操作一律中止）。' },
        root: ROOT,
      },
    },
    ...passthrough(),
    async handler(args) {
      const { root, headless, confirm, ...rest } = args ?? {}
      return api.quantize(rest, { root, options: { headless }, confirm: confirm ?? null })
    },
  },
  {
    name: 'inference_quantization_get_report',
    description: '读取量化报告。不给 selector 时列出全部 run（含五要素完整性与去泄漏/源模型哈希证据）；给了则返回该 run 的完整报告（含逐档尝试记录与 QAT 计划）。',
    parameters: {
      type: 'object', additionalProperties: false,
      properties: {
        selector: { type: 'string', description: 'runId / runKey / 产出路径 / 源模型路径。' },
        runId: { type: 'string', description: '同上（显式 runId）。' },
        includeReport: { type: 'boolean', description: '是否内联完整报告，默认 true。' },
        root: ROOT,
      },
    },
    ...passthrough(),
    async handler(args) { const { root, ...rest } = args ?? {}; return api.getReport(rest, { root }) },
  },
  {
    name: 'inference_quantization_list',
    description: '列出量化矩阵（方法/精度/性能/平台/状态 五要素 + 完整性与去泄漏证据）。可按状态、源模型过滤。只读。',
    parameters: {
      type: 'object', additionalProperties: false,
      properties: {
        status: { type: 'string', enum: ['verified', 'failed', 'rolled-back', 'qat-plan-required'] },
        sourceModel: { type: 'string' },
        verifiedOnly: { type: 'boolean' },
        root: ROOT,
      },
    },
    ...passthrough(),
    async handler(args) { const { root, ...rest } = args ?? {}; return api.list(rest, { root }) },
  },
  {
    name: 'inference_quantization_rollback',
    description: '回退一次量化 run：标记为 rolled-back；可选删除产出目录（只删本插件登记的路径，且校验它不是源模型目录）。源模型不受影响。',
    parameters: {
      type: 'object', additionalProperties: false, required: ['selector'],
      properties: {
        selector: { type: 'string', description: 'runId / runKey / 产出路径 / 源模型路径。' },
        purgeOutput: { type: 'boolean', description: '是否删除产出目录（默认 false，保留产出）。' },
        confirm: CONFIRM,
        root: ROOT,
      },
    },
    ...passthrough(),
    async handler(args) { const { root, ...rest } = args ?? {}; return api.rollback(rest, { root }) },
  },
  {
    name: 'inference_quantization_qat_plan',
    description:
      '产出 QAT 计划（**不执行训练**）。计划含：数据（校准/测试集与去泄漏要求）、配置（起点/epoch/LR/显存）、'
      + '预期收益（含"这是经验估计不是承诺"的声明）、风险、回退路径、以及需每次确认的授权令牌。'
      + 'authorize=true 时记录沙箱降级授权（每次确认、不缓存），但仍然**不代跑**训练。',
    parameters: {
      type: 'object', additionalProperties: false,
      properties: {
        runId: { type: 'string', description: '为哪个 run 产出 QAT 计划；省略则用最近一次。' },
        selector: { type: 'string' },
        authorize: { type: 'boolean', description: '是否记录沙箱降级授权（默认 false，只出计划）。' },
        allowSandboxDegrade: { type: 'boolean', description: 'authorize=true 时必须同时为 true（QAT 落 R3，本机无 VM 沙箱）。' },
        confirm: CONFIRM,
        root: ROOT,
      },
    },
    ...passthrough(),
    async handler(args) { const { root, ...rest } = args ?? {}; return api.qat(rest, { root }) },
  },
  {
    name: 'inference_quantization_validate',
    description:
      '验收一个**已存在**的量化 checkpoint（不重新量化）：baseline = 量化前的原始模型，candidate = 已有产出目录。'
      + '复用与 quantize **完全相同**的实现与阈值语义 —— 精度对比走同一个 parityCheck（同一测试集、同一困惑度公式）、'
      + '测量成立性走 gate.assertMeasurementValid、精度门禁走 gate.evaluateAccuracyGate、阈值走同一条'
      + '"显式参数 > gates.yaml > 拒绝"链（**不新增默认值**）、产出形状走同一个 verifyQuantOutput'
      + '（没有 quantization_config 就是"没真的量化"）。gate="accuracy"（默认）只判精度维；'
      + 'gate="accuracy+perf" 与 quantize 完全同一次 evaluateGates 调用。'
      + '返回里的 scheme 一律**从候选产出自身读回**并给出 schemeRecognized。不写量化矩阵（不改既有工具行为），'
      + '验收记录落在 <runDir>/validate-existing/<id>.json 并进审计链。',
    parameters: {
      type: 'object', additionalProperties: false, required: ['candidate'],
      properties: {
        candidate: { type: 'string', description: '【必填】待验收的量化 checkpoint 目录（本地绝对路径，必须已含 config.json.quantization_config 与权重）。' },
        baseline: { type: 'string', description: '量化前的原始模型，作为精度基线（本地路径 / 模型 ID / env:VAR）。缺省时用 model / path。' },
        model: { type: 'string', description: '同 baseline（与 quantize 的入参命名保持一致）。' },
        path: { type: 'string', description: '同 baseline（显式本地路径）。' },
        max_accuracy_drop_pct: ACCURACY_DROP,
        min_perf_gain_pct: PERF_GAIN,
        gate: { type: 'string', enum: ['accuracy', 'accuracy+perf'], description: '判定档。accuracy（默认）只判精度维；accuracy+perf 需要两个阈值并与 quantize 同一次 evaluateGates。' },
        samples: { type: 'integer', description: '测试集采样条数（0 = 全量）。默认 0。' },
        calibrationSamples: { type: 'integer', description: '（仅用于准备测试集，不用于量化）默认 512。' },
        datasetRepo: { type: 'string', description: '测试集 repo，默认 Salesforce/wikitext。' },
        datasetConfig: { type: 'string', description: '测试集 config，默认 wikitext-2-raw-v1。' },
        testSamples: { type: 'integer', description: '测试集样本数，默认 400。' },
        allowUnverifiedImage: { type: 'boolean', description: '允许使用未通过"扫描+签名"验证的镜像（会记入审计）。' },
        confirm: CONFIRM,
        headless: { type: 'boolean', description: '强制 headless（需确认的操作一律中止）。' },
        root: ROOT,
      },
    },
    ...passthrough(),
    async handler(args) {
      const { root, headless, confirm, ...rest } = args ?? {}
      return api.validate(rest, { root, options: { headless }, confirm: confirm ?? null })
    },
  },
]

export const ADMIN_TOOL_SPECS = [
  {
    name: 'inference_quantization_dataset_prepare',
    description: '下载数据集并按内容哈希去重后切分校准/测试集，**宿主侧独立复核**去泄漏（交集必须为空）。幂等：同参数复用已有目录并复核。',
    parameters: {
      type: 'object', additionalProperties: false,
      properties: {
        repo: { type: 'string' }, config: { type: 'string' }, split: { type: 'string' },
        seed: { type: 'integer' }, calib: { type: 'integer' }, test: { type: 'integer' },
        force: { type: 'boolean', description: '强制重新生成（默认复用并复核）。' },
        root: ROOT,
      },
    },
    ...passthrough(),
    async handler(args) { const { root, ...rest } = args ?? {}; return api.datasetPrepare(rest, { root }) },
  },
  {
    name: 'inference_quantization_dataset_verify',
    description: '对一个数据集目录做去泄漏复核（宿主独立重算内容哈希并比对血缘）。只读。',
    parameters: { type: 'object', additionalProperties: false, properties: { dir: { type: 'string' }, root: ROOT } },
    ...passthrough(),
    async handler(args) { const { root, ...rest } = args ?? {}; return api.datasetVerify(rest, { root }) },
  },
  {
    name: 'inference_quantization_image_verify',
    description: '验证量化镜像：固定 base digest + 依赖版本清单 + 漏洞扫描 + 签名。缺任一项都会明确报告（不假装通过）。build=true 先构建再验证。',
    parameters: { type: 'object', additionalProperties: false, properties: { image: { type: 'string' }, build: { type: 'boolean' }, root: ROOT } },
    ...passthrough(),
    async handler(args) { const { root, ...rest } = args ?? {}; return api.imageVerify(rest, { root }) },
  },
  {
    name: 'inference_quantization_metrics',
    description: '指标：量化耗时、降级次数、门禁通过/失败/近上限、去泄漏检查次数与拒绝次数、压缩比。',
    parameters: { type: 'object', additionalProperties: false, properties: { root: ROOT } },
    ...passthrough(),
    async handler(a) { return api.metrics({ root: a?.root }) },
  },
  {
    name: 'inference_quantization_audit_verify',
    description: '校验审计日志哈希链完整性（检出任何删改）。',
    parameters: { type: 'object', additionalProperties: false, properties: { root: ROOT } },
    ...passthrough(),
    async handler(a) { return api.auditVerify({ root: a?.root }) },
  },
  {
    name: 'inference_quantization_doctor',
    description: '环境与能力自检：GPU/统一内存/容器/沙箱三层/量化方法可用性/镜像验证状态/testing-core 契约/策略来源/代理状态。只读。',
    parameters: { type: 'object', additionalProperties: false, properties: { deep: { type: 'boolean' }, root: ROOT } },
    ...passthrough(),
    async handler(a) { return api.doctor({ deep: a?.deep === true, root: a?.root }) },
  },
]

export const ALL_TOOL_SPECS = [...TOOL_SPECS, ...ADMIN_TOOL_SPECS]
export function findToolSpec(name) { return ALL_TOOL_SPECS.find((t) => t.name === name) ?? null }

/**
 * 把工具返回值规整成**无损 JSON**（树，不是 DAG）。
 *
 * ## 为什么适配层必须做这件事
 *
 * DSH 的工具契约要求返回值可被无损快照；它比 `JSON.stringify` **更严**：
 * 共享引用（同一个对象被两处引用）、数组里的 `undefined`、函数、非有限数
 * 都会被判为 `value is not lossless JSON`。
 *
 * 真机事故：`inference_serving_doctor` 的输出里 `environment.container.runtimes`
 * 与 `warnings[0].details` 是**共享引用** —— 于是工具一被调用就报
 * `tool "inference_serving_doctor" returned invalid output: value is not lossless JSON`。
 *
 * 这个缺陷**此前被另一个缺陷掩盖了**：那三个包的注册当时因为缺 `output.render`
 * 全部失败，工具**根本不存在**，自然没人调用、也没人发现输出不可序列化。
 * （"装了 ≠ 生效"的又一层：**注册成功 ≠ 调用成功**。）
 *
 * 修在适配层而不是逐个 doctor：这是**契约边界**的责任 —— 任何工具的返回值
 * 都必须过这一关，新增工具自动受保护。
 */
export function toLosslessJson(value) {
  if (value === undefined) return null
  try {
    // JSON 往返同时做了三件事：DAG → 树、丢掉 undefined、拒绝真循环/函数/BigInt
    return JSON.parse(JSON.stringify(value))
  } catch (e) {
    return {
      ok: false,
      code: 'E_OUTPUT_NOT_SERIALIZABLE',
      message: `工具返回值无法序列化为无损 JSON：${String(e?.message ?? e).slice(0, 200)}`,
      remediation: '这是本包输出构造的问题（常见于循环引用或 BigInt）；请报告该工具名与入参。',
    }
  }
}

export function toDshTool(spec) {
  return {
    name: spec.name,
    description: spec.description,
    parameters: spec.parameters,
    // DSH 的工具契约要求 output 同时给出 `schema` **与** `render`。
    // 只给 schema 会让 loader 抛 `must declare output { schema, render, … }`；
    // 而本包的注册被 try/catch 包着 → 失败被**吞掉** → 插件"挂上了"但工具一个都没有。
    // 真机实测：这条静默失败正是"装了 ≠ 生效"的典型（见 inference-contracts 的 C9）。
    output: spec.output ?? {
      schema: { type: 'object', additionalProperties: true },
      render(_args, value) { return [{ type: 'text', text: JSON.stringify(value ?? null, null, 2) }] },
    },
    async execute(args, exec) {
      if (exec?.signal?.aborted) {
        return { ok: false, code: 'E_CANCELLED', message: '调用已取消', remediation: '重新发起调用。' }
      }
      return toLosslessJson(await spec.handler(args ?? {}, exec))
    },
  }
}
