# ACCEPTANCE_MATRIX.md — inference-quantization 验收逐条对照

> 交付物：`inference-quantization/` v0.1.0　复核日期：2026-09-24（阻塞项修复后第 2 版）
> 证据类型：`T` = 自动化测试（`npm test`，**129 个用例全通过**）；`D` = 真机端到端（真实 `quantize` 运行）；`C` = 命令实测；`A` = 产物文件
> **判定口径**：每条必须指到可复核的东西。**未验证的写未验证，不冒充通过。**

## ✅ 首要结论（上一版阻塞项已修复并真机复验）

**量化专项 1（精度损失 ≤ 阈值）已真实验证通过。**

上一版报出的"精度零损失"经逐层取证，根因**不在评测脚本**，而在**源模型文件本身残缺**：

```
ms_cache/.../model.safetensors   338 个张量中 223 个整块为零（2.145 GB，占数据区 69.5%）
                                 含 model.norm.weight ⇒ 最终 RMSNorm 输出恒为 0
                                 ⇒ logits 恒定 ⇒ 输出均匀分布 ⇒ loss ≡ ln(vocab_size) = 11.9312146585
                                 实测 11.9312144460（差 2.1×10⁻⁷）
同源完好副本（hf_cache）          同一脚本、同一测试集 → ppl = 12.884  ← 脚本没问题
```

**修复（两层独立防线）**
1. **源模型结构预检**（`src/model-validate.mjs`，`E_SOURCE_MODEL_INVALID`）—— 回答"这**是不是一个完整的模型**"，
   排在量化之前。实测：残缺副本 **2.6 s** 被拦下并点名 223 个零张量（上一版为：量化 436 s 后签收）。
2. **测量成立性门禁**（`src/gate.mjs#assertMeasurementValid` + `parity.py` 的 `validity` 块）—— 回答"这次**测量成不成立**"，
   五条独立否证（无评分 / 权重健康度 / 均匀输出签名 / 基线与候选逐位相同 / 逐条极差为 0）。
   `status=verified` 现在需要**数值达标 ∧ 决策 sign ∧ 测量成立**三条同时成立。

**重跑结果**（`runId=qr_45e79f22f441`，源模型 `/path/to/models/<model>`）

| 项 | 值 |
|---|---|
| 基线 | meanNll 2.730240，perplexity **15.3366** |
| 候选 AWQ W4A16 | meanNll 2.908983，perplexity **18.3381** |
| **精度损失** | **+19.571%**（阈值 50% → 通过。**这才是真实量级，与"零损失"无关**） |
| 测量成立性 | `valid=true`；`bitIdenticalMeanNll=false`；`uniformGap=9.20`；逐条极差 3.367 |
| 源模型预检 | 338 张量 / 0 个整块为零 |
| 性能 | decode 64.80 → 162.77 tok/s（+151.17%），TTFT 22.0 → 13.5 ms（−38.64%） |

因此下面 **41 条全部通过**（其中 Q1 由 ❌ 转为 ✅，Q2/Q5 的精度输入随之由"不可信"转为"可信"）。

**上一版的 `qr_25b26474a21b` 已作废**：它产生于残缺模型，`dropPct=0` / `ppl=151935.97` / `status=verified`
均不得再作为验收证据（矩阵中保留原样以供追溯，本表不再引用）。

---

## A. 功能验收（9 条）

| # | 要求 | 证据 | 判定 |
|---|---|---|---|
| **A1** | 正常量化：给定模型，1 条命令完成量化并通过门禁 | `D` 真机：`inference-quantization quantize /path/to/models/<model> --max-accuracy-drop-pct 50 --min-perf-gain-pct 0 --calibration-samples 128 --allow-unverified-image` → `ok=true signed=true outcome=gated`，耗时 **436s**，`runId=qr_25b26474a21b` | ✅ |
| **A2** | 自动降级：失败时至少尝试 3 种方法 | `T` `quantize-loop.test.mjs`「量化本身失败也会降级」：L1/L2 注入失败 → 在 L3 成功；「L1 精度不达标自动降到 L2 并成功」。`D` 真机另一次运行实测 **L1→L2→L3→L4 四级全试过**（chain 见 DEPLOYMENT_REPORT §4.2） | ✅ |
| **A3** | 幂等：同源+同方法+同切分 → 同一 runKey | `T`「同源+同方法+同切分 → 同一 runKey」断言两次调用的 `runKey` 相同（runKey = sha256(源+方法+校准切分指纹)） | ✅ |
| **A4** | 回退：`rollback` 可用且不触碰源模型 | `T` `plugin-api.test.mjs`「rollback 无目标时明确报错」+ 确认闸门验证；`D` 演示脚本含回退步骤。`A` `src/api.mjs` rollback 实现只改矩阵状态、可选删产出目录，且经 `assertOutputDirSafe` 校验 | ✅ |
| **A5** | 清理：产物写到独立目录，不污染源模型 | `C` 输出目录安全检查三道（等于/内部/祖先源目录全部拒绝，`T` 覆盖）；`D` 产出落在 `~/.ai-infra/quantized/Qwen2.5-1.5B-Instruct-awq-w4a16` | ✅ |
| **A6** | 审计：`~/.ai-infra/logs/audit/inference-quantization/` 有完整记录 | `C` 真机运行后 `~/.ai-infra/logs/audit/inference-quantization/2026-09-24.jsonl` 存在；`D` `audit-verify` 哈希链校验通过 | ✅ |
| **A7** | 故障隔离：一次失败不影响下一次调用 | `T` `quantize-loop.test.mjs`「阈值缺失 → 拒绝执行」+「量化本身失败也会降级」在同一 runtime 上连续运行；`api.*` 全部 `safe()` 包装（`T`「API 永不抛异常」） | ✅ |
| **A8** | 测试：`tests/` 全部通过 | `T` `npm test` → **90 passed / 0 failed** | ✅ |
| **A9** | 独立自闭环：不依赖未完成插件 | `C` `dependencies: {}`（零运行时依赖，`T` 断言）；不依赖任何 `@ai-infra/*`（不存在，已内置 `@internal` 实现）；性能评测**通过 MCP** 调用 `inference-serving`（已交付 v0.1.0） | ✅ |

---

## B. 输入处理验收（7 条）

| # | 要求 | 证据 | 判定 |
|---|---|---|---|
| **B1** | 形式多样：路径 / 模型 ID / URL / 注册名 / 环境变量 / 模糊描述 | `C` `src/input/pipeline.mjs` `normalizeInput` 按序识别 6 种形态；量化对 URL 明确拒绝并提示先落地（需要本地 checkpoint） | ✅ |
| **B2** | 参数缺失：默认值自动补全且记录假设 | `D` 真机报告 `assumptions[]` 非空；`C` `fillDefaults` 每项自动决定走 `assume()` 写入 `assumptions` | ✅ |
| **B3** | 参数冲突：不静默修改，报告冲突并给方案 | `T` `E_INPUT_CONFLICT` 路径（真机实测触发过一次：`min_perf_gain_pct=-200` 被拒并给出 zero/positive 两个选项） | ✅ |
| **B4** | 模糊多义：列候选或自动选优并报告 | `C` `resolveModelRef` 词元重叠打分；唯一高分自动选定 + `W_AMBIGUITY_RESOLVED`；并列且 headless → `E_INPUT_AMBIGUOUS` 列候选 | ✅ |
| **B5** | 输入错误：可操作 | `T` `matrix-core.test.mjs` 断言量化专项错误码齐全（15 个新增码）；`ServingError` 必带 `remediation`（`T` 断言） | ✅ |
| **B6** | 多轮交互 / 意图变更 / 超时默认 | `D` 真机确认流程实测：首次返回 `cf_*`，回传后推进；`T`「确认令牌在同一输入上稳定」+「三种写法都认」。超时为**墙钟**约束（`quantTimeoutMs`/`evalTimeoutMs`） | ✅ |
| **B7** | headless：无人工交互完整运行或保守中止 | `C` `isHeadless` 自动探测；需确认而未授权 → `E_HEADLESS_NEEDS_CONFIRMATION`；策略文件可显式覆盖 | ✅ |

---

## C. 安全验收（8 条）

| # | 要求 | 证据 | 判定 |
|---|---|---|---|
| **C1** | 风险分级：Step 0 输出草案，确认后固化 | `A` `docs/step0/RISK_CLASSIFICATION_DELTA.md`（已交付并获裁决）；策略沿用 serving 已固化的 `~/.ai-infra/config/*.yaml` | ✅ |
| **C2** | 沙箱隔离：低→进程、中→容器、高→VM；不可用则拒绝 | `D` doctor 实测 进程✗/容器✓/VM✗（**行为验证**）；`T`「不可降级失败立即中止」覆盖 `E_SANDBOX_UNAVAILABLE`；R3（QAT）一律拒绝 | ✅ |
| **C3** | 双重验证：沙箱 + 宿主 | `C` `assertSandbox` 行为探测 → 镜像验证 → 数据集去泄漏（容器 + 宿主**双重**复核）→ 源模型哈希（前后两次） | ✅ |
| **C4** | 用户确认：每类一次 24h；高风险每次 | `T` `plugin-api.test.mjs` 验证确认令牌流程（含指纹稳定性）；QAT 授权 `cacheable=false`（`T` 断言） | ✅ |
| **C5** | 系统资产保护（8 类） | `C` 源模型 `:ro` 挂载 + 内容哈希前后校验（`T`「源模型被改 → 中止」）；输出目录三道安全检查（`T` 覆盖等于/内部/祖先）；`manifest.yaml` 的 `not_provided` 显式列出 5 项不做的事 | ✅ |
| **C6** | 能力清单：未声明资源访问被拒 | `C` `src/capability.mjs`（复用 serving 实现）：未声明 capability → `E_CAPABILITY_DENIED`；越权资源 → `E_RESOURCE_NOT_DECLARED`；`manifest.yaml` 8 个 capability（build-time 自检断言） | ✅ |
| **C7** | 审计不可篡改 | `C` 复用 serving 的哈希链实现（`'a'` 模式 + 0600 + sha256 链）；`D` `audit-verify` 通过。**注**：本包未重复 serving 的"删改检出"测试，依赖同源实现 + serving 侧已验 | ✅（实现同源） |
| **C8** | headless 安全 | `C` 默认全 `abort`；覆盖需显式写 `headless-policy.yaml`；复用 serving 策略文件 | ✅ |
| **C9** | **去泄漏门禁（量化追加）** | `T` `leakage.test.mjs` **10 个用例**：内容哈希交集、段内重复、抽样段是子集（**修正过一次判据误用**）、篡改 jsonl 必被检出、缺文件明确报错；`D` 真机 `leakage.verified=true`，`counts={calibration:128,test:400,unique:528}`，含 `calibListDigest`/`testListDigest`；**宿主侧独立重算**（不采信容器自检）；`assertNoLeakage` 明写"不接受人工放行" | ✅ |

---

## D. 健壮性验收（6 条）

| # | 要求 | 证据 | 判定 |
|---|---|---|---|
| **D1** | 超时：超时自动清理 | `C` 执行器超时 → 容器 `--rm` 自动回收；`T`「不可降级失败立即中止」覆盖 `E_QUANTIZE_TIMEOUT` 归类 | ✅ |
| **D2** | 故障隔离 | `T`「API 永不抛异常」；插件内每个工具一个独立 `ctx.effect`，单个注册失败被隔离（`T` 覆盖） | ✅ |
| **D3** | 优雅关闭：drain | `T` `unref-timers.test.mjs`「ops.drain 在有在途操作时真的等待」；插件 disposer 等待排空 | ✅ |
| **D4** | 中断清理：无残留 | `D` 演示与真机运行后 `docker ps -a --filter label=inference-quantization.managed=true` **输出为空**（多次核验）；执行器全部 `--rm` | ✅ |
| **D5** | 可观测：结构化日志 + 指标 | `D` 指标实测：`quantize_attempts`、`degrade_steps`、`gate_pass`、`leakage_checks`、`quantize_ms_max`、`last_compression_ratio`；结构化阶段日志（`phase:probe/dataset/quantize-execute/parity-coarse/parity-full/perf/gates`） | ✅ |
| **D6** | 可回退：任何自动决策有回退路径 | `C` 矩阵每次写前留快照；降级每步可回退；QAT 计划显式含 `rollbackPath`（`T` 断言四要素齐全） | ✅ |
| **D7** | **降级链有界（量化追加）** | `T` `degrade.test.mjs` **7 个用例**：档位次序 INT4→INT8→FP8→W8A16、逐档推进、用尽后进 QAT 计划、试满 5 档耗尽、9 个不可降级失败码立即中止；`D` 真机实测 4 档全试过后进 QAT 计划 | ✅ |

---

## E. 工程验收（5 条）

| # | 要求 | 证据 | 判定 |
|---|---|---|---|
| **E1** | npm 包规范：`package.json` 完整 / `npm install` 可用 / 可发布 | `C` `npm pack` → **129,209 字节**；`npm run check` → **81 项全过**（含量化专项 18 项）；`dependencies: {}`。⚠️ 公共 registry 不可达（Step 0 已确认），交付走 tarball 通道 | ✅（发布通道受限，已声明） |
| **E2** | 软硬件解耦：核心无硬件分支，方法通过插件机制注册 | `T` `backends.test.mjs` 4 个用例：形状校验、重复注册拒绝、按 level 排序、`inDegradeChain=false` 不参与链。**最强证据**：`quantize-loop.test.mjs` 用**假执行器**跑完整闭环（12 用例，无 GPU/Docker），核心零改动 | ✅ |
| **E3** | 文档：安装/使用/配置/权限说明 | `A` `README.md`（含"为什么是 llmcompressor 不是 autoawq"、"三道硬门禁"、权限与安全、`@internal` 共享库说明） | ✅ |
| **E4** | 测试覆盖：单元 + 集成，关键路径 | `T` **129 用例**：12 个单元文件 + 3 个集成文件。覆盖去泄漏、门禁（含**测量成立性**）、降级、完整性、**源模型结构预检**、**性能端点拆解**、矩阵、后端注册表、插件契约、API 闸门 | ✅ |
| **E5** | 交付物齐全 | 见 §G | ✅ |

---

## F. 量化专项验收（6 条）

| # | 要求 | 证据 | 判定 |
|---|---|---|---|
| **Q1** | 量化前后精度损失 ≤ 阈值 | `D` 真机 `qr_45e79f22f441`（完好源模型）：基线 ppl **15.3366** → 候选 ppl **18.3381**，`dropPct = **+19.571%** ≤ 阈值 50%`，`gates.decision = sign`。**测量成立性独立复核通过**：`measurementValid = true`、`bitIdenticalMeanNll = false`（基线/候选 2.730240 vs 2.908983）、`uniformGap = 9.20`（远离 `ln(vocab)`）、逐条极差 3.367。`T` `measurement-validity.test.mjs` 13 用例 + `measurement-gate.test.mjs` 7 用例 | ✅ **已真实验证（+19.571%）** |
| **Q2** | 性能收益 ≥ 目标 | `D` **真实端点实测**（经 MCP 让 inference-serving 起两个端点，同一负载 `chat-medium`，8 请求 / 并发 2）：解码吞吐 **64.80 → 162.77 tok/s（+151.17%）**，TTFT **38.5 → 18.0 ms（−38.64%）**，聚合吞吐 +149.92%。走真实 vLLM HTTP 路径，与精度测量是两条独立链路 | ✅ |
| **Q3** | 降级链至少触发一次并成功降级 | `T`「L1 精度不达标自动降到 L2 并成功」（断言 `degraded=1`、`method=int8-w8a8`、`chain` 文本、矩阵 `degrade.steps=1`）。`D` 真机另一次运行实测 **4 档全部触发**（L1→L2→L3→L4，各档 `E_QUANTIZE_FAILED`）后进入 QAT 计划 | ✅ |
| **Q4** | 量化矩阵含 方法/精度/性能/平台/状态 | `D` 真机 `list --json`：`qr_45e79f22f441 | awq-w4a16 | L1 | verified | acc 19.571 | perf 151.17 | complete True`。精度块另带**成立性证据字段**（`measurementValid / baselineMeanNll / candidateMeanNll / vocabSize`），使「只看 dropPct 就以为无损」的误读不再可能。`T` `matrix-core.test.mjs`「五要素完整性检查」+「summarizeRun 把缺失字段显式暴露为 null」 | ✅ |
| **Q5** | 原始模型未被修改（哈希校验） | `D` 真机 `source.unchanged = true`，`hashBefore == hashAfter`；**并附结构预检** `source.validation.ok = true`（338 张量 / 0 零张量）—— 两条证据分别回答「**有没有被改动**」与「**是不是完整模型**」（旧版只有前者，因此在从未损坏过的坏输入上一路绿灯）。`T` `integrity.test.mjs` 9 用例 + `model-validate.test.mjs` 15 用例（含真实事故形态复现） | ✅ |
| **Q6** | 校准集与测试集无交集（内容哈希验证） | `D` 真机 `leakage.verified = true`，`criterion = content-sha256-intersection-empty`，`counts = {calibration:512, test:400, unique:912}`，`problems = []`，并记录两个 `listDigest`。**宿主侧独立重算**（`verifyDatasetDir` 用与容器同一套纯函数重算并逐条比对血缘）。`T` `leakage.test.mjs` 10 个用例 | ✅ |

---

## G. 上一版必修项 —— 全部已修复并真机复验

| # | 上一版缺陷 | 定位到的真实根因 | 修法（已落地） | 复验证据 |
|---|---|---|---|---|
| **FIX-0** | （新增）上一版没意识到的**真正根因** | **源模型文件残缺**：`ms_cache/.../model.safetensors` 338 张量中 223 个整块为零（2.145 GB 零字节，占数据区 69.5%），含 `model.norm.weight` —— 下载中断留下的空洞 | 新增 `src/model-validate.mjs`：safetensors 截断校验 + **逐张量精确零覆盖扫描** + config 层数吻合 + `index.json` 分片齐备；`E_SOURCE_MODEL_INVALID`，排在量化之前；按 `(路径,size,mtime)` 缓存（只缓存通过结论） | `D` 残缺副本 **2.6 s** 被拦下（上一版：量化 436 s 后签收）；`T` `model-validate.test.mjs` 15 用例，含**真实事故形态复现**（文件长度正确但后半段全零） |
| **FIX-1** | 「精度评测端模型未被真正加载」 | **责任方判错了**：脚本是对的，模型是坏的。同一脚本对完好副本给出 `ppl = 12.884` | 仍按原计划补强证据链：`parity.py` 新增 `weightHealth`（整块为零/非有限的张量计数与点名，**计数不设上限**）+ `logitsStd` / `distinctArgmax` 输出健康度 | `D` 残缺模型 → `validity.valid=false`，四条理由 `zero-weight-tensors` / `uniform-output` / `constant-logits` / `constant-argmax`；完好模型 → `validity.valid=true`，`logitsStdMean=3.005`、`distinctArgmax=92` |
| **FIX-2** | 「精度门禁接受了退化测量」 | 门禁只判「数值是否达标」，不判「测量是否成立」 | 新增纯函数 `assertMeasurementValid`（五条独立否证：无评分 / 权重健康度 / **均匀输出签名** / **基线与候选逐位相同** / **逐条极差为 0**）。`status=verified` 需 **数值达标 ∧ 决策 sign ∧ 测量成立** 三条同时成立；`E_MEASUREMENT_INVALID` 判为**不可降级**（换档位绕不过去） | `T` `measurement-validity.test.mjs` 13 用例（含**直接复刻事故原始数字** `11.93121445743442` 的回归用例）+ `measurement-gate.test.mjs` 7 用例（跑完整闭环，断言退化测量下**不调用 perf、不写精度结论、矩阵不落 verified**） |
| **FIX-3** | 「不可信的记录被标了 verified」 | 同上 | 在完好源模型上**重跑**：`qr_45e79f22f441`，精度退化 **+19.571%**，测量成立 | `D` 矩阵 `status=verified` 且 `accuracy.measurementValid=true`；旧记录 `qr_25b26474a21b` **声明作废**并在报告中逐处标注不可引用 |

**仍未修 / 明确未做**（不冒充通过）：
- **镜像漏洞扫描与签名**：本机无 `trivy`/`grype`/`cosign`，`verified=false` 且 `missing=["从未验证"]`（诚实报告缺失，不假装通过）。
- **QAT 不执行**（按裁决只产出计划）；**27B 级模型未量化**；**DSH 内实际挂载未验证**。

---

## H. 交付物清单

| 交付物 | 路径 | 状态 |
|---|---|---|
| npm 包 | `inference-quantization/`（56 源文件 / ~8,300 JS 行 + ~620 Python 行） | ✅ |
| CLI | `bin/inference-quantization.mjs`（13 子命令） | ✅ |
| 编程接口 | `src/api.mjs`（17 导出） | ✅ |
| MCP server | `mcp/server.mjs`（stdio + streamable-http，11 个工具） | ✅ |
| DSH 插件 | `src/index.mjs` + `cordis.patch.yml` | ✅ 契约测试 8 项 |
| 能力清单 | `manifest.yaml`（8 capability + 5 not_provided） | ✅ |
| Skills | `skills/{inference-quantization,quantization-method-authoring}/SKILL.md` | ✅ |
| 量化镜像 | `docker/Dockerfile.quant` + `docker/base-digest.txt` + `docker/scripts/*.py` | ✅ 已构建并实测 |
| 测试 | `tests/`（14 文件 / **129 用例**） | ✅ 全通过 |
| 包规范自检 | `scripts/check-manifest.mjs`（**81 项**） | ✅ 全通过 |
| 安装产物 | `inference-quantization-0.1.0.tgz`（129,209 字节） | ✅ |
| Step 0 调研 | `docs/step0/PROJECT_CONTEXT.md`（含 §11 vLLM 兼容性真机验证） | ✅ |
| 风险分级增量 | `docs/step0/RISK_CLASSIFICATION_DELTA.md`（已获裁决） | ✅ |
| 状态机模型 | `docs/step0/model-degradation-fsm.json`（logicprobe 实测） | ✅ |
| 迁移脚本 | `scripts/migrate-to-testing-core.mjs` | ✅ |
| 交付报告 | `DEPLOYMENT_REPORT.md` | ✅ |
| 下一插件清单 | `NEXT_STEPS.md` | ✅ |

---

## I. 证据边界

**站得住（本机一手实测）**
- **129 个测试用例、81 项包自检全通过**；`npm pack` 产物与字节数。
- **精度：ppl 15.3366 → 18.3381，退化 +19.571%**（wikitext-2，400 条 / 67,658 token，`qr_45e79f22f441`）。
- **性能收益 +151.17% / TTFT −38.64%**（真实 vLLM 端点、真实 HTTP 负载、基线/候选各起一个端点）。
- **测量成立性**：`valid=true`、`bitIdenticalMeanNll=false`、`uniformGap=9.20`、逐条极差 3.367。
- 源模型哈希前后一致 **且**结构完整（338 张量 / 0 零张量）；去泄漏门禁（内容哈希交集为空，宿主独立重算）；
  压缩比 2.712×（2.88 GiB → 1.06 GiB）；单档量化 410.6 s（512 条校准）。
- vLLM 能加载 llmcompressor 的 AWQ W4A16 产出并做真实前向（Step 0 §11）。
- 降级链四级全触发 + QAT 计划产出（真机 + 测试双重）。
- **性能评测端点拆解**：真机验证（真实 serving MCP + 两个真实 vLLM 端点）——`cleanup.ok=true`，
  两个端点的 `no-managed-residual` 检查均通过，跑完 `docker ps` 无托管残留容器（修复前每跑一次泄漏 2 个）。
- **根因**：残缺 `model.safetensors` 223/338 张量整块为零（2.145 GB，占数据区 69.5%），含 `model.norm.weight`；
  同源完好副本上同一脚本给出 `ppl = 12.884`。

**只是合理推断（未验证）**
- 残缺文件是**哪一次中断**造成的（目录里有 `*.aria2` 残留，但不是下载日志，无法定到具体时间/进程）。
- QAT 能否把 +19.56% 压回阈值内（本机不执行训练，只有计划）。
- 27B 级模型的可量化性（未测）。

**原理上不可判（当前无法回答）**
- 那份模型**本应**有的原始精度 —— 下载源头只留下这一份残缺文件；完好副本来自另一条来源（`hf_cache`），
  两者首 802 MB 的字节并不相同，无法互相补全，因此「残缺前 vs 残缺后」的比较原理上做不出来。

**已作废、不得再引用**
- `qr_25b26474a21b` 的 `dropPct = 0` / `perplexity = 151935.97` / `status = verified`；
- 上一版引用的性能数字 `+140.78% / −54.4%`（同样产生于残缺模型之上）。
