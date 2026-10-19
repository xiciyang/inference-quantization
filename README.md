# inference-quantization

**Quantize a model and prove it — or get an explicit failure.** Give it a checkpoint, an accuracy
tolerance and a performance target; it runs PTQ (default AWQ + INT4 via llmcompressor) inside a
container, then applies two independent gates:

- **Accuracy gate** — perplexity delta on a leakage-free split (content-hash intersection of
  calibration and test sets must be empty; there is no manual override).
- **Measurement-validity gate** — is the number even meaningful? Uniform output, no discriminating
  power, input-independent results, unhealthy weights and "never scored" all fail with
  `E_MEASUREMENT_INVALID` instead of being reported as "0% loss".

A downloaded-but-truncated model once quantized for 436 s and got signed off as "zero accuracy loss"
because its output was uniform and both perplexities were bit-identical. That is the incident these
two gates exist for. Source-model structure preflight (`E_SOURCE_MODEL_INVALID`) is the other half.

- Zero runtime dependencies, Node built-ins only
- Source model mounted read-only, hashes re-checked after quantization
- Bounded degradation: INT4 → INT8 → FP8 → W8A16 → (QAT plan only, no training)
- Every run writes a hash-chained audit trail

**Not a fit if** you just want `llmcompressor` or AutoAWQ recipes — those do the quantization; this
is the proof layer around it (gates, degradation, audit, rollback).

The real path needs a GPU, Docker and the quantizer image (`E_METHOD_UNAVAILABLE` is the explicit
failure when the image is missing — it never silently "passes").

## Install (DSH plugin / CLI)

```bash
dsh plugin --profile <name> add github:xiciyang/inference-quantization
# or, from this repo:
npm pack && npm install -g ./xiciyang-inference-quantization-0.1.0.tgz
```

给模型 + 精度容忍 + 性能目标，**量化它并证明它达标** —— 或者给你一个明确的失败结论。

```
输入 → 源模型哈希基线 + 结构预检（是完整模型吗）
  → 数据集准备（内容哈希去泄漏）
  → 选方法（默认 AWQ + INT4，走 llmcompressor）
  → 容器内 PTQ（源目录只读挂载）
  → 精度对比（同一测试集算困惑度退化）
  → 性能对比（经 MCP 让 inference-serving 起真实端点测吞吐）
  → 双门禁判定：**数值是否达标** ∧ **测量是否成立**
  → 通过 → 写入量化矩阵标记 verified
  → 失败 → 降级 INT4→INT8→FP8→W8A16→（QAT 计划）→ 回退
  → 源模型哈希复核（必须未变）
```

- **零运行时依赖**（只用 Node 内建模块）
- **精度门禁独立**：量化不能自己定阈值、自己签收精度
- **测量成立性门禁**：不只看"损失是多少"，还看"这个数成不成立"（见下）
- **源模型结构预检**：量化之前先确认这**是一个完整的模型**，不只是一堆能打开的文件
- **去泄漏是硬门禁**：校准集与测试集的内容哈希交集必须为空，**无人为放行通道**
- **源模型只读**：独立输出目录 + 量化前后哈希校验
- **降级有界**：最多 5 级，第 5 级只出 QAT 计划不执行训练

### 为什么"测量成立性"必须单独判

真机出过一次事故：一份**下载中断**的模型（338 个张量里 223 个整块为零，含 `model.norm.weight`）
被完整量化了 436 秒，并签收了"精度零损失"。原因是那份模型的输出恒为均匀分布，
于是 `perplexity` 恰好等于词表大小、基线与候选**逐位相同**、`drop = 0%` ——
数值在数学上没错，在语义上毫无意义。

现在有两层独立防线：

| 防线 | 回答的问题 | 失败码 |
|---|---|---|
| 源模型结构预检 | 内容**是不是一个完整模型**（截断 / 整块为零 / 层数不符 / 分片缺失） | `E_SOURCE_MODEL_INVALID` |
| 测量成立性门禁 | 这次**测量成不成立**（均匀输出 / 无区分力 / 与输入无关 / 权重不健康 / 没评到分） | `E_MEASUREMENT_INVALID` |

两条都是**不可降级**的失败 —— 换量化档位解决不了，所以直接停，不浪费 GPU。
`status=verified` 需要三条同时成立：**数值达标 ∧ 决策为 sign ∧ 测量成立**。

> 目录哈希（"内容有没有被改动"）与结构预检（"是不是完整模型"）是**两个不同的问题**。
> 只做前者，就会在一份**从未损坏过**的坏输入上一路绿灯。

---

## 安装

```bash
npm install -g @xiciyang/inference-quantization
# 或（本机 npm 官方源不可达时的推荐通道）
npm pack && npm install -g ./xiciyang-inference-quantization-0.1.0.tgz
```

作为 **DSH 插件**：`dsh plugin --profile web add <tgz>`，重启后生效。

**依赖**：`inference-serving`（部署验证，通过 **MCP** 调用）。本插件用下列顺序定位它：
`options.servingPath` → `~/.ai-infra/config/serving-path.txt` → 同级目录 `../inference-serving` → `node_modules/`。
找不到会明确报 `E_SERVING_UNAVAILABLE` 并列出探测过的路径。

---

## 快速上手

```bash
inference-quantization doctor                     # 环境 / 方法 / 镜像 / 沙箱 / 契约 自检
inference-quantization image verify --build       # 量化镜像验证（依赖清单 + 漏洞扫描 + 签名 + digest）
inference-quantization quantize /data/models/Qwen2.5-1.5B-Instruct \
    --max-accuracy-drop-pct 5 --min-perf-gain-pct 20
inference-quantization list --json                # 量化矩阵
inference-quantization report <runId>             # 完整报告
```

---

## 命令

| 命令 | 说明 |
|---|---|
| `quantize <model>` | 量化并过双门禁（幂等：同源+同方法+同切分 → 同一 runKey） |
| `report [selector]` | 读量化报告；不给 selector 则列全部 run |
| `list [--status S] [--verified]` | 列出量化矩阵（方法/精度/性能/平台/状态 五要素 + 完整性） |
| `rollback <selector> [--purge-output]` | 回退一次 run；可选删除产出目录（只删自己登记的，且校验不是源目录） |
| `qat [--authorize --allow-sandbox-degrade]` | 产出 QAT 计划（**不执行训练**） |
| `dataset <prepare\|verify\|list>` | 校准/测试集准备与去泄漏复核 |
| `image <verify\|build>` | 量化镜像验证 |
| `metrics` / `audit-verify` / `doctor` / `contract` | 指标 / 审计链 / 自检 / testing-core 契约 |

**退出码**：`0` 成功 · `1` 业务失败 · `2` 用法错误 · `3` 需要确认。

---

## MCP 工具

`inference_quantization_quantize` · `validate` · `get_report` · `list` · `rollback` · `qat_plan`
（另有运维用 `dataset_prepare` / `dataset_verify` / `image_verify` / `metrics` / `audit_verify` / `doctor`）

> **`validate`（新增，2026-09-29）**：验收一个**已存在**的量化 checkpoint（**不重新量化**）——
> `baseline` = 量化前的原始模型，`candidate` = 已有产出目录。
> 它**复用**与 `quantize` 完全相同的判据与阈值语义（同一个 `parityCheck` / `assertMeasurementValid` /
> `evaluateAccuracyGate` / `evaluateGates` / `verifyQuantOutput` / `dataset prepare`，
> 同一条"显式参数 > `gates.yaml` > 拒绝"的阈值链），**不改** `quantize` 的门禁语义，
> 也**不写**量化矩阵（不改既有工具行为）；验收记录落在 `<runDir>/validate-existing/<id>.json`。
> 解的是「外部产出的量化 checkpoint（例如 QAT 产出）无法被单独验收」这个跨插件缺口。
> 契约：`../inference-contracts/contracts/qat-verification-contract.md`。

stdio 与 streamable-http 两种传输：`node mcp/server.mjs [--http --port N]`。
DSH 侧走原生工具注册，两条通道共用 `src/tools.mjs` 的同一份规格与 `src/api.mjs` 的实现。

---

## 三道硬门禁

### 1. 去泄漏（`E_CALIB_LEAKAGE_DETECTED`）

判据是 **内容 sha256 交集为空**，不是"索引不重叠"。
同一段文本经常同时出现在数据集的 train / test 两个 split 里，按索引判断发现不了。

流程：容器内下载 → 按内容哈希去重 → 按种子切分 → 自检；
**宿主侧用同一套纯函数独立复核**（重算哈希、逐条比对血缘），不采信容器的自检结论。
血缘（来源 URL / revision / 种子 / 两段哈希清单 + listDigest）落盘，事后可重算。

**没有"确认后放行"这条路径** —— 泄漏会让精度数字失去意义。

### 2. 源模型只读（`E_SOURCE_MODEL_MODIFIED`）

量化前后对源目录算内容哈希（排序后的 `相对路径\0文件哈希` 再哈希，**不含 mtime/权限**）。
不一致即中止。第一道防线是输出目录安全检查：不得等于 / 位于 / 祖先于源目录。

### 3. 双门禁（精度 + 性能）

- **精度阈值必须调用方传入**（`--max-accuracy-drop-pct`）；未传则读 `~/.ai-infra/config/gates.yaml`；
  两者都无 → **拒绝执行**（`E_THRESHOLD_NOT_PROVIDED`）。插件内**没有**默认阈值 ——
  否则"量化不能自己签收精度"就名存实亡。
- 精度损失 ≥ 阈值的 90% → 判为"接近上限"，需回传令牌确认后才签收。
- 性能只在精度通过后才测（精度不及格测性能是浪费）。

---

## 降级链

`awq-w4a16(L1) → int8-w8a8(L2) → fp8-dynamic(L3) → w8a16(L4) → QAT 计划(L5) → 回退`

- **最多 5 级**；第 5 级只产出计划并停下等裁决（`E_QAT_REQUIRES_APPROVAL`），**不执行训练**；
- **不可降级的失败立即中止**：源模型被改、校准泄漏、阈值缺失、serving 不可用、镜像未验证 ——
  换档位没用，重试只是假装在努力；
- 非最终档位用**抽样粗筛**（≥200 条，阈值放宽 2×）省成本，**最终签收必须完整评测**；
- 状态机最长路径 15 步（logicprobe 穷举实测）→ 超时预算按 16 留余量。

### 为什么是 llmcompressor 而不是 autoawq

`autoawq` 在 PyPI 无 wheel，其内核包 `autoawq-kernels` 只有 `x86_64`/`win_amd64` wheel ——
本机是 **aarch64**，装不上。而 `llmcompressor` 是纯 Python wheel，其 AWQ 变换器**只依赖 torch**。
语义上仍是 "AWQ + INT4"。**已真机验证**：产出能被 vLLM 加载并做真实前向
（详见 `docs/step0/PROJECT_CONTEXT.md` §11）。

---

## 配置

### `~/.ai-infra/config/gates.yaml`（首次运行生成**空模板**）

```yaml
version: 1
max_accuracy_drop_pct: null    # null = 未设置 → 调用时必须显式传参
min_perf_gain_pct: null
near_limit_factor: 0.9
coarse_screen:
  min_samples: 200
  relax_factor: 2.0
```

风险分级 / 网络策略 / headless 三份**沿用 inference-serving 已固化的文件**。

### 插件配置（DSH 行内 `config`）

`root` / `quantImage` / `outputRoot` / `calibrationSamples` / `calibrationSeed` /
`datasetRepo` / `datasetConfig` / `maxDegradeSteps` / `quantTimeoutMs` / `evalTimeoutMs` /
`servingPath` / `headless` / `modelRoots` / `containerRuntime` / `exposeAdminTools` / `logLevel`。

未知键、类型不符、`maxDegradeSteps < 1` 都会**加载期响亮失败**。

---

## 权限与安全

### 能力清单（`manifest.yaml` 是白名单）

| capability | risk | resources |
|---|---|---|
| `quantize` | write | `model:read` `gpu:write` `disk:write` `state:write` `network:egress` |
| `get_report` / `list` | read | `state:read` |
| `rollback` | write | `disk:write` `state:write` |
| `qat_plan` | write | `state:write` `model:read` |
| `dataset_prepare` / `image_verify` / `image_build` | write | `network:egress` `disk:write` `state:write` |

`not_provided` 显式声明**设计上不做**的事：`qat_execute`（训练，R3）、`modify_source_model`、
`disable_leakage_gate`、`sudo_exec`、`install_driver`。

### 沙箱

R0 只读不问沙箱；R2（本插件全部写操作）走**容器级**；
R3–R5（含 QAT 训练）需 VM 级 → **本机没有 → 一律拒绝**。
沙箱不可用不静默降级（`E_SANDBOX_UNAVAILABLE`）。

### 量化镜像

`image verify` 检查四件事并**明确报告缺哪一项**（不假装通过）：
固定 base digest（`docker/base-digest.txt`）、依赖版本清单、漏洞扫描（trivy/grype/docker scout）、
签名（cosign/docker trust）。
未通过验证的镜像在量化时会被拒绝，除非显式传 `allow_unverified_image`（记入审计）。

量化镜像**必须独立于推理镜像**：llmcompressor 要求 `compressed-tensors==0.19.0`，
而推理镜像钉 `0.17.0` —— 装进推理镜像会破坏 vLLM 的依赖。

---

## 共享状态

```
~/.ai-infra/state/quantization-matrix.yaml     量化矩阵（五要素 + 去泄漏/哈希证据）
~/.ai-infra/state/datasets/                    数据集 + 血缘
~/.ai-infra/state/images/                      镜像元数据
~/.ai-infra/quantized/                         量化产出（独立于模型根目录）
~/.ai-infra/logs/audit/inference-quantization/ 审计（{date}.jsonl，只追加 + 哈希链）
~/.ai-infra/reports/inference-quantization/    完整报告
```

---

## 内置共享库（`@internal`）

题面把精度/性能验证划给共享库 `@ai-infra/testing-core`，但**该库在本机不存在**（npm/node_modules/工作区三处均无）。
按授权，本包内置**最小实现**并标注 `@internal`：

- `src/testing/parity.mjs` —— `parityCheck()`：按 token 加权的 mean NLL / perplexity
- `src/testing/perf.mjs` —— `perfValidation()`：走真实服务端点测 TTFT / decodeTps / throughput
- 契约版本 `1.0.0`，用 `inference-quantization contract` 可打印
- 一旦真库发布，替换 provider 即可，调用方零改动（迁移脚本见 `scripts/migrate-to-testing-core.mjs`）

---

## 开发

```bash
npm test          # 单元 + 集成（不需要 GPU/Docker）
npm run check     # 包规范 + 量化专项自检
npm run demo      # 真机端到端演示（需要 GPU + Docker + 量化镜像）
```

**量化方法插件化**：新增一种方法 = 在 `src/backends/` 加一个声明 + 在 `docker/scripts/quantize.py` 加一条 recipe，
核心流程零改动。契约见 `skills/quantization-method-authoring/SKILL.md`。

## License

MIT

## 兼容性

| 项 | 范围 |
|---|---|
| Node.js | >= 20.11 |
| DSH | `engines.dsh: ">=0.2.0 <0.3.0"`，已在 **0.2.0-rc.2** 上验证 |
| 操作系统 | Linux / macOS（x64 / arm64）；量化本身需要容器运行时 |
| 运行时依赖 | 零（只用 Node 内建模块）；量化在容器内完成 |
| 后端 | Docker / Podman + `llmcompressor` 镜像 |

测试分层：`npm run test:unit` 在**无 GPU** 的 CI（GitHub 托管 runner）上运行；
`npm run test:integration` 需要 GPU + 容器 + nvidia runtime —— 条件不满足时显式报
`E_METHOD_UNAVAILABLE`，不会静默跳过。本机实测 146/146。

跨包契约套件（`inference-contracts`）属于 monorepo 的内部闸门：单独 clone 本仓库时
`npm run test:contract` 会明确跳过，不会伪装成通过。

## 许可

MIT，见 [LICENSE](./LICENSE)。
