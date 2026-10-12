---
name: inference-quantization
description: 当需要把模型量化并证明它达标时使用（给定精度容忍与性能目标，自动选方法→PTQ 校准→精度/性能双门禁→签收或降级回退）。也用于排查"量化后精度掉太多""性能没提升""校准集是否泄漏"等问题。
---

# inference-quantization 使用与排障

给模型 + 精度容忍 + 性能目标 → 拿到一个**被证明达标**的量化模型，或一个明确的失败结论。

## 三步上手

```bash
inference-quantization doctor                                   # 环境/方法/镜像/契约自检
inference-quantization image verify --build                     # 量化镜像验证（扫描+签名+依赖+可复现）
inference-quantization quantize /data/models/Qwen2.5-1.5B-Instruct \
    --max-accuracy-drop-pct 5 --min-perf-gain-pct 20
```

## 验收一个**已经存在**的量化 checkpoint（`validate`）

当量化产出不是**本次 `quantize` 造的**（例如由 `training-qat` 的 QAT 产出、或别处搬来的），
不要再用 `quantize` 去"验收"它 —— 那会**二次量化**。用 `validate`：

```bash
inference-quantization validate <candidate目录> --baseline <量化前的原始模型> --max-accuracy-drop-pct 2
```

- `gate=accuracy`（默认）：只判精度维（QAT 的职责是**精度恢复**，不是性能）；
- `gate=accuracy+perf`：与 `quantize` 完全同一次 `evaluateGates` 调用（需要两个阈值）。
- 判据**全部复用** `quantize` 的那一套（`parityCheck` / `assertMeasurementValid` /
  `evaluateAccuracyGate` / `verifyQuantOutput` / `dataset prepare`），阈值走同一条
  "显式参数 > `gates.yaml` > 拒绝"链；`quantize` 的门禁语义**未改动**。
- 返回带 `scheme` 与 `schemeRecognized` —— scheme 一律**从候选产出自身读回**，不由调用方声明。
- 不写量化矩阵（不改既有 `list` / `get_report` 行为）；验收记录在
  `<runDir>/validate-existing/<id>.json`。

## 为什么必须传阈值

题面要求"**精度门禁独立：量化不能自己签收精度**"。所以本插件**不猜默认阈值**：
不传 `--max-accuracy-drop-pct` / `--min-perf-gain-pct`，且 `~/.ai-infra/config/gates.yaml` 里也没设 →
直接拒绝（`E_THRESHOLD_NOT_PROVIDED`）。这不是麻烦，是防止"量化自定标准、自己判达标"。

## 闭环做了什么

```
输入 → 源模型哈希基线 → 数据集准备（内容哈希去泄漏）→ 选方法（默认 AWQ+INT4）
  → 容器内 PTQ（源目录只读挂载）
  → 精度对比（同一测试集算困惑度退化）
  → 性能对比（通过 MCP 让 inference-serving 起**真实端点**测吞吐）
  → 双门禁
  → 通过 → 写量化矩阵标记 verified
  → 失败 → 降级 INT4→INT8→FP8→W8A16→（QAT 计划）→ 回退
  → 源模型哈希复核（必须未变）
```

## 降级链

`awq-w4a16(L1) → int8-w8a8(L2) → fp8-dynamic(L3) → w8a16(L4) → QAT 计划(L5) → 回退`

- **最多 5 级**；第 5 级**只产出计划，不执行训练**；
- **不可降级的失败立即中止**（源模型被改、校准泄漏、阈值缺失、serving 不可用、镜像未验证）——
  这些换档位也没用；
- 非最终档位用**抽样粗筛**（≥200 条，阈值放宽 2×）省钱，但**最终签收必须完整评测**。

## 三道硬门禁

| 门禁 | 判据 | 失败行为 |
|---|---|---|
| **去泄漏** | 校准集与测试集的**内容 sha256 交集为空**（不是索引不重叠） | 中止，**不接受人工放行** |
| **源模型只读** | 量化前后目录内容哈希一致 | 中止并告警 |
| **双门禁** | 精度损失 ≤ 阈值 **且** 性能收益 ≥ 目标 | 降级；精度接近上限（≥90%）需确认 |

去泄漏判据为什么按内容：同一段文本经常同时出现在数据集的 train/test 两个 split 里，按索引判断发现不了。
内容哈希是**可证伪**的 —— 交集为空可以逐条复核，血缘（来源 URL + revision + 种子 + 两段哈希清单）落盘后可事后重算。
宿主侧会用同一套函数**独立复核**容器的产出，不采信容器自己的结论。

## 常用命令

```bash
inference-quantization list --json                     # 量化矩阵（五要素 + 完整性）
inference-quantization report <runId>                  # 完整报告（逐档尝试 + QAT 计划）
inference-quantization qat --runId <id>                # 产出 QAT 计划（不执行）
inference-quantization rollback <selector> --purge-output
inference-quantization dataset verify <dir>            # 复算去泄漏
inference-quantization audit-verify                    # 审计哈希链
inference-quantization contract                        # testing-core 契约声明
```

## 共享状态

```
~/.ai-infra/state/quantization-matrix.yaml     量化矩阵（五要素 + 去泄漏/哈希证据）
~/.ai-infra/state/datasets/                    数据集与血缘
~/.ai-infra/state/images/                      量化镜像元数据（digest/扫描/签名）
~/.ai-infra/quantized/                         量化产出（**独立于模型根目录**）
~/.ai-infra/logs/audit/inference-quantization/ 审计，只追加 + 哈希链
~/.ai-infra/config/gates.yaml                  门禁阈值（模板里是 null，必须自己填）
```

## 排障

| 症状 | 先看什么 |
|---|---|
| 精度掉太多 | 报告里的 `parity`：`baseline/candidate` 困惑度与逐条 NLL；再考虑换更低损档或 QAT |
| 性能没提升 | `perf.baseline/candidate` 的 decodeTps 与 ttft。注意：小模型 + 短序列时收益本就有限，别把噪声当收益 |
| 一直不达标 | `report` 的 `attempts[]` 逐档记录；确认阈值是否设得过严 |
| 校准集被拒 | `dataset verify <dir>`，看 `problems`；换种子或换数据集，**不能绕过** |
| 源模型哈希变了 | 硬失败。检查是否把输出目录设到了源目录内（插件会先拦，但手工跑容器不受保护） |
| 镜像未验证 | `image verify --build`；缺扫描器/签名工具时会明说缺什么 |
| QAT 要跑 | `qat --authorize --allow-sandbox-degrade`，但仍需你自行执行计划里的命令 |

## 安全边界（本插件不做的事）

- **不执行 QAT 训练**（R3，本机无 VM 沙箱，按规则拒绝；只出计划）；
- **不修改源模型**（只读挂载 + 哈希校验）；
- **不放行泄漏的校准集**（无人工确认通道）；
- 不猜门禁阈值；
- 不装驱动 / 不提权 / 不动宿主服务（`manifest.yaml` 的 `not_provided` 显式列出）。
