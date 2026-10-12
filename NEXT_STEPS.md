# NEXT_STEPS.md — 下一个插件 `inference-tuning` 启动清单

> 由 `inference-quantization` 的实施经验直接导出。每条都标明**为什么现在能确定**、**依据是什么**、**哪些必须先探测**。
> 题目约束：不得提前实现 `inference-tuning`，不得修改 `dsh-tuning-engine`。

---

## 0. 先读什么（20 分钟）

| 材料 | 位置 | 读它的理由 |
|---|---|---|
| `inference-quantization/docs/step0/PROJECT_CONTEXT.md` | 量化包内 | 量化工具链的实测矩阵；**§11 有 vLLM 兼容性的真机结论**（tuning 会复用同一条链路） |
| `inference-quantization/docs/step0/RISK_CLASSIFICATION_DELTA.md` | 量化包内 | 已在 serving 基线上追加过一轮增量；tuning 的增量应叠加在这份之上 |
| `inference-quantization/ACCEPTANCE_MATRIX.md` | 量化包内 | 41 条验收的证据形态；tuning 照此结构交付 |
| `inference-quantization/DEPLOYMENT_REPORT.md` | 量化包内 | **§五"实施中发现并修掉的真问题"** —— 那 8 条里有 4 条是可直接复用的教训 |
| `inference-quantization/skills/quantization-method-authoring/SKILL.md` | 量化包内 | "方法插件化"的契约模板；tuning 的调优策略注册表应照此设计 |
| `dsh-tuning-engine/`（**只读**） | 工作区 | 原 tuning-engine 的 8 个子包结构（`collect`/`attributor`/`decider`/`guard`/`contracts`/`statistics`/`operator`/`ceilings`）与 `lib/backends`、`lib/walls` |

---

## 1. 已经可以确定的事（不需要重新探测）

| 结论 | 依据 |
|---|---|
| **两个接缝设计可直接复用** | 量化包里的 `src/backends/registry.mjs`（方法注册表）与 `src/executors/`（执行器接缝）。tuning 需要的是"调优策略注册表" + "引擎控制器执行器"，形状完全一致 |
| **执行器接缝是让验收不靠 mock 的关键** | 量化包的 `tests/integration/quantize-loop.test.mjs` 用假执行器跑完整闭环（12 个用例，无 GPU），而生产路径零改动。tuning 必须照做，否则"真机演示"会退化成"只能人工跑" |
| **零运行时依赖可行且应坚持** | 两个包都是 `dependencies: {}`；`npm install` 秒过，且避开本机 npm 官方源不可达 |
| **共享状态外置 + 文件锁 + 原子写 + 写前快照** | 量化的 `state/matrix.mjs` 与 serving 的 `state/registry.mjs` 是同一套，已验证过并发与恢复 |
| **审计只由 ops 闸门层写** | 两个包都遵守；FSM 层不写审计（避免两处写导致语义分叉） |
| **确认令牌必须指纹稳定** | 量化踩过：指纹把 `confirm` 自己算进去 → 令牌每次重试都变 → 确认流程**永远走不通**。指纹只能由"要做什么"决定，与"怎么授权的"无关 |
| **确认令牌要认四个位置** | 第一参数的 `confirm`、`input.options.confirm`、第二参数的 `confirm`、第二参数的 `options.confirm`。少认一个就出现"传了令牌还被再问一次"的鬼打墙 |
| **被 await 的定时器绝不能 `unref`** | serving 踩过（进程静默退出 exit 0）。两个包都有回归测试（`tests/unit/unref-timers.test.mjs`） |
| **配置覆盖要剔除 `undefined`** | 量化踩过：CLI 产出的 `{modelRoots: undefined}` 把默认值清空 → `modelRoots is not iterable` |
| **大文件哈希必须流式** | 量化踩过：`fs.readFile` 在单文件 > 2 GiB 时抛 `ERR_FS_FILE_TOO_LARGE`，而 1.5B 模型的单个 safetensors 就有 2.88 GiB |
| **沙箱**：进程✗ / 容器✓ / VM✗ | 行为验证（不是查二进制） |
| **量化镜像与推理镜像必须分离** | llmcompressor 要 `compressed-tensors==0.19.0`，推理镜像钉 `0.17.0`；装一起会破坏 vLLM |
| **pip 走 TUNA**（aliyun 同步滞后） | 量化 Step 0 实测三条 |
| **`@ai-infra/*` 五个共享库都不存在** | 已内置 `testing-core` 的 `@internal` 契约实现 + 迁移脚本 `scripts/migrate-to-testing-core.mjs`；tuning 应复用同一份，不要再造一个 |

---

## 2. 必须先探测的（Step 0，只读 + 行为验证）

1. **`dsh-tuning-engine` 的真实接口面**（**只读**）
   - 它是 workspace 根包 + 8 个子包（`@dsh-tuning/*`），入口 `lib/index.mjs`，bundle entry id = `tuner`；
   - **本任务只读，不改**。要回答的是："inference-tuning 与它的关系是什么？"——
     是**改名**（题目原话"原 tuning-engine，后续改名"）还是**并存**？
     这直接决定 `NEXT_STEPS` 之后是"重命名迁移"还是"新写一个包"。**这是需要裁决的第一个问题。**
2. **调优的目标引擎与可达性**
   - 本机既有 `qwen3-embedding`（vLLM，`0.0.0.0:8001`）；sglang 镜像两个；
   - 必须实测：能起哪些引擎、各自的 `/metrics` 与 `/v1/loads`（sglang）暴露什么字段。
     tuning 的"根因墙"判据依赖这些字段，字段缺失就必须声明不可判。
3. **实测带宽与算力**（用户长期指令：**不得用规格书理论峰值**）
   - 用户对 tuning 的方法论要求是硬约束：判墙的分子分母都必须来自**本机实测 kernel 流量与耗时**；
   - Step 0 必须先建立"本机实测带宽/算力"的取法，并写清限定条件（当前 kernel 实现 / 状态缓存格式 / 量化精度）。
4. **量化产出的可调优性**
   - 量化包已能产出 AWQ W4A16；tuning 能否在量化模型上调优？
   - 需要实测：量化模型在 vLLM 上的 `/metrics` 是否与 BF16 一致（tuning 依赖这些指标）。
5. **`lib/walls` 与 `lib/backends` 的现状**（只读）
   - 原包已有 `walls` 目录与 `backends/{controllers,engines,gpuspecs,probes,registry}`；
   - 要弄清哪些可复用什么必须重写 —— 但**不要动它们的文件**。
6. **失败与回退的边界**
   - 调优是改运行参数，回退容易；但若涉及容器重启，需要与 serving 的 teardown/deploy 协同。
     探测 serving 与 quant 的 MCP 接口能否支撑"重配→重启→复测"的闭环。

---

## 3. 需要裁决的架构点（Step 0 输出时一并提出）

| # | 问题 | 选项 | DSH 倾向 |
|---|---|---|---|
| 1 | **与 `dsh-tuning-engine` 的关系** | (a) 就地改名为 `inference-tuning` / (b) 新写一个标准 npm 包，原包保留只读 / (c) 抽取共享库 | **(b)**：题目要求"标准 npm 包 + 零运行时依赖"，而原包是 workspace 多包结构、有 3 个 `.bak` 文件与 `MIGRATION.md`；就地改名风险高且可能破坏既有会话。等你裁决 |
| 2 | 调优单元 | (a) 单引擎参数 / (b) 跨引擎选型 / (c) 两者都要 | **(a) 优先**：编排（跨引擎）题目明确留给 RSI 阶段 |
| 3 | 根因墙的证据来源 | (a) 引擎 `/metrics` + DCGM / (b) 需要 kernel 级 profiler | 先做 (a) 并把"哪些墙原理上不可判"写进报告；(b) 作为可选增强 |
| 4 | 与量化产出的组合 | (a) 只调 BF16 / (b) 也调量化模型 | **(b)**：量化的性能收益与调优叠加才是真实收益；但要注意用户对主模型量化的排除指令（见 §5） |
| 5 | 精度回归的执行者 | 复用 `testing-core` 契约实现 | **复用**，不要再写一份（口径分裂是真实风险） |

---

## 4. 交付形态（照量化包的模板）

```
inference-tuning/
  package.json          零运行时依赖；engines/os/cpu；dsh.bundle.patch；bin+exports
  manifest.yaml         能力清单白名单 + not_provided（显式列出不做的事）
  cordis.patch.yml      id/name/inject: [tools]
  README.md  LICENSE  .npmignore
  docs/step0/{PROJECT_CONTEXT.md, RISK_CLASSIFICATION_DELTA.md, NETWORK_POLICY_DRAFT.md}
  ACCEPTANCE_MATRIX.md  DEPLOYMENT_REPORT.md  NEXT_STEPS.md
  src/
    index.mjs api.mjs tools.mjs cli.mjs ops.mjs errors.mjs config.mjs version.mjs
    log.mjs metrics.mjs policy-defaults.mjs
    util/{yaml,hash,fsx,proc,http,lock}.mjs      ← 可直接复用
    capability.mjs risk.mjs netpolicy.mjs sandbox.mjs audit.mjs   ← 可直接复用
    drivers/ executors/                          ← 接缝，可复用形状
    backends/                                     ← 调优策略注册表（新写）
    state/{matrix→tuning-matrix}.mjs             ← 照 matrix.mjs 改
    input/pipeline.mjs probe/{env,model}.mjs     ← 可直接复用
    tuning.mjs  degrade.mjs  gate.mjs            ← 核心（新写）
    testing/                                      ← 复用 testing-core 契约
  bin/  mcp/  skills/  tests/  scripts/  docker/
```

**可直接整文件复用的**（从量化包拷贝，改包名与路径）：
`src/util/*`、`src/capability.mjs`、`src/risk.mjs`、`src/netpolicy.mjs`、`src/sandbox.mjs`、
`src/audit.mjs`、`src/ops.mjs`、`src/log.mjs`、`src/errors.mjs`（扩码）、`src/drivers/*`、
`src/probe/*`、`src/integrity.mjs`、`src/gate.mjs`（判定骨架）。

---

## 5. 必须守住的既有指令（来自记忆，不是本任务的推断）

- **主模型不考虑量化优化**（用户 2026-09-18 明确，是硬排除，不是优先级问题）。
  tuning 若想通过"改量化格式"换来性能，**不允许**。
- **给出目标后优化中途不能停止**：须自主推进到"一轮完成目标或达到物理瓶颈"为止；
  "已达标"与"预算耗尽"都不构成停止理由。
- **根因墙驱动，不是盲搜参数空间**：先给每堵墙定上界，再按世界经验匹配 TOP1 干预，最后自闭环验证。
  TOP1 干预无效时**先停下反思五种原因**（杠杆未生效/墙找错/墙转移/测量不成立/真硬件极限），
  不许直接换参数再试。
- **任何"硬件到顶"判定必须写明限定条件**，不得当作物理理论上界。
- **任务级精度评估在每个优化前后各做一次**，且不能用字符串精确匹配代替。

---

## 6. 不要做的事

- 不得修改 `inference-serving` 与 `dsh-tuning-engine` 的任何文件；
- 不得提前实现 `inference-tuning`（本清单只是启动准备，不是实现）；
- 不引入未声明的依赖（保持 0 依赖标准）；
- 不绕过沙箱、不提权；
- 不把"未发现反例"呈现成接近结论；缺证据时显式声明"某类假设原理上不可判"。

---

## 7. 建议的开工顺序

```
1. 只读探测 §2 的六项（含 §2.1 那个必须先裁决的关系问题），产出 PROJECT_CONTEXT.md
2. 就 §3 的五个架构点给草案，等裁决
3. 照 §4 搭包，复用 §4 列出的可直接复用清单
4. 先跑一次真实的"基线测量 → 判墙 → TOP1 干预 → 复测"，把实测数字写进报告
   （这一步必须先于任何自动化，否则自动化的是错的判据）
5. 补齐调优策略注册表、降级/回退、审计与文档
6. 交付 ACCEPTANCE_MATRIX.md + DEPLOYMENT_REPORT.md + 下一个阶段的清单
```
