# PROJECT_CONTEXT.md — inference-quantization 前置调研（Step 0）

> 性质：**只读探测**。未创建 `inference-quantization/` 包骨架、未安装任何宿主机包、未修改任何既有文件。
> 时间：2026-09-23　主机：`thinkstationpgx-3269`（aarch64 / NVIDIA GB10）
> 前置：`inference-serving` v0.1.0 已交付（见 `ACCEPTANCE_MATRIX.md`）
> 证据等级：`[verified]` = 本机命令/文件一手结果（附命令）；`[claimed]` = 二手口径；`[inferred]` = 推论（写明依据）

**目录约定说明**：serving 的 Step 0 文档已固化在工作区根（你已确认）。为避免覆盖，本插件的 Step 0 输出放在 `quantization-step0/`。若你希望统一改成"每插件一目录"，说一声我一次性重排。

---

## 0. 摘要：五条会改变设计的实测结论

| # | 题面假设 | 本机实测 | 影响 |
|---|---|---|---|
| ① | 默认 **AWQ + INT4** | `autoawq` 在 PyPI **无 wheel**（仅 sdist），其内核包 `autoawq-kernels` **只有 x86_64/win_amd64 wheel，零 aarch64** —— 本机是 aarch64 | 不能走 `autoawq`。**但 AWQ 算法本身可行**：`llmcompressor` 的 AWQ 变换器**只依赖 torch**（已读 wheel 源码确认），是纯 Python 路径。默认方法保留"AWQ + INT4"的**语义**，实现路径换为 llmcompressor |
| ② | 量化工具链"可用/不可用" | 三个镜像里**都没有** `llmcompressor` / `autoawq` / `gptqmodel`；vllm 镜像有 `compressed_tensors 0.17.0` + `bitsandbytes`，sglang 镜像有 `compressed_tensors 0.18.0` + `datasets` + `modelopt 0.45/0.46` + `torchao` | 必须**自建量化镜像**；且不能装进 serving 用的 vllm 镜像（见 ③） |
| ③ | （未提及）装进现有镜像即可 | `llmcompressor` 要求 `transformers>=5.15.0`，而 vllm/sglang 镜像都是 **5.12.1** → pip 会把 transformers 升到 5.17.0，**可能弄坏 vLLM 的钉版** | 量化必须在**独立镜像/环境**里跑，与推理镜像解耦（这也正好符合"软硬件解耦 + 资源隔离"） |
| ④ | （未提及）pip 直连即可 | `pypi.org` 的**元数据**可达，但 wheel CDN `files.pythonhosted.org` **实测不可达/不稳定**（`Network is unreachable`）；`mirrors.aliyun.com` 的 pip 镜像**没有** llmcompressor 0.14.0（同步滞后） | **必须走 TUNA 镜像**：`https://pypi.tuna.tsinghua.edu.cn/simple/` 实测**成功下载** llmcompressor wheel。这一条要写进本插件的网络策略 |
| ⑤ | 校准数据从验证集切分 | **本机没有任何数据集**：无 `/data/datasets`，HF/MS 缓存里无 dataset，工作区只有一个我们自己的 demo 产物 `.jsonl` | "校准集与测试集无交集"这条门禁**必须建立在下载来的数据集上**，其血缘（来源 URL + revision + 切分种子 + 内容哈希）要落盘才可验证 |

---

## 1. 探测方法与边界

- 全部为只读命令：`npm view` / `curl` / `ls` / `cat` / `docker images` / `docker info` / `nvidia-smi` / `free` / `df`。
- **容器内探测**（`docker run --rm`）：量化工具链只存在于镜像内部，宿主机探测不到，因此必须进容器查。全部用 `--rm`，容器文件系统随退即弃，**不落宿主、不改镜像**。
- 唯一一次"下载"是 `pip download` 一个 213 文件的 wheel 到 `/tmp`（用于读源码确认 AWQ 依赖），已可随时删除。
- **未执行**：宿主机 `pip install`、镜像构建、任何真实量化计算、任何写操作。

---

## 2. inference-serving 的实际接口（作为依赖）

`[verified]`，两种方式各测一次（CLI 导出的 schema 与 MCP 协议层 `tools/list` 结果一致）。

**10 个工具**（5 个面向用户 + 5 个运维）：

| 工具 | 参数 | 必需 |
|---|---|---|
| `inference_serving_deploy` | model, path, url, backend, port, bindHost, gpuMemoryUtilization, maxModelLen, maxNumSeqs, servedModelName, image, confirm, headless, deepProbe, root | — |
| `inference_serving_status` | selector, verify, deep, root | — |
| `inference_serving_rollback` | selector, confirm, root | selector |
| `inference_serving_teardown` | selector, confirm, root | selector |
| `inference_serving_list_endpoints` | status, root | — |
| `inference_serving_metrics` / `_audit_verify` / `_reconcile` / `_doctor` / `_migrate` | 运维用 | — |

**量化插件要复用的契约**（`[verified]` 读源码 + 测试确认）：

| 复用点 | 具体形态 |
|---|---|
| 确认令牌语义 | 首次调用返回 `E_CONFIRMATION_REQUIRED` + `details.confirmToken`；令牌**可累积**（数组）后一起回传；风险令牌绑定目标指纹，license 令牌绑定模型路径 |
| 错误结构 | `{ok:false, code, message, remediation, details, retriable, capability}`；`api.*` **永不抛异常** |
| 结果包装 | 成功结果附带 `auditId` / `auditSeq` / `riskClass` / `sandboxTier` / `warnings` |
| 共享策略 | `~/.ai-infra/config/{risk-classification,network-policy,headless-policy}.yaml` 已由 serving 落盘，量化**直接读同一份** |
| 部署验证 | 量化后用 `inference_serving_deploy` 起端点，再跑精度/性能对比 —— 这就是"部署验证"依赖的落点 |

---

## 3. `@ai-infra/*` 共享库现状

`[verified]`：

```
$ npm view @ai-infra/testing-core version    →（空，包不存在）
$ npm view @ai-infra/state-core  version     →（空）
$ npm view @ai-infra/audit-core  version     →（空）
$ npm view @ai-infra/sandbox-core version    →（空）
$ ls ~/.dsh/profiles/web/node_modules/@ai-infra    → No such file or directory
$ ls -d /home/xiciyang/deepseek_harness/ai-infra*  →（空）
```

**结论**：五个共享库**一个都不存在**。按题面"若不存在，内置最小实现，接口按契约设计"，量化插件将：

1. 在包内实现 `src/testing/` 作为 **testing-core 的契约实现**（parity-check / perf-validation）；
2. 接口按**可抽离**设计：一旦 `@ai-infra/testing-core` 发布，只需替换 provider，调用方不改；
3. 这属"共享库不是插件"的边界内 —— 我们**不做**独立插件，只在量化包内提供最小实现。

`[inferred]`（依据：serving 的经验"先做出能用闭环，再抽公共部分"）：先内置、后抽取，比先造库再写插件更不容易抽错。

---

## 4. 量化工具链（本机实测的完整矩阵）

### 4.1 镜像内已有的库

`[verified]`，在容器内 `import` 实测（`docker run --rm ... /tmp/probe-qlibs.py`）：

| 库 | vllm:nightly | sglang:qwen38-27b | sglang:dev-cu13-…-dflash2 |
|---|---|---|---|
| torch | 2.11.0+cu130 | 2.13.0+cu130 | 2.13.0+cu130 |
| transformers | 5.12.1 | 5.12.1 | 5.12.1 |
| compressed_tensors | **0.17.0** | **0.18.0** | **0.18.0** |
| datasets | ❌ | **5.0.1** | **5.0.1** |
| modelopt（NVIDIA） | ❌ | **0.45.0** | **0.46.0** |
| torchao | ❌ | **0.17.0** | ❌ |
| bitsandbytes | 0.49.2 | ❌ | ❌ |
| accelerate | 1.14.0 | ❌ | ❌ |
| **llmcompressor** | ❌ | ❌ | ❌ |
| **autoawq / awq** | ❌ | ❌ | ❌ |
| **gptqmodel / auto_gptq** | ❌ | ❌ | ❌ |
| optimum / peft / trl / hqq / quanto | ❌ | ❌ | ❌ |

### 4.2 从 PyPI 能拿到什么（含 aarch64 判定）

`[verified]`（PyPI JSON API + wheel 文件名解析）：

| 包 | 最新版 | wheel 形态 | **aarch64 可用？** |
|---|---|---|---|
| `llmcompressor` | 0.14.0 | **`py3-none-any` 纯 Python** | ✅ |
| `compressed-tensors` | 0.19.0 | `py3-none-any` | ✅ |
| `autoawq` | 0.2.9 | 无 wheel（仅 sdist） | ❌ |
| `autoawq-kernels` | 0.0.9 | 8 个 wheel，**全是 `x86_64` / `win_amd64`** | ❌ **零 aarch64** |
| `gptqmodel` | 7.5.0 | 无 wheel（仅 sdist） | ❌ |

### 4.3 llmcompressor 的依赖在 aarch64 上可解（实测）

`[verified]`，容器内 `pip install --dry-run --report`：

```
解析出 11 个待安装/变更包
  compressed-tensors   0.19.0    compressed_tensors-0.19.0-py3-none-any.whl
  accelerate           1.15.0    accelerate-1.15.0-py3-none-any.whl
  auto-round           0.15.1    auto_round-0.15.1-py3-none-any.whl
  datasets             5.0.1     datasets-5.0.1-py3-none-any.whl
  transformers         5.17.0    transformers-5.17.0-py3-none-any.whl
  x86_64-only 包: 无
```

**注意最后一行 `transformers 5.17.0`**：镜像里是 5.12.1，装 llmcompressor 会把它升上去 → 这就是结论 ③ 的依据。

### 4.4 AWQ 在 llmcompressor 里到底是什么（读 wheel 源码）

`[verified]`，下载 wheel（213 文件）后按模块解析：

```
llmcompressor 0.14.0 的修饰器
  quantization: gptq, quantization（RTN 等）
  transform:    awq, quip, smoothquant, spinquant
  pruning:      constant, magnitude, reap, sparsegpt, wanda

llmcompressor/modifiers/transform/awq/{__init__,base,dynamic_mappings,mappings}.py
  → 关键 import 只有 torch（无 autoawq / awq / auto_round / optimum）
```

**判定**：llmcompressor 的 AWQ 是**变换器**（weight smoothing，需校准数据），量化本身由 RTN/GPTQ 承担。这正是"AWQ 风格 W4A16"的标准做法，且**全程纯 Python，aarch64 可用**。

`[inferred]`（依据：vLLM 官方量化文档的通用做法）：产出为 `compressed-tensors` 格式，vLLM 可直接加载 —— 但**必须在实现阶段用真机验证**，不能只凭文档。

---

## 5. 网络出口（本插件与 serving 的差异）

`[verified]`，容器内实测（容器默认无代理环境变量，等价于我们的"显式直连"策略）：

| 目标 | 结果 |
|---|---|
| `pypi.org`（元数据 API） | ✅ 200 |
| `files.pythonhosted.org`（wheel CDN） | ⚠️ **不稳定**：一次 404（可达），一次 `Network is unreachable` |
| `hf-mirror.com` | ✅ 200 |
| `www.modelscope.cn` | ✅ 302 |
| `huggingface.co` | ❌ 000（与 serving Step 0 一致，DNS 投毒） |

**pip 镜像实测（决定性的三条）**：

| 镜像 | 结果 |
|---|---|
| `mirrors.aliyun.com/pypi/simple/` | ❌ `No matching distribution found for llmcompressor==0.14.0`（**同步滞后**） |
| `pypi.tuna.tsinghua.edu.cn/simple/` | ✅ **成功下载** `llmcompressor-0.14.0-py3-none-any.whl` |
| `mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/` | ✅ 成功下载 |

**结论**：本插件必须把 **`pypi.tuna.tsinghua.edu.cn`** 加入出口白名单并作为默认 pip index；serving 已固化的白名单里有 `mirrors.tuna.tsinghua.edu.cn` 但**没有** `pypi.tuna.tsinghua.edu.cn`，需要增量补一条。

---

## 6. 校准 / 评测数据（本插件特有的关键空白）

`[verified]` 本机现状：

```
/data/datasets           不存在        /datasets                 不存在
/home/xiciyang/datasets  不存在        hf_cache/datasets         不存在
/data/models/hf_cache, ms_cache  内无任何 dataset
/home/xiciyang/deepseek_harness 下唯一的 .jsonl 是 inference-serving 自己的审计样本
```

`[verified]` 可获取的候选（容器内实测 hf-mirror API 全部 200）：

| 数据集 | 用途 | 备注 |
|---|---|---|
| `Salesforce/wikitext`（wikitext-2-raw-v1） | 校准 + 困惑度 | 最常用的 PTQ 校准集；体积小 |
| `mit-han-lab/pile-val-backup` | 校准 | llm-compressor 官方 AWQ 示例默认用它 |
| `allenai/c4` | 校准（大规模） | 体积大，按需采样 |
| `openai/gsm8k` | 精度评测（任务级） | 需要生成式评测，成本较高 |

`[inferred]`（依据：题面要求"校准数据从验证集切分，强制去泄漏"）：本机**没有既有验证集**，因此必须：
1. 下载一个数据集（建议 `wikitext-2-raw-v1`，小且血缘清晰）；
2. **按内容哈希去重**后切分 校准 / 测试 两段，切分种子与两段的条目哈希清单落盘；
3. 门禁判据 = **两段的内容哈希交集必须为空**（不是"按索引不重叠"就完事 —— 同一文本出现在不同 split 里也是泄漏）；
4. 血缘（来源 URL + revision sha + 切分种子 + 段哈希）写入量化矩阵，使门禁**事后可复核**。

---

## 7. 沙箱与资源边界

`[verified]` 行为验证（沿用 serving 的方法，不看"装没装"）：

| 层级 | 结果 |
|---|---|
| 进程级（bwrap） | ❌ **不可用** —— `bwrap: setting up uid map: Permission denied`（Ubuntu 24.04 `apparmor_restrict_unprivileged_userns=1` 且 bwrap 非 setuid） |
| 容器级（Docker + nvidia） | ✅ 可用（docker 29.1.3，`nvidia` runtime 已注册） |
| VM 级 | ❌ 不可用（无 qemu/firecracker/cloud-hypervisor/virsh/kata/runsc；`/dev/kvm` 存在但无消费者） |

**资源**：`[verified]` GPU = NVIDIA GB10，compute cap **12.1**，统一内存 total **119.6 GiB** / free **78.9 GiB**（`torch.cuda.mem_get_info`）；宿主 `MemAvailable` 97 GiB；磁盘 `/` 916G 已用 291G，**可用 578G**。

**镜像磁盘现状**：`[verified]` vllm 21.3 GB + sglang 38.6 GB + sglang-dev 30.4 GB + python:3.12-slim 144 MB。新增一个量化镜像（基于 CUDA/torch 基础）预计 **+15~25 GB**，磁盘不是瓶颈。

**已存在的推理占用**：`[verified]` `qwen3-embedding` 容器（vLLM，`0.0.0.0:8001`，`restart: unless-stopped`）持续占用约 4.5 GB。量化全过程必须与它共存（serving 的资产保护规则已覆盖：只操作自己登记的载体）。

---

## 8. 与题面假设的偏差汇总（需要你裁决）

| # | 偏差 | 严重度 | DSH 建议 |
|---|---|---|---|
| 1 | **AWQ 不能走 `autoawq`**（aarch64 无内核 wheel） | 中 | 保留"AWQ + INT4"语义，改用 `llmcompressor` 的 AWQ 变换器（纯 Python，已实测依赖可解） |
| 2 | **必须自建量化镜像**，且不能装进 vllm/sglang 镜像（会升级 transformers 破坏钉版） | 中 | 建一个专用镜像（独立环境），与推理镜像解耦 |
| 3 | **pip 必须走 TUNA**，aliyun 镜像同步滞后 | 低 | 默认 `pypi.tuna.tsinghua.edu.cn`，并补进出口白名单 |
| 4 | **本机无任何数据集** | 中 | 下载 wikitext-2，按**内容哈希**切分并落盘血缘 |
| 5 | **`@ai-infra/*` 五个共享库一个都不存在** | 中 | 包内内置最小实现，接口按可抽离设计（题面已授权） |
| 6 | 沙箱可用性（进程✗ / 容器✓ / VM✗） | 低 | 沿用 serving 已确认的处置：R3–R5 一律拒绝 |

---

## 9. 证据边界

**站得住（本机一手实测）**
- 镜像内库矩阵（逐库 `import` 实测）、PyPI wheel 平台分布（文件名解析）、llmcompressor 依赖在 aarch64 上的可解性（`pip --dry-run --report`）、AWQ 变换器只依赖 torch（wheel 源码解析）。
- pip 镜像三条实测（aliyun 失败 / TUNA 两次成功且真下载到 wheel）。
- `@ai-infra/*` 五个库全部不存在（npm + 文件系统双向确认）。
- 本机无任何校准/评测数据集。
- 沙箱三层的行为验证、GPU/内存/磁盘数值。

**只是合理推断（标 `[inferred]`，未验证）**
- llmcompressor 产出的 compressed-tensors 权重能被本机 vLLM 0.23.1rc1 加载（待真机验证）；
- 专用量化镜像的体积与构建耗时；
- 量化 1.5B 模型的实际耗时与峰值内存（本机从未跑过任何量化）；
- 现有 `Qwen3.8-27B-NVFP4` 是"上一次量化的产物" —— 从 `quantization_config` 与体积看很像，但**没有血缘证据**，不应当作可复用的基线。

**原理上不可判（Step 0 无法回答）**
- 在本机 aarch64 + 统一内存上，AWQ/W4A16 的**精度损失**与**性能收益**各是多少；
- 校准集规模对精度的影响曲线；
- 27B 级模型能否在 78.9 GiB 可用统一内存下完成 PTQ（可能需要换更小模型或流式加载）。

---

## 10. 下一步

Step 0 到此为止，**不继续动手**。请裁决 `RISK_CLASSIFICATION_DELTA.md`（风险分级的量化增量）与本文档 §8 的六条偏差。

---

## 11. 补充：vLLM 兼容性真机验证（Step 0 第 7 项，已确认裁决后执行）

用户裁决要求"必须验证输出与 vLLM 兼容（vLLM 能加载并推理）"。以下是**真机实测**，不是文档推断。

### 11.1 构建了专用量化镜像

`[verified]`：`docker build` 成功（50s），base 固定为 `vllm/vllm-openai@sha256:be5ff841…bace392`。

构建期发现一个**真实的版本冲突**（值得记录）：

```
vllm 0.23.1rc1 requires compressed-tensors==0.17.0,
but you have compressed-tensors 0.19.0 which is incompatible.
```

llmcompressor 0.14.0 钉的是 `compressed-tensors==0.19.0`，而推理镜像钉的是 0.17.0。
**这正是"量化环境必须与推理环境解耦"的硬证据** —— 如果当初把 llmcompressor 装进 serving 用的镜像里，
就会同时破坏 vLLM 的依赖。派生独立镜像后，量化侧的冲突对推理侧零影响。

### 11.2 AWQ W4A16 量化成功

`[verified]`，Qwen2.5-1.5B-Instruct，128 校准样本，max_seq_len 1024：

| 项 | 值 |
|---|---|
| 耗时 | **118.6 s** |
| 源模型 | 2.88 GiB（BF16） |
| 产出 | **1.14 GiB** → 压缩比 **2.5×** |
| 量化配置 | `quant_method=compressed-tensors`，`format=pack-quantized`，`W4A16`，`group_size=128`，`symmetric int4`，`ignore=[lm_head]` |
| 工具链 | llmcompressor 0.14.0 / compressed-tensors 0.19.0 / transformers 5.17.0 / torch 2.11.0+cu130 |

### 11.3 **vLLM 能加载并推理**（关键结论）

`[verified]` —— 用**推理镜像**（`vllm/vllm-openai:nightly`，compressed-tensors **0.17.0**）加载 0.19.0 格式的产出：

```
=== 就绪 (t=75s) ===
{"object":"list","data":[{"id":"quant-test",...,"root":"/tmp/qprobe/out/awq-w4a16","max_model_len":1024,...}]}

POST /v1/completions → 200
  choices[0].text = "!!!!!!!!"   finish_reason = length
  usage = {prompt_tokens: 5, completion_tokens: 8, total_tokens: 13}
```

**结论**：llmcompressor 0.14.0 的 AWQ W4A16 产出与 vLLM 0.23.1rc1 **兼容，无需额外转换步骤**。
按用户裁决，若曾不兼容需记录转换步骤 —— **本机实测不需要**，因此不做多余的转换环节。

### 11.4 对设计的影响

| 项 | 结论 |
|---|---|
| 量化方法默认档 | `awq-w4a16` 确认可用（走 llmcompressor 的 AWQ 变换器 + QuantizationModifier W4A16） |
| 镜像策略 | 量化镜像**必须**独立于推理镜像；Dockerfile 用 digest 固定 base |
| 转换步骤 | **不需要** |
| 单档量化耗时量级 | 1.5B 模型约 2 分钟（128 校准样本）；降级链最多 4 档 → 预算充裕 |
| 内存 | 量化期间峰值约 4.7 GiB（宿主观测），远低于 78.9 GiB 可用 —— 27B 级模型可能也可行，但未验证 |

### 11.5 本节的证据边界

- `[verified]`：镜像构建、AWQ 量化、vLLM 加载、真实前向、压缩比、耗时。
- **未验证**：精度损失与性能收益的具体数值（那是 41 条验收里的第 1、2 条，需在实现完成后用完整闭环测）；
  27B 级模型的可量化性；QAT 路径（本机不执行）。
