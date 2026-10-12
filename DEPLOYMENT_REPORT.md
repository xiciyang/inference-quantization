# DEPLOYMENT_REPORT.md — inference-quantization 交付与真机验收报告

> 生成：2026-09-24　交付物：`inference-quantization/` v0.1.0（零运行时依赖）
> 配套：`ACCEPTANCE_MATRIX.md`（41 条逐条对照）· `docs/step0/*`（前置调研与风险裁决）· `NEXT_STEPS.md`

---

## 一、结论先行

| 项 | 结论 |
|---|---|
| 闭环 | ✅ 完整实现并真机跑通：源模型预检 → 选方法 → PTQ 校准（去泄漏）→ 精度对比 → 性能对比（真实端点）→ 双门禁（**数值 + 测量成立性**）→ 签收/降级/QAT 计划/回退 |
| 工程 | ✅ **129 个测试全通过**、**81 项包自检全通过**、零运行时依赖、CLI/编程接口/MCP/DSH 插件四入口 |
| **精度（真机，有效测量）** | ✅ **AWQ W4A16 实测困惑度退化 +19.571%**（15.3366 → 18.3381，wikitext-2，400 条 / 67,658 token），阈值 50% → 通过。测量成立性判定 `valid=true` |
| 真机性能收益 | ✅ **+151.17%** 解码吞吐（64.80 → 162.77 tok/s），TTFT −38.64%（38.5 → 18.0 ms）—— 走真实 vLLM 端点 |
| 真机降级链 | ✅ 实测触发 **4 档**（L1→L2→L3→L4）后产出 QAT 计划 |
| 硬门禁 | ✅ 去泄漏（内容哈希交集为空）、源模型只读（哈希前后校验）、**源模型结构完整**、**测量成立性**、有界降级（≤5 档）全部真机验证通过 |
| 上一版阻塞项 | ✅ **已定位并修复**：根因是**源模型文件本身残缺**（下载中断），不是评测脚本的错。见 §五 |

**一句话**：上一版报出的"精度零损失"是**源模型文件残缺**造成的假象 —— 那份 `model.safetensors`
只有前 0.747 GiB 是真数据，其后 2.128 GiB（223/338 个张量，含 `model.norm.weight`）全是 `0x00`。
模型因此输出均匀分布，`perplexity` 恒等于词表大小，`drop = 0%` 毫无意义。
这一轮补了**两层独立防线**（源模型结构预检 + 测量成立性门禁）并**在真实完好模型上重跑**，
拿到了本机第一组有效的精度数字。

---

## 二、实施过程中的 16 个真问题（最有信息量的部分）

这些都是**真机跑出来的**，不是推演。前 5 个是设计/实现缺陷，中间 6 个是工程陷阱，
最后 5 个是**上一版交付后被追出来的**（其中 12 号是本次阻塞项的根因，16 号是重跑时新发现的资源泄漏）。

| # | 问题 | 怎么发现的 | 修法 |
|---|---|---|---|
| 1 | **`fs.readFile` 在单文件 > 2 GiB 时直接抛** `ERR_FS_FILE_TOO_LARGE`；而 1.5B 模型的单个 safetensors 就是 **3.09 GB** | 首次真机量化，源模型哈希阶段直接崩 | `hashDirectory` 改为**流式哈希**（`createReadStream` + `createHash`）；补 2 个回归测试（造 3 GiB 稀疏文件，并对照断言 `fs.readFile` 会抛） |
| 2 | **去泄漏判据误用**：我把"两段并集 == 去重全集"当成不变量 | 真机报 `两段并集 528 条 ≠ 去重全集 14324 条（有样本丢失）` | 两段是从语料**抽样**的子集，要求覆盖全集在逻辑上就是错的。改为"并集规模不得超过全集"+"某段不得超过全集"，并补测试断言**抽样段不应被要求覆盖全集** |
| 3 | **确认令牌的指纹把 `confirm` 自己算进去了** | 真机 + 测试双重复现：带令牌重试又抛一次"需要确认" | 指纹只由"要做什么"决定，排除 `confirm`/`options` 等记账字段。**这个缺陷会让确认流程永远走不通**，而错误信息看起来像用户没确认 |
| 4 | **令牌四个位置只认了三个** | 同一个测试 | `input.confirm` / `input.options.confirm` / 第二参数 `confirm` / 第二参数 `options.confirm` 全部认。少认一个就出现"传了令牌还被再问一次"的鬼打墙 |
| 5 | **闸门顺序：先查镜像再查阈值** | 测试期望 `E_THRESHOLD_NOT_PROVIDED` 却拿到 `E_IMAGE_NOT_VERIFIED` | 配置类错误应最先失败（不需要碰环境、更便宜、更可操作）。阈值检查提到镜像检查之前 |
| 6 | **迁到执行器接缝时漏挂了 datasets 目录** | 真机四级全部 `E_QUANTIZE_FAILED`，**每级只耗时 3–6 秒** —— 一眼就能看出不是真在跑 | 补挂 `datasetDir:ro`。**教训**：挂载清单在重构中丢失不会有编译期报错，只有"快得离谱"这一个信号 |
| 7 | **CLI 的 `undefined` 覆盖项把默认值清空** | `doctor` 崩在 `runtime.options.modelRoots is not iterable` | `createRuntime` 与 `buildCommonOptions` 在合并前剔除 `undefined`。与 serving 踩过的是同一类坑 |
| 8 | **`oneshot` 要 HF `Dataset` 而不是 list** | 首次真实量化：`'list' object has no attribute 'column_names'` | 校准集改用 `datasets.Dataset.from_list` |
| 9 | **量化镜像与推理镜像的压缩库版本冲突** | 镜像构建日志：`vllm 0.23.1rc1 requires compressed-tensors==0.17.0, but you have 0.19.0` | 量化环境**必须独立于推理环境**（这正是"资源隔离"的硬证据）。实测 0.17.0 **能读** 0.19.0 的产出，无需转换步骤 |
| 10 | **hf-mirror 连接会抖**（`handshake operation timed out`） | 真机数据集准备失败一次 | 加**有界重试**（3 次 + 退避 + 清半成品）+ 挂**持久化 HF 缓存**（重试不必重下） |
| 11 | **`allowUnverifiedImage` / 门禁阈值在 CLI 路径上被静默忽略** | 真机连续两次失败：先 `E_IMAGE_NOT_VERIFIED`，再 `E_THRESHOLD_NOT_PROVIDED` | 第二参数的 `options.*` 与 input 的字段统一提升；阈值支持三种写法（camel/snake/options.*）。**"看起来传了其实没传"是本插件最不该发生的事** —— 阈值缺失必须显式拒绝 |
| **12** | **源模型文件残缺（下载中断）而流程毫无察觉** —— 338 个张量里 223 个整块为零，含 `model.norm.weight` | 上一版签收了"精度 0% 损失"，怀疑数字太漂亮 → 逐张量扫描 safetensors | 新增**源模型结构预检**：截断校验 + 逐张量精确零覆盖 + 层数吻合 + 分片齐备（§五 FIX-1）。真机 **2.6 s** 拦住 |
| **13** | **去泄漏/哈希校验回答不了"内容是否完整"** —— 目录哈希前后一致，因为从来没人改过它 | 与 12 号同源：两条旧检查全部通过 | 明确区分"**被改动**"与"**是完整模型**"两个命题，各配一条独立防线（§五 5.3） |
| **14** | **门禁把"测量不成立"当成了"数值达标"** —— 均匀输出 ⇒ `drop = 0%` ⇒ 签收 | 12 号的下游后果 | 门禁加入**测量成立性判定**（五条否证），`verified` 现在需要"数值达标 ∧ 决策 sign ∧ 测量成立"三条同时成立（§五 FIX-2） |
| **15** | **测试夹具本身是"假测量"** —— 假执行器的逐条 NLL 是常量，正好是 validity 门禁要拦的形态 | 实现极差判定后，集成测试**全部失败** | 夹具改为产出**有真实方差的**逐条 NLL（偏移对基线/候选相同，故 `drop%` 仍等于配置值）。**夹具不真实，测试就在替 bug 背书** |
| **16** | **性能评测只部署不拆解 —— 每跑一次量化泄漏两个占显存的 vLLM 容器** | 重跑后 `docker ps` 里又出现 2 个 up 着的 `inference-serving-*`；回看上一版也是 2 个 —— 两次运行、两次泄漏 | `perfValidation` 改为 `try/finally`：**测量成功或抛异常都拆**；拆解失败不改判性能结论，但写 `cleanup.ok=false` 并推 `W_RESIDUAL_CLEANUP`。新增 4 个用例（含"deploy 次数必须等于 teardown 次数"） |

### 2.1 这些发现说明了什么

- **第 1、2、6、11 号只有真机跑才会暴露**：本地测试全绿的时候，这四个一个都没露头。
- **第 3、4、5 号是契约层面的**，靠"追问一句这样对吗"发现，不是靠跑。
- **第 6 号是个典型的"重构静默丢东西"**：唯一的信号是"快得离谱"。值得记进方法论。
- **第 12–14 号是同一件事的三个面**：坏输入 → 旧防线答错了问题 → 门禁把沉默当成了合格。
  这三条串起来才构成完整的根因；单看任何一条都会得出错误的结论
  （上一版就止步于"打分脚本没加载权重"，指错了责任方）。
- **第 15 号最反直觉**：测试全绿不代表夹具是真实的。夹具喂了一个"与输入无关的常量测量"，
  于是我原有那批用例一起替这个 bug 背了书。**夹具的真实性本身需要被审视。**
- **第 16 号的信号非常便宜却两次被忽略**：`docker ps | grep serving`。
  "跑完就好了"是一个需要被显式验证的断言，不是默认成立的。

---

## 三、真机验收的关键数字

### 3.1 主闭环（`runId=qr_45e79f22f441`，**有效测量**）

| 项 | 值 |
|---|---|
| 命令 | `quantize /data/models/Qwen2.5-1.5B-Instruct-verified --max-accuracy-drop-pct 50 --min-perf-gain-pct 0 --allow-unverified-image` |
| 结果 | `ok=true signed=true outcome=gated`，门禁判定 `sign`，`measurementValid=true` |
| 总耗时 | **733.6 s**（其中量化 410.6 s —— 校准集 512 条，比上一版的 128 条多 4 倍） |
| 方法 | `awq-w4a16`（L1，默认档），**未降级** |
| 产出 | `~/.ai-infra/quantized/Qwen2.5-1.5B-Instruct-verified-awq-w4a16` |
| 体积 | 2.88 GiB → **1.06 GiB**，压缩比 **2.712×** |
| 量化格式 | `compressed-tensors` / `pack-quantized` / W4A16 / group_size=128 / symmetric int4 / ignore=[lm_head] |
| 工具链 | llmcompressor 0.14.0 · compressed-tensors 0.19.0 · transformers 5.17.0 · torch 2.11.0+cu130 |
| **精度** | **ppl 15.3366 → 18.3381，退化 +19.571%**（400 条 / 67,658 token） |

> 上一版的 `qr_25b26474a21b` 已确认**不可作为验收证据**：它是在残缺源模型上跑出来的，
> 其 `verified` 结论无效（详见 §五）。矩阵中该条记录保留原样以供追溯，本报告不再引用它的精度数字。

### 3.2 性能验收

经 MCP 让 `inference-serving` 把基线与候选**各起一个真实端点**，对两个端点发同一份负载
（`chat-medium`，8 请求，并发 2）。下表为重跑数据：

| 指标 | 基线 BF16 | 候选 AWQ W4A16 | 变化 |
|---|---:|---:|---:|
| 解码吞吐（tok/s，中位数） | 64.80 | **162.77** | **+151.17%** |
| 首 token 延迟 TTFT（ms，中位数） | 22.0 | **13.5** | **−38.64%** |
| 聚合吞吐（tok/s） | — | — | +149.92% |

**为什么这个数据可信**：它走的是真实 vLLM HTTP 路径，与"精度测量"是两条独立链路；
数值差异巨大且方向符合 W4A16 的物理预期（权重流量降到 1/4）。

> 上一版的性能数字（+140.78% / −54.4%）方向与量级一致，但它是在**残缺模型**上测的
> （layers 11–27 退化成直通，两侧都被"加速"过），已一并作废，不再引用。

### 3.3 降级链验收

另一次真机运行（阈值收紧）实测**四档全部触发**：

```
L1:awq-w4a16(E_QUANTIZE_FAILED) → L2:int8-w8a8(E_QUANTIZE_FAILED)
  → L3:fp8-dynamic(E_QUANTIZE_FAILED) → L4:w8a16(E_QUANTIZE_FAILED)
  → 产出 QAT 计划（不执行训练）
```

> 这一级的失败原因是当时数据集目录漏挂（问题 #6），**但降级链本身的行为是对的**：
> 四级依次推进、每级记录失败码、用尽后进 QAT 档、矩阵写 `qat-plan-required`。
> 修好挂载后同一条链路在 L1 就成功了（§3.1）。

### 3.4 三道硬门禁（真机数据）

| 门禁 | 真机结果 |
|---|---|
| **去泄漏** | `verified=true`，`criterion=content-sha256-intersection-empty`，`counts={calibration:512, test:400, unique:912}`，`problems=[]`，记录 `calibListDigest`/`testListDigest`；**宿主侧独立重算** |
| **源模型只读** | `unchanged=true`，`hashBefore == hashAfter`（流式 sha256） |
| **源模型结构完整** | `ok=true`，338 张量 / **0 个整块为零**；对残缺副本则 **2.6 s** 判 `E_SOURCE_MODEL_INVALID` 并点名 223 个零张量 |
| **测量成立性** | `valid=true`；`bitIdenticalMeanNll=false`、`uniformGap=9.20`、逐条极差 `3.367`；对残缺模型判 `valid=false`（四条理由） |
| **降级有界** | 最多 5 档；真机实测 4 档后进 QAT 档，未越界 |

---

## 四、本次交付**没有**做到的事（逐条声明）

1. ~~精度门禁无判别力~~ —— **已修复并真机复验**（§五）。上一版那个阻塞项不再存在。
2. ~~量化专项 1 未能真实验证~~ —— **已重跑通过**：`qr_45e79f22f441`，精度退化 +19.571%（阈值 50%），测量成立。
3. **`@ai-infra/testing-core` 不存在**，用了包内 `@internal` 最小实现（已按你的授权；接口按契约、标注 `@internal`、有迁移脚本）。
4. **QAT 未执行**（按裁决只产出计划）。
5. **27B 级模型未量化**（未测可量化性）。
6. **量化镜像未通过完整验证**：`verified=false`，缺 `["从未验证"]` —— 本机没有 `trivy`/`grype`/`cosign`，所以"漏洞扫描"与"签名"两项**无法完成**。演示用 `allow_unverified_image=true` 显式授权并记入审计（这正是设计要的行为：**缺什么就说什么，不假装通过**）。
7. **DSH 内实际挂载未验证**（同 serving：用 mock ctx 做了 8 项契约断言，但没有真注入运行中的 DSH 进程）。

---

## 五、已定位并修复：上一版"精度零损失"的根因是**源模型文件残缺**

### 5.0 结论（先给结论，再给证据）

上一版把阻塞项写成了"精度评测端模型未被真正加载"。**这个判断只对了一半，而且指错了责任方。**
逐层取证后，根因是：

> **打分脚本是对的，模型文件是坏的。**
> 源模型 `/data/models/Qwen2.5-1.5B-Instruct`（软链 → `ms_cache/.../snapshots/master`）的
> `model.safetensors` 只有前 **0.747 GiB** 是真实权重，其后 **2.128 GiB 全是 `0x00`**
> —— 一次**下载中断**留下的空洞（同目录还留着 `model.safetensors.download.aria2`）。
> 338 个张量里 **223 个整块为零**，**包括 `model.norm.weight`**（最终 RMSNorm 的缩放权重）。

### 5.1 根因推导链（每一步都有独立证据）

**① 先排除"文件被截断"** —— 解析 safetensors 头部：

```
头部声明数据终点 = 3,087,467,144      实际文件大小 = 3,087,467,144      差 = 0
```

文件长度是对的（下载器预分配过），**所以"截断"这条假设被证伪**。这正是旧流程看不见它的原因。

**② 逐张量扫描，找到真正的形态** —— 按头部声明的每个张量字节区间逐个读：

```
整块为零的张量： 223 / 338
零字节：         2,145,107,968 B（占数据区 69.5%）
首个零区起点：   data offset 802,587,648
被点名的张量：   model.layers.11.* … model.layers.27.*，以及 model.norm.weight
```

**③ 从机制上解释"为什么损失恰好等于 ln(vocab_size)"** —— 这条是判决性的：

```
layers 11–27 的权重全为 0  ⇒  这些层退化成残差直通（attention/MLP 输出恒为 0）
model.norm.weight 全为 0   ⇒  最终 RMSNorm 的输出恒为 0
lm_head 与 embed 绑定      ⇒  logits = W(0) = 0
所有 logits 相同           ⇒  softmax = 均匀分布  ⇒  loss ≡ ln(vocab_size)
```

数值核对：`ln(151936) = 11.9312146585`，实测 `11.9312144460`，差 **2.1×10⁻⁷** ——
正好落在 float32 的累积误差量级内。

**④ 同一份模型、换一个完好副本，直接对照**（真实硬件，各 20 条）：

| 模型副本 | meanNll | perplexity | 逐条方差 | 判定 |
|---|---|---|---|---|
| `ms_cache`（残缺） | 11.9312144460 | **151935.968** | 0 | 均匀分布 |
| `hf_cache`（完好） | **2.556013** | **12.884** | 0.445 | 正常 |

同一个模型、同一份测试集、同一条代码路径 —— 差异只能来自权重文件本身。

**⑤ 候选侧为什么也一样** —— 候选是用**同一份残缺权重**量化出来的，
零权重量化后仍是零，所以基线与候选逐位相同、`drop = 0%`。

### 5.2 为什么它有害

它**不报错**。管线照常跑完，`drop = 0.0000%` 看起来是"量化无损"——比真实结果还漂亮。于是：

```
gates.accuracy.passed = true  →  gates.decision = 'sign'  →  矩阵 status = 'verified'
```

**一个退化的测量骗过了精度门禁，并把一次毫无意义的量化标记成了 `verified`。
而且它还烧掉了 436 秒 GPU 时间去量化一份废权重。**

### 5.3 修复：两层互相独立的防线

原来的流程有两条检查，但**两条都回答不了这个问题**：

| 原检查 | 它回答的问题 | 为什么拦不住 |
|---|---|---|
| `E_SOURCE_MODEL_UNREADABLE` | 目录**读得到**吗 | 读得到 |
| 目录哈希前后一致 | 内容**有没有被改动** | 没被改动 —— 它一开始就是残缺的 |

新增的两层，各自回答一个新问题：

**FIX-1 — 源模型结构预检（`src/model-validate.mjs`，`E_SOURCE_MODEL_INVALID`）**
—— 回答"这**是不是一个完整的模型**"。排在数据集准备与量化**之前**（坏输入要最便宜地失败）：
safetensors 头部与文件长度的截断校验、**逐张量精确零覆盖扫描**、
config 声明层数 vs 权重实际层数、`index.json` 分片齐备性。
结果按 `(路径, 字节数, mtime)` 缓存，只缓存"通过"结论。

**FIX-2 — 测量成立性门禁（`src/gate.mjs#assertMeasurementValid` + `parity.py` 的 `validity` 块）**
—— 回答"这次**测量成不成立**"。两侧各五条互相独立的否证，任一条成立即拒绝签收：

| # | 否证 | 判据 |
|---|---|---|
| ① | 没有评到分 | `scoredSamples = 0` |
| ② | 权重健康度异常 | 存在整块为零 / 含 NaN·Inf 的大张量 |
| ③ | **均匀输出签名** | `\|meanNll − ln(vocab_size)\| ≤ 1e-4` |
| ④ | **无区分力** | 基线与候选 `meanNll` **逐位相同**（两个不同权重给同一个数） |
| ⑤ | **与输入无关** | 逐条 `meanNll` **极差 = 0** |

判定与测量**刻意分离**：`parity.py` 只报事实（`ok` 仍为 `true`，数值照给，另附 `validity` 块），
"要不要据此拒绝"由纯函数门禁决定 —— 于是门禁可以被穷举测试，测量端可以被独立复核。
`verified` 状态现在需要**三条同时成立**：数值达标 ∧ 决策为 `sign` ∧ 测量成立。

第 ③ 条刻意用**相对词表大小**而不是"ppl > 1000"这类魔数：
均匀输出的困惑度在数学上恒等于词表大小，这是一个**可推导的等式**，不是经验阈值。
第 ⑤ 条刻意用**极差（max − min）**而不是标准差：标准差要先把 N 个数求和再相除，
浮点误差会让"全都相同"算出 ~1e-15，于是必须引入容差魔数；极差是精确比较，不需要任何阈值。

### 5.4 修复前后的真机对照

| | 修复前 | 修复后 |
|---|---|---|
| 残缺源模型 | 量化 436 s → 签收"精度 0% 损失" | **2.6 s 失败**：`E_SOURCE_MODEL_INVALID`，点名 223 个零张量、给出零区与 remediation |
| 打分脚本对残缺模型的判定 | `valid`（无此字段），照常出数 | `validity.valid = false`，四条理由：`zero-weight-tensors` / `uniform-output` / `constant-logits` / `constant-argmax` |
| 打分脚本对完好模型的判定 | — | `validity.valid = true`，`uniformGap = 9.375`、`logitsStdMean = 3.005`、`distinctArgmax = 92` |

### 5.5 在完好模型上重跑（`量化专项 1`）

`runId = qr_45e79f22f441`，源模型 `/data/models/Qwen2.5-1.5B-Instruct-verified`
（sha256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`）。

| 项 | 值 |
|---|---|
| 基线 | meanNll **2.730240**，perplexity **15.3366**（400 条 / 67,658 token） |
| 候选（AWQ W4A16） | meanNll **2.908983**，perplexity **18.3381** |
| **精度损失** | **+19.571%**（阈值 50% → 通过） |
| 测量成立性 | `valid = true`；`bitIdenticalMeanNll = false`；`uniformGap = 9.20`；逐条极差 `3.367` |
| 源模型预检 | 338 张量 / **0 个整块为零** / `ok = true` |
| 性能 | decode 64.80 → 162.77 tok/s（**+151.17%**），TTFT 22.0 → 13.5 ms（**−38.64%**） |
| 压缩比 | 2.712×（1,142,692,360 B） |
| 门禁 | `decision = sign`，`measurementValid = true` → 矩阵 `status = verified` |
| 闭环耗时 | 733.6 s（12.2 min，其中量化 410.6 s） |
| 端点拆解 | `cleanup.ok = true`，两个端点的 `no-managed-residual` 检查均通过（跑完 0 残留容器） |

**这就是之前"原理上不可判"的那个数字**：AWQ W4A16 在本机对 Qwen2.5-1.5B 的真实精度代价是
**困惑度 +19.57%**（15.34 → 18.34），换来解码吞吐 +151%。与"零损失"无关 ——
一个 4-bit 权重量化在 1.5B 模型上损失近 20% 困惑度，是完全合理的量级。

**与上一版的口径差异（如实声明）**：上一版校准集为 128 条、本轮为默认 **512 条**（其余参数相同）。
所以 19.571% 与上一版那个无效的 `0%` 之间**不可直接比较** ——
上一版那个数本身没有意义，两者之间不存在"退化"关系。

### 5.6 对 `/data/models` 的处理（刻意未做修改）

- **没有**修复或重下 `ms_cache` 里那份残缺文件，**也没有**改 `Qwen2.5-1.5B-Instruct` 这个软链
  —— 它是本次事故的现场证据，我无权替你销毁。
- 重跑用的是完好副本 `hf_cache/.../8b1c83b87f53a...` 复制到
  `/data/models/Qwen2.5-1.5B-Instruct-verified`（不覆写任何既有目录）。
- **建议你手工做一件事**（一行，可随时撤销）：
  ```bash
  # 方案 A：让规范路径重新指向完好副本
  ln -sfn /data/models/Qwen2.5-1.5B-Instruct-verified /data/models/Qwen2.5-1.5B-Instruct
  # 方案 B：重新完整下载，修好 ms_cache 本身
  modelscope download --model Qwen/Qwen2.5-1.5B-Instruct
  ```
  不做也行：新预检会在 3 秒内拦下它并告诉你原因，不会再烧 436 秒。

### 5.7 这次事故的正面价值

它暴露了一条**通用设计原则**，已带进 `inference-tuning` 的验收标准：

> **门禁不仅要判"数值是否达标"，还要判"测量是否成立"。**
> 一个沉默的坏测量比一个显式的失败更危险 —— 它会以"完美结果"的形态通过所有下游检查。

以及一条同样重要的推论：

> **"内容没被改动"与"内容是一个完整可用的东西"是两个问题。**
> 哈希校验只能回答前者；把前者当成后者，就会在**从未损坏过**的坏输入上一路绿灯。

---

## 六、复现方式

```bash
cd inference-quantization

npm test              # 129 用例，不需要 GPU/Docker
npm run check         # 81 项包规范 + 量化专项自检
node bin/inference-quantization.mjs doctor          # 环境/方法/镜像/契约自检
node bin/inference-quantization.mjs image verify    # 镜像四要素（扫描/签名会明确报告缺失）
node bin/inference-quantization.mjs dataset verify <dir>   # 去泄漏复核（宿主独立重算）
node bin/inference-quantization.mjs list --json     # 量化矩阵（五要素 + 完整性）
```

真机端到端（需 GPU + Docker + 已构建的量化镜像，约 8–15 分钟）：

```bash
node scripts/e2e-demo.mjs --model /data/models/Qwen2.5-1.5B-Instruct-verified \
    --accuracy-drop 50 --perf-gain 0 --calibration-samples 128
```

**只看"源模型预检"这一层**（3 秒，不需要 GPU）：

```bash
# 残缺副本 → 2.6 s 内 E_SOURCE_MODEL_INVALID，点名 223 个零张量
node bin/inference-quantization.mjs quantize /data/models/Qwen2.5-1.5B-Instruct \
    --max-accuracy-drop-pct 50 --min-perf-gain-pct 0 --allow-unverified-image --json

# 完好副本 → 预检通过并进入量化
node bin/inference-quantization.mjs quantize /data/models/Qwen2.5-1.5B-Instruct-verified \
    --max-accuracy-drop-pct 50 --min-perf-gain-pct 0 --allow-unverified-image --json
```

镜像构建（已在本机执行过）：

```bash
docker build -f docker/Dockerfile.quant \
  --build-arg BASE_IMAGE=$(cat docker/base-digest.txt) \
  -t inference-quantization/toolchain:0.1.0 .
```

---

## 七、证据边界

**站得住（本机一手实测）**

- 129 个测试用例、81 项包自检。
- **精度：ppl 15.3366 → 18.3381，退化 +19.571%**（wikitext-2，400 条 / 67,658 token；`runId=qr_45e79f22f441`）。
- **性能 +151.17% / TTFT −38.64%**（真实 vLLM 端点 × 2，真实负载）。
- 压缩比 2.712×、单档量化 410.6 s（512 条校准）、全闭环 733.6 s。
- **测量成立性**：`valid=true`、`bitIdenticalMeanNll=false`、`uniformGap=9.20`、逐条极差 3.367。
- 去泄漏（内容哈希交集为空 + 宿主独立重算）、源模型哈希前后一致、**源模型结构完整（338 张量 / 0 零张量）**、降级链四级触发 + QAT 计划。
- **性能评测端点拆解**：真机验证（真实 serving MCP + 两个真实 vLLM 端点）——`cleanup.ok=true`，
  两个端点的 `no-managed-residual` 检查均通过，跑完 `docker ps` 无托管残留容器（修复前每跑一次泄漏 2 个）。
- vLLM（compressed-tensors 0.17.0）能加载 llmcompressor 0.14.0 的 AWQ W4A16 产出（0.19.0 格式）并做真实前向 —— Step 0 §11。
- **根因**：残缺 `model.safetensors` 中 223/338 张量整块为零（2.145 GB 零字节，占数据区 69.5%），
  含 `model.norm.weight`；换用完好副本后同一脚本给出 ppl 12.88 的合理值。

**只是合理推断（未验证）**

- 残缺文件是**哪一次中断**造成的（目录里有 `*.aria2` 残留，但不是下载日志，无法定到具体时间/进程）。
- QAT 能否把 +19.56% 压回阈值内；27B 级模型的可量化性。

**原理上不可判（当前无法回答）**

- 原模型**本应**有的精度（无从得知——下载源头只给了这份残缺文件；
  完好副本来自另一条来源 `hf_cache`，两者首 802 MB 的字节并不相同，无法互相补全）。

**已作废、不得再引用的结论（保留在此以免误用）**

- `qr_25b26474a21b` 的 `dropPct = 0`、`perplexity = 151935.97`、`status=verified`；
- 上一版的性能数字 `+140.78% / −54.4%` 与压缩比上下文 —— 全部产生于残缺模型之上。

**与上一版的口径差异**

- 校准集 **512 条（本轮）vs 128 条（上一版）**，因重跑时未显式指定 `--calibration-samples`。
  这使本轮与上一版之间**不可做严格 A/B**；但上一版那个精度数字本身无效，二者不存在可比关系。
