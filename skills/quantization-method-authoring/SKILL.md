---
name: quantization-method-authoring
description: 当需要给 inference-quantization 新增一种量化方法（新 bit 宽、新方案、新工具链）时使用。说明方法注册表的契约、降级链档位如何声明、以及必须在容器侧同步的改动点。
---

# 新增量化方法

"量化方法插件化"是本插件的核心原则：**核心流程不认识任何具体方法**。
新增方法 = 加一个声明 + 在容器脚本里加一条 recipe，**核心零改动**。

## 两处改动（缺一不可）

### ① JS 侧：注册方法（`src/backends/`）

```js
import { register } from './registry.mjs'

register({
  id: 'my-method',            // 唯一 id，出现在 MCP 参数与矩阵里
  displayName: '我的方法',
  level: 5,                   // 降级链档位（≥1；越小越优先）
  bits: 4,                    // 权重位宽（矩阵里会用）
  scheme: 'W4A16',            // 交给容器侧的方案名
  transform: 'AWQ',           // 可选的变换器
  lossy: true,                // 是否**有损**（有损档会在报告里显式标注）
  isAvailable(env) { return Boolean(env?.gpu?.present && env?.container?.available && env?.container?.nvidiaRuntime) },
  describe(env) { return { id: 'my-method', level: 5, bits: 4, lossy: true } },
  recipe() { return { methodId: 'my-method', transform: 'AWQ', scheme: 'W4A16', bits: 4, lossy: true } },
  async preflight({ env, sourceBytes }) { return { ok: true, details: {} } },
})
```

然后在 `src/backends/index.mjs` 里加一行 `import './my-method.mjs'`。

**注册表会拒绝**：缺 `recipe`/`describe`/`preflight`、`level` 不是 ≥1 的整数、id 重复。

### ② Python 侧：加 recipe（`docker/scripts/quantize.py`）

```python
if method == "my-method":
    from llmcompressor.modifiers.quantization import QuantizationModifier
    return [QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])], "我的方法说明"
```

并把 id 加进 `METHODS` 元组（argparse 的 choices 会据此校验）。

## 必须验证的四件事

1. **产出能被 vLLM 加载并推理** —— 不是"文件写出来了"就算。用推理镜像起一次端点并做真实前向。
2. **产出格式与推理镜像的压缩库版本兼容** —— 本机已知：量化镜像用 compressed-tensors 0.19.0，
   推理镜像用 0.17.0，实测**能读**（见 docs/step0/PROJECT_CONTEXT.md §11）。
   新方法换了 format（如 mxfp4）时必须重新验证。
3. **aarch64 可用** —— 本机是 aarch64。任何依赖 x86 预编译 kernel 的方案（如 `autoawq-kernels`）
   在本机不可用；优先选纯 Python 路径。
4. **降级链档位不与既有冲突** —— `level` 唯一，且顺序应符合"损失小的在前"。

## 测试要求

- 注册表单测：形状校验、重复注册拒绝、`chainOrdered()` 顺序正确；
- **假方法（test double）跑完整闭环**：不需要 GPU 就能验证降级链会把新档位正确纳入；
- 若新方法引入新的失败码，必须加进 `src/degrade.mjs` 的 `TERMINAL_FAILURES` 或明确其可降级性。
