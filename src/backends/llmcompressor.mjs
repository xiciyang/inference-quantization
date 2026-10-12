/**
 * llmcompressor 方法族（内置）。
 *
 * 降级链与题面一致：INT4 → INT8 → FP8 → W8A16 →（QAT 计划）→ 回退
 *
 * 为什么是 llmcompressor 而不是 autoawq（Step 0 实测结论）：
 *   autoawq 在 PyPI 无 wheel，其内核包 autoawq-kernels 只有 x86_64/win_amd64 wheel，
 *   本机是 aarch64，装不上。而 llmcompressor 是 **py3-none-any 纯 Python**，
 *   其 AWQ 变换器只依赖 torch（已读 wheel 源码确认）—— 语义上仍是 "AWQ + INT4"。
 *
 * 方法对象刻意保持**纯声明 + 一次预检**：它不自己去起容器，
 * 只把 recipe 名字与约束交给 quantize.mjs 的执行器。这样"方法"与"执行"解耦，
 * 测试里可以用一个假方法跑完整闭环（不需要 GPU）。
 */
import { register } from './registry.mjs'
import { run } from '../util/proc.mjs'

/** 探测镜像里的 llmcompressor 是否可用（一次，缓存）。 */
const imageCache = new Map()
export async function probeQuantImage(runtime, { image }) {
  if (imageCache.has(image)) return imageCache.get(image)
  const script = 'import llmcompressor, compressed_tensors, transformers, torch, datasets, json;'
    + 'print(json.dumps({"ok":True,"llmcompressor":llmcompressor.__version__,"compressed_tensors":compressed_tensors.__version__,'
    + '"transformers":transformers.__version__,"torch":torch.__version__,"datasets":datasets.__version__}))'
  const r = await run('docker', ['run', '--rm', '--entrypoint', 'python3', image, '-c', script], {
    timeoutMs: Math.min(runtime.options.commandTimeoutMs * 4, 300_000),
  })
  let caps
  if (r.code === 0) {
    const line = r.stdout.trim().split('\n').filter(Boolean).pop() ?? ''
    try { caps = { ok: true, image, ...JSON.parse(line) } } catch { caps = { ok: false, image, reason: '输出解析失败', raw: line.slice(0, 300) } }
  } else {
    caps = { ok: false, image, reason: (r.stderr.trim() || r.spawnError?.message || '镜像探测失败').slice(0, 400) }
  }
  imageCache.set(image, caps)
  return caps
}

export function __resetImageCache() { imageCache.clear() }

const BASE_REQUIRES = { gpu: true, container: true, quantizer: 'llmcompressor' }

function makeMethod({ id, displayName, level, bits, scheme, transform, lossy, desc, inChain = true }) {
  return {
    id,
    displayName,
    level,
    bits,
    scheme,
    transform: transform ?? null,
    lossy: lossy === true,
    inDegradeChain: inChain,
    requires: BASE_REQUIRES,

    isAvailable(env) {
      return Boolean(env?.gpu?.present && env?.container?.available && env?.container?.nvidiaRuntime)
    },

    describe(env) {
      return { id, displayName, level, bits, scheme, transform, lossy, gpu: env?.gpu?.gpus?.[0]?.name ?? null }
    },

    /** 配方声明（真正的构造在容器内的 quantize.py 里，按 id 分派）。 */
    recipe() {
      return { methodId: id, transform, scheme, bits, lossy }
    },

    /** 预检：在起容器之前判断这个档位值不值得试。 */
    async preflight({ env, sourceBytes = null }) {
      const caps = await probeQuantImage({ options: { commandTimeoutMs: 60_000 } }, { image: 'inference-quantization/toolchain:0.1.0' })
      if (!caps.ok) {
        return {
          ok: false, code: 'E_METHOD_UNAVAILABLE',
          message: `量化镜像不可用：${caps.reason}`,
          remediation: '先构建量化镜像（inference-quantization image build），或修正 options.quantImage。',
          details: caps,
        }
      }
      const free = env?.gpu?.gpus?.[0]?.availableBytes ?? env?.capability?.availableMemoryBytes ?? null
      if (free !== null && sourceBytes !== null && sourceBytes * 2.5 > free) {
        return {
          ok: false, code: 'E_METHOD_UNAVAILABLE',
          message: `预计内存不足完成 PTQ：源模型 ${(sourceBytes / 1024 ** 3).toFixed(2)} GiB，`
            + `按 2.5× 峰值估算需 ${((sourceBytes * 2.5) / 1024 ** 3).toFixed(2)} GiB，宿主可用仅 ${(free / 1024 ** 3).toFixed(2)} GiB`,
          remediation: '换更小的模型；或先释放内存（同一台机上另有推理服务在跑）。',
          details: { sourceBytes, estimatedPeakBytes: sourceBytes * 2.5, availableBytes: free },
        }
      }
      return { ok: true, details: { image: caps, estimatedPeakBytes: sourceBytes !== null ? sourceBytes * 2.5 : null } }
    },
  }
}

/** 降级链第 1 档 —— 题面默认：AWQ + INT4 */
register(makeMethod({
  id: 'awq-w4a16', displayName: 'AWQ 变换 + W4A16（INT4，group-wise）',
  level: 1, bits: 4, scheme: 'W4A16', transform: 'AWQ', lossy: true,
  desc: '精度优先：4bit 权重 + FP16 激活，AWQ 做权重平滑',
}))

/** 第 2 档 —— INT8 W8A8 */
register(makeMethod({
  id: 'int8-w8a8', displayName: 'INT8 W8A8',
  level: 2, bits: 8, scheme: 'INT8', transform: null, lossy: true,
  desc: '权重与激活都 8bit，精度损失更小但收益也小',
}))

/** 第 3 档 —— FP8 dynamic */
register(makeMethod({
  id: 'fp8-dynamic', displayName: 'FP8 dynamic',
  level: 3, bits: 8, scheme: 'FP8_DYNAMIC', transform: null, lossy: true,
  desc: 'FP8 动态量化；对 Hopper/Blackwell 有原生支持',
}))

/** 第 4 档 —— W8A16 仅权重 */
register(makeMethod({
  id: 'w8a16', displayName: 'W8A16（仅权重 8bit）',
  level: 4, bits: 8, scheme: 'W8A16', transform: null, lossy: true,
  desc: '最保守的一档：只压权重，激活保持 FP16',
}))

export const DEGRADE_CHAIN_IDS = ['awq-w4a16', 'int8-w8a8', 'fp8-dynamic', 'w8a16']
