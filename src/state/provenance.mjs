/**
 * 产物溯源边车（F-4）。
 *
 * ## 为什么有这个文件
 *
 * 量化产物（`~/.ai-infra/quantized/<name>/`）里只有**量化配方**（`recipe.yaml`），
 * **没有来源模型信息**。而 serving 只按产物路径查 license：
 *
 *   - 基座 `/data/models/Qwen2.5-1.5B-Instruct-verified` → 无 LICENSE 文件 → `unknown`
 *     → 人工确认一次（`lic_109655b7c0c4cc00`）
 *   - 派生 `~/.ai-infra/quantized/...-awq-w4a16` → 同样无 LICENSE → **又是** `unknown`
 *     → **再要求人工确认一次**（`lic_231e62b4935bd42a`，真机审计 seq 14 实测）
 *
 * 后果：headless 下每个派生模型都撞 `E_HEADLESS_NEEDS_CONFIRMATION`，链路卡死；
 * 非 headless 下也要人反复点 —— 而基座的 license **早已确认过**。
 *
 * 溯源数据本来就是现成的（矩阵里有 `outputPath → sourceModel`），只是没落到**产物**里。
 * 这个边车把它落下来，serving 据此把基座的确认沿链继承。
 *
 * ## 契约
 *
 * 文件名 `ai-infra-provenance.json`，schema `ai-infra/provenance@1`。
 * 它是**跨包契约的一部分**：serving 会读它（见 `inference-contracts/contracts/serving-contract.md`）。
 * 缺失时 serving 退回"按路径查 license"的旧行为，不会更糟。
 */
import { promises as fs } from 'node:fs'
import { join } from 'node:path'
import { atomicWriteJson } from '../util/fsx.mjs'

export const PROVENANCE_FILE = 'ai-infra-provenance.json'
export const PROVENANCE_SCHEMA = 'ai-infra/provenance@1'

/** 读一个产物目录里的溯源边车；没有或损坏则返回 null（**不猜**）。 */
export async function readProvenance(dir) {
  try {
    const raw = await fs.readFile(join(dir, PROVENANCE_FILE), 'utf8')
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return null
    return parsed
  } catch {
    return null
  }
}

/**
 * 写溯源边车。**失败不应让量化本身失败** —— 调用方负责吞掉并留 warning。
 */
export async function writeProvenance(runtime, {
  outputPath,
  runId,
  standard,
  method = null,
  level = null,
  sourceRunKey = null,
  extra = {},
}) {
  const doc = {
    schema: PROVENANCE_SCHEMA,
    runId: runId ?? null,
    sourceRunKey: sourceRunKey ?? null,
    /** 派生自哪个模型目录 —— serving 据此沿链继承 license 确认 */
    sourceModel: standard?.source?.path ?? standard?.source?.display ?? null,
    /** 基座自己声明的 license（可能为 null：本机两个模型都没有 LICENSE 文件） */
    sourceLicense: standard?.source?.license
      ? { license: standard.source.license.license ?? null, class: standard.source.license.class ?? null }
      : null,
    method: method ?? null,
    level: level ?? null,
    createdAt: new Date().toISOString(),
    note: '本文件由 inference-quantization 写入，供推理侧其它插件（尤其是 serving 的 license 解析）沿链继承溯源。手工修改会导致 license 判定失真。',
    ...extra,
  }
  await atomicWriteJson(join(outputPath, PROVENANCE_FILE), doc)
  return doc
}
