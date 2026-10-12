/**
 * 版本常量（与 package.json / manifest.yaml 三处一致，由 scripts/check-manifest.mjs 断言）。
 */
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

export const PACKAGE_NAME = 'inference-quantization'
export const PACKAGE_VERSION = '0.1.0'

/** MCP 工具接口语义化版本。破坏性变更升 major。 */
export const MCP_INTERFACE_VERSION = '1.0.0'
export const CLI_INTERFACE_VERSION = '1.0.0'

/** 量化矩阵文件格式版本。 */
export const MATRIX_FORMAT_VERSION = 1
/** 审计记录格式版本。 */
export const AUDIT_FORMAT_VERSION = 1
/** 数据切分血缘格式版本。 */
export const LINEAGE_FORMAT_VERSION = 1

export const PACKAGE_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

export function readManifestText() {
  return readFileSync(join(PACKAGE_ROOT, 'manifest.yaml'), 'utf8')
}
