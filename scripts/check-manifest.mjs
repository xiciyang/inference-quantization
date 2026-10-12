#!/usr/bin/env node
/**
 * 包规范自检（`npm run check`）。
 *
 * 为什么不放 `prepack`：题目要求「prepare / postinstall 不执行危险操作」。
 * 最安全的做法是**安装期零脚本**，把检查交给显式的 `npm run check`。
 *
 * 检查项：
 *   1. package.json 必需字段
 *   2. engines / os / cpu 已声明
 *   3. files 白名单覆盖所有运行期必需文件
 *   4. dependencies 为空（零运行时依赖是本包的硬约束）
 *   5. 版本号与 manifest.yaml / src/version.mjs 三处一致
 *   6. manifest.yaml 可被本包自己的受限 YAML 解析器解析
 *   7. 没有把 tests/scripts 打进包里
 */
import { readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parse } from '../src/util/yaml.mjs'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const failures = []
const notes = []

function check(cond, message) {
  if (cond) notes.push(`  ok   ${message}`)
  else failures.push(`  FAIL ${message}`)
}

const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))

// 1
for (const f of ['name', 'version', 'description', 'license', 'type', 'main', 'bin', 'exports', 'files']) {
  check(pkg[f] !== undefined, `package.json 有字段 ${f}`)
}
check(pkg.type === 'module', 'package.json type=module（DSH 插件要求 ESM）')

// 2
check(pkg.engines?.node, `声明 engines.node（${pkg.engines?.node}）`)
check(Array.isArray(pkg.os) && pkg.os.length > 0, `声明 os（${JSON.stringify(pkg.os)}）`)
check(Array.isArray(pkg.cpu) && pkg.cpu.length > 0, `声明 cpu（${JSON.stringify(pkg.cpu)}）`)

// 3
const requiredInFiles = ['bin/', 'src/', 'mcp/', 'skills/', 'docker/', 'manifest.yaml', 'README.md', 'LICENSE', 'cordis.patch.yml']
for (const r of requiredInFiles) {
  check(pkg.files?.includes(r), `files 白名单包含 ${r}`)
}
const mustExist = ['README.md', 'LICENSE', 'manifest.yaml', 'cordis.patch.yml', 'bin/inference-quantization.mjs', 'mcp/server.mjs', 'src/index.mjs', 'src/api.mjs']
for (const m of mustExist) {
  check(existsSync(join(root, m)), `文件存在：${m}`)
}

// 4 —— 零依赖是硬约束（题目第七节「不得引入未声明依赖」的更强形式）
const deps = Object.keys(pkg.dependencies ?? {})
check(deps.length === 0, `dependencies 为空（当前 ${deps.length} 个：${deps.join(', ') || '无'}）`)

// 5
const manifest = parse(readFileSync(join(root, 'manifest.yaml'), 'utf8'))
const versionSrc = readFileSync(join(root, 'src/version.mjs'), 'utf8')
const m = /PACKAGE_VERSION\s*=\s*'([^']+)'/.exec(versionSrc)
check(m && m[1] === pkg.version, `src/version.mjs 版本与 package.json 一致（${m?.[1]} vs ${pkg.version}）`)
check(manifest?.version === pkg.version, `manifest.yaml 版本与 package.json 一致（${manifest?.version} vs ${pkg.version}）`)
check(manifest?.name === pkg.name, `manifest.yaml name 与 package.json 一致（${manifest?.name} vs ${pkg.name}）`)

// 6 —— manifest.yaml 结构
check(Array.isArray(manifest?.capabilities) && manifest.capabilities.length > 0, `manifest.yaml 声明了 ${manifest?.capabilities?.length ?? 0} 个 capability`)
const declaredIds = (manifest?.capabilities ?? []).map((c) => c.id)
for (const expected of ['quantize', 'get_report', 'list', 'rollback', 'qat_plan', 'dataset_prepare', 'image_verify', 'image_build']) {
  check(declaredIds.includes(expected), `manifest 声明了 capability: ${expected}`)
}
// 每个 capability 必须声明 resources（最小权限）
for (const c of manifest?.capabilities ?? []) {
  check(Array.isArray(c.resources) && c.resources.length > 0, `capability ${c.id} 声明了 resources`)
  check(['read', 'write', 'system'].includes(c.risk), `capability ${c.id} 的 risk 合法（${c.risk}）`)
}

// 7
check(!pkg.files?.some((f) => f.startsWith('tests') || f.startsWith('scripts')), 'files 白名单不含 tests/ 或 scripts/')

// DSH bundle 清单
check(pkg.dsh?.bundle?.patch === './cordis.patch.yml', 'package.json 声明 dsh.bundle.patch')

// 量化专项检查
const dockerfile = existsSync(join(root, 'docker/Dockerfile.quant')) ? readFileSync(join(root, 'docker/Dockerfile.quant'), 'utf8') : ''
check(/BASE_IMAGE=\S+/.test(dockerfile), 'Dockerfile 声明 BASE_IMAGE（可复现构建）')
check(!/^FROM\s+(?!\$\{BASE_IMAGE\})/m.test(dockerfile) || /FROM\s+\$\{BASE_IMAGE\}/.test(dockerfile), 'Dockerfile 从 BASE_IMAGE 派生（不硬编码 tag）')
const digest = existsSync(join(root, 'docker/base-digest.txt')) ? readFileSync(join(root, 'docker/base-digest.txt'), 'utf8').trim() : ''
check(/^\S+@sha256:[0-9a-f]{64}$/.test(digest), `base-digest.txt 是固定 digest（${digest.slice(0, 40)}…）`)
const gatesSrc = readFileSync(join(root, 'src/policy-defaults.mjs'), 'utf8')
check(/max_accuracy_drop_pct:\s*null/.test(gatesSrc), '门禁阈值**没有**内置默认值（裁决：不猜默认）')
const degradeSrc = readFileSync(join(root, 'src/degrade.mjs'), 'utf8')
check(/TERMINAL_FAILURES/.test(degradeSrc), '降级链声明了不可降级失败集合')
const errorsSrc = readFileSync(join(root, 'src/errors.mjs'), 'utf8')
check(/E_CALIB_LEAKAGE_DETECTED/.test(errorsSrc), '存在去泄漏失败码')
check(/E_SOURCE_MODEL_MODIFIED/.test(errorsSrc), '存在源模型被改失败码')
// --- 测量成立性（用户裁决：门禁不仅要判数值是否达标，还要判测量是否成立）---
check(/E_SOURCE_MODEL_INVALID/.test(errorsSrc), '存在源模型结构残缺失败码（区别于"被改动"）')
check(/E_MEASUREMENT_INVALID/.test(errorsSrc), '存在测量不成立失败码')
check(/E_SOURCE_MODEL_INVALID/.test(degradeSrc), '源模型残缺被判为不可降级（换档位解决不了）')
check(/E_MEASUREMENT_INVALID/.test(degradeSrc), '测量不成立被判为不可降级（不能靠换档位绕过）')
const gateSrc = readFileSync(join(root, 'src/gate.mjs'), 'utf8')
check(/export function assertMeasurementValid/.test(gateSrc), '门禁里有独立的测量成立性判定（与阈值判定并列）')
check(/uniform-output/.test(gateSrc) && /indiscriminate/.test(gateSrc) && /no-item-variance/.test(gateSrc),
  '测量成立性覆盖三条否证：均匀输出 / 无区分力 / 与输入无关')
const quantizeSrc = readFileSync(join(root, 'src/quantize.mjs'), 'utf8')
check(/assertMeasurementValid\(/.test(quantizeSrc), '闭环里真的调用了测量成立性判定')
check(/assertSourceModelValid\(/.test(quantizeSrc), '闭环里真的调用了源模型结构预检')
check(/measurementValid !== false/.test(quantizeSrc), 'verified 状态额外要求测量成立（缺一不可）')
const parityPy = readFileSync(join(root, 'docker/scripts/parity.py'), 'utf8')
check(/def param_health/.test(parityPy), '精度执行端采集权重健康度（权重是否真的加载）')
check(/"validity"/.test(parityPy) && /uniform-output/.test(parityPy), '精度执行端输出测量成立性证据')
const perfSrc = readFileSync(join(root, 'src/testing/perf.mjs'), 'utf8')
check(/inference_serving_teardown/.test(perfSrc), '性能评测端点跑完会拆（真机曾泄漏 2 个容器）')
check(/finally/.test(perfSrc) && /W_RESIDUAL_CLEANUP/.test(perfSrc),
  '测量异常路径也拆，且拆解失败如实记账（不静默）')

process.stdout.write(`inference-quantization 包规范自检\n${notes.join('\n')}\n`)
if (failures.length > 0) {
  process.stdout.write(`\n${failures.length} 项失败：\n${failures.join('\n')}\n`)
  process.exit(1)
}
process.stdout.write(`\n全部通过（${notes.length} 项）\n`)
