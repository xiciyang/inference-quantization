/**
 * inference-serving 的 MCP 客户端。
 *
 * 题面要求"部署验证通过 MCP 调用 inference-serving"，所以这里**真的走 MCP 协议**
 * （stdio JSON-RPC 2.0），而不是去 import 它的内部模块 —— 后者会把两个包的内部结构
 * 焊死在一起，也绕过了 serving 自己的能力清单/风险闸门/审计。
 *
 * 定位 inference-serving 的顺序（可复现、不猜）：
 *   1. `options.servingPath`（显式指定，最高优先）
 *   2. `~/.ai-infra/config/serving-path.txt`（一次写死，长期可用）
 *   3. 本包的同级目录 `../inference-serving`（开发布局）
 *   4. `node_modules/inference-serving`（作为依赖安装时）
 * 全部失败 → E_SERVING_UNAVAILABLE，附上探测过的路径。
 */
import { spawn } from 'node:child_process'
import { promises as fs } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'
import { fail, ERROR_CODES } from './errors.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG_ROOT = resolve(HERE, '..')

/**
 * 定位 inference-serving 的 MCP 入口。
 *
 * ## 返回形状是**契约**（见 inference-contracts/contracts/serving-client-contract.md）
 *
 * 本函数在 `inference-quantization` 与 `inference-tuning` 里各有一份实现。
 * 曾经两份返回**不同的形状**（`{path,how,tried:string[]}` vs `{path,from,root,tried:object[]}`），
 * 而 `path` 一个指"包根目录"、一个指"server.mjs 文件" —— 任何共享消费方
 * （未来的 @ai-infra/testing-core、编排器）都没法一视同仁。
 *
 * 现在两边统一为：
 *   `{ serverPath: <server.mjs 的绝对路径>, root: <包根目录>, from: <命中方式>, tried: [{path, from}] }`
 * 字段名没有歧义：**要 spawn 的就是 serverPath**。
 */
export async function resolveServingPath(runtime) {
  const tried = []
  const candidates = []
  if (runtime.options.servingPath) candidates.push({ path: runtime.options.servingPath, from: 'options.servingPath' })
  candidates.push({ path: join(homedir(), '.ai-infra', 'config', 'serving-path.txt'), from: 'config file', isFile: true })
  candidates.push({ path: resolve(PKG_ROOT, '..', 'inference-serving'), from: 'sibling directory' })
  candidates.push({ path: resolve(PKG_ROOT, 'node_modules', 'inference-serving'), from: 'node_modules' })

  for (const c of candidates) {
    let p = c.path
    tried.push({ path: p, from: c.from })
    if (c.isFile) {
      try {
        const txt = (await fs.readFile(p, 'utf8')).trim()
        if (txt) p = txt
        else continue
      } catch { continue }
    }
    const serverPath = join(p, 'mcp', 'server.mjs')
    try {
      await fs.access(serverPath)
      return { serverPath, root: p, from: c.from, tried }
    } catch { /* 继续 */ }
  }
  fail(ERROR_CODES.E_SERVING_UNAVAILABLE, '找不到 inference-serving（它提供部署验证）', {
    remediation: `用 options.servingPath 指定，或把路径写入 ~/.ai-infra/config/serving-path.txt。已探测：${tried.map((t) => t.path).join(' / ')}`,
    details: { tried },
  })
}

/**
 * 客户端接缝的统一入口解析：既接受 server.mjs 文件，也接受包根目录。
 * （契约规定传文件；容错接受目录是为了让旧调用点不必同时改。）
 */
export function resolveServerEntry(serverPath) {
  return serverPath.endsWith('.mjs') ? serverPath : join(serverPath, 'mcp', 'server.mjs')
}

/**
 * 一个极简 MCP stdio 客户端：send(method, params) → result。
 * 按行分隔的 JSON-RPC 2.0，与 mcp/server.mjs 的约定一致。
 */
export class ServingMcpClient {
  constructor({ serverPath, env = process.env, timeoutMs = 120_000 }) {
    this.serverPath = serverPath
    this.env = env
    this.timeoutMs = timeoutMs
    this.nextId = 1
    this.pending = new Map()
    this.buffer = ''
    this.child = null
    this.stderr = ''
    this.closed = false
  }

  async start() {
    this.child = spawn(process.execPath, [resolveServerEntry(this.serverPath)], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...this.env, INFERENCE_SERVING_ROOT: this.env.INFERENCE_SERVING_ROOT ?? undefined },
    })
    this.child.stdout.on('data', (chunk) => this.#onData(chunk))
    this.child.stderr.on('data', (chunk) => { this.stderr += chunk.toString('utf8').slice(0, 4000) })
    this.child.on('error', (err) => this.#failAll(err))
    this.child.on('exit', (code) => {
      if (!this.closed) this.#failAll(new Error(`serving MCP server 退出，code=${code}；stderr=${this.stderr.slice(-800)}`))
    })
    // 握手
    await this.request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'inference-quantization', version: '0.1.0' },
    })
    this.notify('notifications/initialized', {})
    return this
  }

  #onData(chunk) {
    this.buffer += chunk.toString('utf8')
    let idx
    while ((idx = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, idx).trim()
      this.buffer = this.buffer.slice(idx + 1)
      if (!line) continue
      let msg
      try { msg = JSON.parse(line) } catch { continue }
      const p = this.pending.get(msg.id)
      if (!p) continue
      this.pending.delete(msg.id)
      clearTimeout(p.timer)
      if (msg.error) p.reject(new Error(`MCP error ${msg.error.code}: ${msg.error.message}`))
      else p.resolve(msg.result)
    }
  }

  #failAll(err) {
    for (const [, p] of this.pending) { clearTimeout(p.timer); p.reject(err) }
    this.pending.clear()
  }

  request(method, params) {
    const id = this.nextId++
    return new Promise((resolvePromise, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`MCP 请求超时（${method}，${this.timeoutMs}ms）`))
      }, this.timeoutMs)
      timer.unref?.()
      this.pending.set(id, { resolve: resolvePromise, reject, timer })
      this.child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
    })
  }

  notify(method, params) {
    this.child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method, params })}\n`)
  }

  /** 调用一个工具，返回**解析后的结构化结果**（serving 的工具返回 JSON 文本）。 */
  async callTool(name, args = {}) {
    const res = await this.request('tools/call', { name, arguments: args })
    const text = (res?.content ?? []).map((c) => c.text ?? '').join('')
    let value = null
    try { value = JSON.parse(text) } catch { value = { raw: text } }
    // 契约：`raw` 一律是**文本**（两份实现同形），完整 MCP 响应放在 `response`。
    // 曾经这里给的是响应对象、tuning 给的是文本，调用方没法统一处理。
    return { isError: res?.isError === true, value, raw: text, response: res }
  }

  async listTools() {
    const res = await this.request('tools/list', {})
    return res?.tools ?? []
  }

  async close() {
    this.closed = true
    try { this.child.stdin.end() } catch { /* 已关 */ }
    await new Promise((r) => setTimeout(r, 50))
    try { this.child.kill('SIGTERM') } catch { /* 已退 */ }
  }
}

/** 便捷：开一个客户端，跑 fn，然后一定关掉。 */
export async function withServing(runtime, fn, { timeoutMs = 300_000 } = {}) {
  const found = await resolveServingPath(runtime)
  const client = new ServingMcpClient({ serverPath: found.serverPath, env: runtime.env, timeoutMs })
  await client.start()
  try {
    return await fn(client, found)
  } finally {
    await client.close()
  }
}

/**
 * 令牌累积重放 —— 复刻 serving 的确认流程契约。
 *
 * serving 有两道门（R2 风险确认 + license 确认），各自的令牌不同，且要用**同一个
 * `details.confirmToken` 字段**回传。这里统一累积，避免每个调用点各写一遍。
 */
export async function callWithConfirm(client, name, args, { maxRounds = 6, log = null } = {}) {
  const tokens = Array.isArray(args.confirm) ? [...args.confirm] : (args.confirm ? [args.confirm] : [])
  let last = null
  for (let round = 0; round < maxRounds; round += 1) {
    // eslint-disable-next-line no-await-in-loop
    const out = await client.callTool(name, { ...args, confirm: tokens })
    last = out
    if (!out.isError && out.value?.ok !== false) return out
    const token = out.value?.details?.confirmToken
    if (!token || tokens.includes(token)) return out
    log?.info(`serving 要求确认：${out.value?.code}`, { tool: name, token })
    tokens.push(token)
  }
  return last
}

export { spawn }
