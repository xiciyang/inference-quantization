/**
 * 测试替身服务：一个最小的 OpenAI 兼容端点。
 * 被测代码不认识它 —— 它只是"某个后端拉起来的服务"。
 */
import http from 'node:http'

const argv = process.argv.slice(2)
const get = (k, d) => {
  const i = argv.indexOf(k)
  return i >= 0 ? argv[i + 1] : d
}

const port = Number(get('--port', '0'))
const name = get('--name', 'fake-model')
const mode = get('--mode', 'ok') // ok | slow | broken

const server = http.createServer((req, res) => {
  if (req.url === '/v1/models') {
    if (mode === 'broken') { res.writeHead(500).end('{}'); return }
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ object: 'list', data: [{ id: name, object: 'model', root: '/fake' }] }))
    return
  }
  if (req.url === '/v1/completions' && req.method === 'POST') {
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({
        id: 'cmpl-fake',
        object: 'text_completion',
        model: name,
        choices: [{ index: 0, text: ' Paris', finish_reason: 'length' }],
      }))
    })
    return
  }
  res.writeHead(404).end('{}')
})

if (mode === 'slow') {
  setTimeout(() => server.listen(port, '127.0.0.1'), 60_000)
} else {
  server.listen(port, '127.0.0.1', () => {
    process.stdout.write(`fixture listening on ${port}\n`)
  })
}

process.on('SIGTERM', () => process.exit(0))
