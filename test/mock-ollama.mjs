/**
 * Scriptable mock of the Ollama HTTP surface used by the thinktune tests:
 * `/api/tags`, `/api/show`, `/api/chat` (NDJSON), and
 * `/v1/chat/completions` (SSE). Scenarios are selected per request model or
 * via the `POST /__scenario` control endpoint; recorded request bodies are
 * exposed through `GET /__dump` for wire-shape assertions.
 */
import { createServer } from 'node:http'

export function createMockOllama() {
  const state = {
    scenario: 'thinking',
    lastChatBody: undefined,
    lastOpenAIBody: undefined,
    chatBodies: [],
    openAIBodies: [],
    chatCalls: 0,
    openAICalls: 0,
    showCalls: 0,
  }

  const MODELS = {
    'qwen3:27b': {
      capabilities: ['completion', 'tools', 'thinking'],
      model_info: { 'qwen3.context_length': 40960 },
    },
    'qwen3.8:27b': {
      capabilities: ['completion', 'tools', 'thinking', 'vision'],
      model_info: { 'qwen3vl.context_length': 262144 },
    },
    'qwen3:custom-budget': {
      capabilities: ['completion', 'tools', 'thinking'],
      model_info: {},
    },
    'llama3:8b': {
      capabilities: ['completion', 'tools'],
      model_info: { 'llama3.context_length': 8192 },
    },
  }

  function ndjson(res, lines, dripMs = 0) {
    res.writeHead(200, { 'content-type': 'application/x-ndjson' })
    if (dripMs <= 0) {
      for (const line of lines) res.write(`${JSON.stringify(line)}\n`)
      res.end()
      return
    }
    let index = 0
    const timer = setInterval(() => {
      if (index < lines.length) {
        res.write(`${JSON.stringify(lines[index])}\n`)
        index += 1
      } else {
        clearInterval(timer)
        res.end()
      }
    }, dripMs)
  }

  function sse(res, chunks) {
    res.writeHead(200, { 'content-type': 'text/event-stream' })
    for (const chunk of chunks) res.write(`data: ${JSON.stringify(chunk)}\n\n`)
    res.write('data: [DONE]\n\n')
    res.end()
  }

  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://mock')
    const collect = async () => {
      let body = ''
      for await (const part of req) body += part
      return body
    }

    if (url.pathname === '/__scenario' && req.method === 'POST') {
      void collect().then((body) => {
        state.scenario = (JSON.parse(body)).scenario
        res.writeHead(204).end()
      })
      return
    }
    if (url.pathname === '/__dump') {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify(state))
      return
    }
    if (url.pathname === '/__reset') {
      state.lastChatBody = undefined
      state.lastOpenAIBody = undefined
      state.chatBodies = []
      state.openAIBodies = []
      state.chatCalls = 0
      state.openAICalls = 0
      res.writeHead(204).end()
      return
    }

    if (url.pathname === '/api/tags' && req.method === 'GET') {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({
        models: [
          { name: 'qwen3.8:27b', model: 'qwen3.8:27b' },
          { name: 'qwen3:27b', model: 'qwen3:27b' },
          { name: 'llama3:8b', model: 'llama3:8b' },
        ],
      }))
      return
    }

    if (url.pathname === '/api/show' && req.method === 'POST') {
      state.showCalls += 1
      void collect().then((body) => {
        const info = MODELS[(JSON.parse(body)).model]
        if (!info) {
          res.writeHead(400, { 'content-type': 'application/json' })
          res.end(JSON.stringify({ error: 'model not found' }))
          return
        }
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify(info))
      })
      return
    }

    if (url.pathname === '/api/chat' && req.method === 'POST') {
      state.chatCalls += 1
      void collect().then((body) => {
        const parsed = JSON.parse(body)
        state.lastChatBody = parsed
        if (state.chatBodies.length < 20) state.chatBodies.push(parsed)
        else state.chatBodies[state.chatCalls % 20] = parsed
        const scenario = state.scenario
        if (scenario === 'server-error') {
          res.writeHead(500, { 'content-type': 'application/json' })
          res.end(JSON.stringify({ error: 'ollama exploded' }))
          return
        }
        if (scenario === 'http-400') {
          res.writeHead(400, { 'content-type': 'application/json' })
          res.end(JSON.stringify({ error: '"qwen3:27b" does not support thinking' }))
          return
        }
        if (scenario === 'tool-call') {
          ndjson(res, [
            { model: parsed.model, message: { role: 'assistant', content: '' }, created_at: 0 },
            {
              model: parsed.model,
              message: {
                role: 'assistant',
                content: '',
                tool_calls: [{ function: { name: 'get_weather', arguments: { city: 'Hangzhou' } } }],
              },
              created_at: 0,
            },
            { model: parsed.model, message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop', prompt_eval_count: 91, eval_count: 17 },
          ])
          return
        }
        if (scenario === 'think-tags') {
          ndjson(res, [
            { model: parsed.model, message: { role: 'assistant', content: 'Let me <th' }, created_at: 0 },
            { model: parsed.model, message: { role: 'assistant', content: 'ink>2+2=4</think>So 4' }, created_at: 0 },
            { model: parsed.model, message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop', prompt_eval_count: 5, eval_count: 9 },
          ])
          return
        }
        if (scenario === 'error-mid') {
          ndjson(res, [
            { model: parsed.model, message: { role: 'assistant', content: 'par' }, created_at: 0 },
            { error: 'model runner has prematurely terminated' },
          ])
          return
        }
        if (scenario === 'length') {
          ndjson(res, [
            { model: parsed.model, message: { role: 'assistant', content: 'one two' }, created_at: 0 },
            { model: parsed.model, message: { role: 'assistant', content: '' }, done: true, done_reason: 'length', prompt_eval_count: 3, eval_count: 7 },
          ])
          return
        }
        if (scenario === 'no-thinking-capability') {
          ndjson(res, [
            { model: parsed.model, message: { role: 'assistant', content: 'plain' }, created_at: 0 },
            { model: parsed.model, message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' },
          ])
          return
        }
        if (scenario === 'slow') {
          ndjson(res, [
            { model: parsed.model, message: { role: 'assistant', thinking: 'slow start' }, created_at: 0 },
            { model: parsed.model, message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop', prompt_eval_count: 1, eval_count: 1 },
          ], 400)
          return
        }
        // Default 'thinking' scenario.
        ndjson(res, [
          { model: parsed.model, message: { role: 'assistant', thinking: 'The user greets' }, created_at: 0 },
          { model: parsed.model, message: { role: 'assistant', thinking: ' me; reply kindly.' }, created_at: 0 },
          { model: parsed.model, message: { role: 'assistant', content: 'Hello!' }, created_at: 0 },
          { model: parsed.model, message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop', prompt_eval_count: 12, eval_count: 34 },
        ])
      })
      return
    }

    if (url.pathname === '/v1/chat/completions' && req.method === 'POST') {
      state.openAICalls += 1
      void collect().then((body) => {
        const parsed = JSON.parse(body)
        state.lastOpenAIBody = parsed
        if (state.openAIBodies.length < 20) state.openAIBodies.push(parsed)
        const scenario = state.scenario
        if (scenario === 'sse-400') {
          res.writeHead(400, { 'content-type': 'application/json' })
          res.end(JSON.stringify({ error: { message: 'maximum context length exceeded' } }))
          return
        }
        if (scenario === 'sse-tool') {
          sse(res, [
            { choices: [{ delta: { role: 'assistant', tool_calls: [{ index: 0, id: 'call_abc', function: { name: 'get_weather', arguments: '{"city"' } }] } }] },
            { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: ':"Hangzhou"}' } }] } }] },
            { choices: [{ delta: {}, finish_reason: 'tool_calls' }] },
            { choices: [], usage: { prompt_tokens: 44, completion_tokens: 6 } },
          ])
          return
        }
        sse(res, [
          { choices: [{ delta: { role: 'assistant', reasoning_content: 'Careful…' } }] },
          { choices: [{ delta: { content: 'Hi there' } }] },
          { choices: [{ delta: { content: '!' }, finish_reason: 'stop' }] },
          { choices: [], usage: { prompt_tokens: 8, completion_tokens: 4 } },
        ])
      })
      return
    }

    res.writeHead(404, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ error: `no mock route for ${req.method} ${url.pathname}` }))
  })

  return { server, state }
}
