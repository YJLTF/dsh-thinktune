/**
 * thinktune test suite: drives `OllamaThinkAdapter` directly against the mock
 * Ollama, then mounts the REAL `LlmRuntime` from @deepseek-ai/dsh-llm to
 * verify service-side effort validation, default materialization, and chunk
 * assembly end-to-end.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { LlmRuntime, LlmError, createUserMessage, createSystemMessage, offloadedImageText } from '@deepseek-ai/dsh-llm'
import { Context } from '@deepseek-ai/cordis'
import { OllamaThinkAdapter } from '../src/adapter.ts'
import { resolveConfig } from '../src/config.ts'
import { parseEfforts } from '../src/efforts.ts'
import { createMockOllama } from './mock-ollama.mjs'

const warns = []
const logger = { warn: (m) => warns.push(String(m)), info: () => {} }

let BASE = 'http://127.0.0.1:11434'

/** Minimal attachment-store stand-in: fixed request versions + call recording. */
function fakeAttachmentStore(refs) {
  const requests = []
  return {
    requests,
    readImageRequest: async (ref, policy) => {
      requests.push({ attachmentId: ref.attachmentId, policy })
      const spec = refs.get(ref.attachmentId) ?? { mediaType: 'image/png', bytes: [1, 2, 3] }
      return {
        variantId: `v_${ref.attachmentId}`,
        attachment: ref,
        data: Uint8Array.from(spec.bytes),
        mediaType: spec.mediaType,
        bytes: spec.bytes.length,
        width: 64,
        height: 64,
        depth: 'uchar',
        space: 'srgb',
        hasAlpha: false,
      }
    },
  }
}

function imageRef(id, extra = {}) {
  return { attachmentId: id, mediaType: 'image/png', bytes: 100, width: 64, height: 64, ...extra }
}

function imageMessage(text, refs) {
  return createUserMessage({
    content: [{ type: 'text', text }, ...refs.map((attachment) => ({ type: 'image', attachment }))],
    source: { kind: 'user' },
  })
}

function makeAdapter(overrides = {}, effortsOverride = undefined, resolveAttachments = undefined) {
  const cfg = resolveConfig({
    endpoint: BASE,
    providers: ['ollama'],
    strategy: 'native',
    nativeLevels: false,
    offSentinel: 'none',
    assumeThinking: 'auto',
    defaultContextWindow: 32768,
    defaultMaxTokens: 8192,
    streamIdleTimeoutMs: 5000,
    historyThinking: 'strip',
    models: [],
    imageCapability: 'auto',
    imageMaxPixels: 1024 * 1024,
    imageMaxBytes: 4 * 1024 * 1024,
    imageMaxPerRequest: 8,
    imageMaxRequestBytes: 32 * 1024 * 1024,
    ...overrides,
  })
  const efforts = effortsOverride ?? parseEfforts(undefined, logger.warn)
  return new OllamaThinkAdapter(cfg, efforts, logger, resolveAttachments)
}

function userRequest(options) {
  return {
    provider: 'ollama',
    model: 'qwen3:27b',
    messages: [createUserMessage({ content: [{ type: 'text', text: 'Hi' }], source: { kind: 'user' } })],
    ...options,
  }
}

async function collect(adapter, options) {
  const chunks = []
  for await (const chunk of adapter.stream(options)) chunks.push(chunk)
  return chunks
}

async function setScenario(server, scenario) {
  const port = server.address().port
  await fetch(`http://127.0.0.1:${port}/__scenario`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ scenario }),
  })
}

const BLOCK_END = (chunks, index) => chunks.find((c) => c.type === 'block-end' && c.index === index)?.block

async function startMock() {
  const { server, state } = createMockOllama()
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  BASE = `http://127.0.0.1:${port}`
  return { server, state }
}

test('resolveModel advertises thinking efforts from /api/show capabilities', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter()
    const info = await adapter.resolveModel('ollama', 'qwen3:27b')
    assert.equal(info.provider, 'ollama')
    assert.equal(info.context.contextWindow, 40960)
    assert.equal(info.defaultMaxTokens, 8192)
    assert.deepEqual(info.reasoning.efforts.map((effort) => effort.id), ['off', 'low', 'medium', 'high'])
    assert.equal(info.reasoning.defaultEffort, undefined)
  } finally {
    server.close()
  }
})

test('resolveModel omits reasoning for models without the thinking capability', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter()
    const info = await adapter.resolveModel('ollama', 'llama3:8b')
    assert.equal(info.reasoning, undefined)
    assert.equal(info.context.contextWindow, 8192)
  } finally {
    server.close()
  }
})

test('resolveModel falls back to configured capacity when /api/show fails', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter({ defaultContextWindow: 12345 })
    const info = await adapter.resolveModel('ollama', 'not-a-model')
    assert.equal(info.name, 'not-a-model')
    assert.equal(info.context.contextWindow, 12345)
    // assumeThinking auto: show failed → efforts still advertised.
    assert.equal(info.reasoning.efforts.length, 4)
  } finally {
    server.close()
  }
})

test('resolveModel honors assumeThinking: no', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter({ assumeThinking: 'no' })
    const info = await adapter.resolveModel('ollama', 'qwen3:27b')
    assert.equal(info.reasoning, undefined)
  } finally {
    server.close()
  }
})

test('resolveModel advertises the configured defaultEffort', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter({ defaultEffort: 'low' })
    const info = await adapter.resolveModel('ollama', 'qwen3:27b')
    assert.equal(info.reasoning.defaultEffort, 'low')
  } finally {
    server.close()
  }
})

test('listModels merges /api/tags with configured overrides', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter({ models: [{ id: 'qwen3:27b', name: 'Qwen3 27B', contextWindow: 100 }, { id: 'custom:1' }] })
    const models = await adapter.listModels('ollama')
    const qwen = models.find((model) => model.id === 'qwen3:27b')
    assert.equal(qwen.name, 'Qwen3 27B')
    assert.ok(models.some((model) => model.id === 'llama3:8b'))
    assert.ok(models.some((model) => model.id === 'custom:1'))
  } finally {
    server.close()
  }
})

test('native strategy: effort off sends think:false and assembles blocks', async () => {
  const { server, state } = await startMock()
  try {
    await setScenario(server, 'thinking')
    const adapter = makeAdapter()
    const chunks = await collect(adapter, userRequest({ reasoningEffort: 'off' }))
    assert.equal(state.lastChatBody.think, false)
    assert.equal(state.lastChatBody.model, 'qwen3:27b')
    assert.equal(state.lastChatBody.messages[0].content, 'Hi')
    assert.equal(BLOCK_END(chunks, 0).type, 'reasoning')
    assert.equal(BLOCK_END(chunks, 0).text, 'The user greets me; reply kindly.')
    assert.equal(BLOCK_END(chunks, 1).type, 'text')
    assert.equal(BLOCK_END(chunks, 1).text, 'Hello!')
    const usage = chunks.find((c) => c.type === 'usage')
    assert.equal(usage.usage.inputTokens, 12)
    assert.equal(usage.usage.outputTokens, 34)
    assert.equal(usage.usage.totalTokens, 46)
    const finish = chunks.at(-1)
    assert.deepEqual(finish.reason, { kind: 'stop' })
  } finally {
    server.close()
  }
})

test('native strategy: effort low sends think:true by default and think:"high" under nativeLevels', async () => {
  const { server, state } = await startMock()
  try {
    const plain = makeAdapter()
    await collect(plain, userRequest({ reasoningEffort: 'low' }))
    assert.equal(state.lastChatBody.think, true)

    const leveled = makeAdapter({ nativeLevels: true })
    await collect(leveled, userRequest({ reasoningEffort: 'high' }))
    assert.equal(state.lastChatBody.think, 'high')
  } finally {
    server.close()
  }
})

test('native strategy: requests carry options, tools, and history shapes', async () => {
  const { server, state } = await startMock()
  try {
    const adapter = makeAdapter()
    await collect(adapter, userRequest({
      reasoningEffort: 'off',
      temperature: 0.3,
      maxTokens: 1234,
      stop: ['END'],
      tools: [{ name: 'get_weather', description: 'weather lookup', parameters: { type: 'object', properties: {} } }],
      system: 'be brief',
    }))
    const body = state.lastChatBody
    assert.equal(body.options.temperature, 0.3)
    assert.equal(body.options.num_predict, 1234)
    assert.deepEqual(body.options.stop, ['END'])
    assert.equal(body.tools[0].function.name, 'get_weather')
    assert.equal(body.messages[0].role, 'system')
    assert.equal(body.messages[0].content, 'be brief')
  } finally {
    server.close()
  }
})

test('soft-switch strategy appends markers instead of think', async () => {
  const { server, state } = await startMock()
  try {
    const off = makeAdapter({ strategy: 'soft-switch' })
    await collect(off, userRequest({ reasoningEffort: 'off' }))
    assert.equal(state.lastChatBody.think, undefined)
    assert.ok(state.lastChatBody.messages.at(-1).content.endsWith('/no_think'))

    const on = makeAdapter({ strategy: 'soft-switch' })
    await collect(on, userRequest({ reasoningEffort: 'high' }))
    assert.equal(state.lastChatBody.think, undefined)
    assert.ok(state.lastChatBody.messages.at(-1).content.endsWith('/think'))
  } finally {
    server.close()
  }
})

test('reasoning-effort strategy sends reasoning_effort on /v1/chat/completions', async () => {
  const { server, state } = await startMock()
  try {
    const adapter = makeAdapter({ strategy: 'reasoning-effort' })
    const chunks = await collect(adapter, userRequest({ reasoningEffort: 'low' }))
    assert.equal(state.openAICalls, 1)
    assert.equal(state.chatCalls, 0)
    assert.equal(state.lastOpenAIBody.reasoning_effort, 'low')
    assert.equal(state.lastOpenAIBody.stream_options.include_usage, true)
    assert.equal(BLOCK_END(chunks, 0).type, 'reasoning')
    assert.equal(BLOCK_END(chunks, 0).text, 'Careful…')
    assert.equal(BLOCK_END(chunks, 1).text, 'Hi there!')
    assert.equal(chunks.at(-1).reason.kind, 'stop')
  } finally {
    server.close()
  }
})

test('reasoning-effort off honors offSentinel none and omit', async () => {
  const { server, state } = await startMock()
  try {
    const none = makeAdapter({ strategy: 'reasoning-effort' })
    await collect(none, userRequest({ reasoningEffort: 'off' }))
    assert.equal(state.lastOpenAIBody.reasoning_effort, 'none')

    const omit = makeAdapter({ strategy: 'reasoning-effort', offSentinel: 'omit' })
    await collect(omit, userRequest({ reasoningEffort: 'off' }))
    assert.equal('reasoning_effort' in state.lastOpenAIBody, false)
  } finally {
    server.close()
  }
})

test('template-kwarg strategy sends enable_thinking and thinking_budget', async () => {
  const { server, state } = await startMock()
  try {
    const efforts = parseEfforts(['off', { id: 'high', budget: 2048 }], logger.warn)
    const adapter = makeAdapter({ strategy: 'template-kwarg' }, efforts)
    await collect(adapter, userRequest({ reasoningEffort: 'high' }))
    assert.deepEqual(state.lastOpenAIBody.chat_template_kwargs, { enable_thinking: true, thinking_budget: 2048 })

    await collect(adapter, userRequest({ reasoningEffort: 'off' }))
    assert.deepEqual(state.lastOpenAIBody.chat_template_kwargs, { enable_thinking: false })
  } finally {
    server.close()
  }
})

test('SSE tool-call fragments aggregate into a tool-call block with a tool-calls finish', async () => {
  const { server, state } = await startMock()
  try {
    await setScenario(server, 'sse-tool')
    const adapter = makeAdapter({ strategy: 'reasoning-effort' })
    const chunks = await collect(adapter, userRequest({ reasoningEffort: 'low', tools: [{
      name: 'get_weather', description: 'weather', parameters: { type: 'object' },
    }] }))
    const block = chunks.find((c) => c.type === 'block-end' && c.block.type === 'tool-call').block
    assert.equal(block.id, 'call_abc')
    assert.equal(block.name, 'get_weather')
    assert.deepEqual(JSON.parse(block.arguments), { city: 'Hangzhou' })
    const usage = chunks.find((c) => c.type === 'usage')
    assert.equal(usage.usage.totalTokens, 50)
    assert.equal(chunks.at(-1).reason.kind, 'tool-calls')
  } finally {
    server.close()
  }
})

test('inline <think> tags in content split into reasoning and text blocks', async () => {
  const { server, state } = await startMock()
  try {
    await setScenario(server, 'think-tags')
    const adapter = makeAdapter()
    const chunks = await collect(adapter, userRequest({ reasoningEffort: 'low' }))
    // Text before the opening tag, then the trace, then the visible answer.
    assert.equal(BLOCK_END(chunks, 0).type, 'text')
    assert.equal(BLOCK_END(chunks, 0).text, 'Let me ')
    assert.equal(BLOCK_END(chunks, 1).type, 'reasoning')
    assert.equal(BLOCK_END(chunks, 1).text, '2+2=4')
    assert.equal(BLOCK_END(chunks, 2).type, 'text')
    assert.equal(BLOCK_END(chunks, 2).text, 'So 4')
  } finally {
    server.close()
  }
})

test('native tool calls map to tool-call blocks and history tool results to role:tool', async () => {
  const { server, state } = await startMock()
  try {
    await setScenario(server, 'tool-call')
    const adapter = makeAdapter()
    const chunks = await collect(adapter, userRequest({
      reasoningEffort: 'off',
      tools: [{ name: 'get_weather', description: 'weather', parameters: { type: 'object' } }],
    }))
    const block = chunks.find((c) => c.type === 'block-end' && c.block.type === 'tool-call').block
    assert.equal(block.name, 'get_weather')
    assert.deepEqual(JSON.parse(block.arguments), { city: 'Hangzhou' })
    assert.equal(chunks.at(-1).reason.kind, 'stop')

    // History replay: assistant tool call + tool result map to wire shapes.
    await collect(adapter, userRequest({
      reasoningEffort: 'off',
      messages: [
        createSystemMessage('sys', 'test'),
        createUserMessage({ content: [{ type: 'text', text: 'weather?' }], source: { kind: 'user' } }),
        {
          ...createUserMessage({ content: [], source: { kind: 'user' } }),
          role: 'assistant',
          content: [{
            type: 'tool-call',
            id: 'call_1',
            name: 'get_weather',
            arguments: '{"city":"Hangzhou"}',
          }],
          source: { kind: 'model', provider: 'ollama', model: 'qwen3:27b' },
        },
        {
          ...createUserMessage({ content: [], source: { kind: 'user' } }),
          content: [{
            type: 'tool-result',
            toolCallId: 'call_1',
            content: [{ type: 'text', text: 'sunny 25C' }],
          }],
          source: { kind: 'tool', callId: 'call_1' },
        },
      ],
    }))
    const wire = state.lastChatBody.messages
    assert.equal(wire[0].role, 'system')
    assert.equal(wire[2].role, 'assistant')
    assert.deepEqual(wire[2].tool_calls[0].function, { name: 'get_weather', arguments: { city: 'Hangzhou' } })
    assert.equal(wire[3].role, 'tool')
    assert.equal(wire[3].content, 'sunny 25C')
    assert.equal(wire[3].tool_calls, undefined)
  } finally {
    server.close()
  }
})

test('historyThinking: keep replays assistant reasoning on the native wire', async () => {
  const { server, state } = await startMock()
  try {
    const adapter = makeAdapter({ historyThinking: 'keep' })
    await collect(adapter, userRequest({
      reasoningEffort: 'off',
      messages: [{
        ...createUserMessage({ content: [], source: { kind: 'user' } }),
        role: 'assistant',
        content: [
          { type: 'reasoning', text: 'earlier thought' },
          { type: 'text', text: 'answer' },
        ],
        source: { kind: 'model', provider: 'ollama', model: 'qwen3:27b' },
      }],
    }))
    const wire = state.lastChatBody.messages
    assert.equal(wire[0].role, 'assistant')
    assert.equal(wire[0].thinking, 'earlier thought')
    assert.equal(wire[0].content, 'answer')
  } finally {
    server.close()
  }
})

test('in-stream error and HTTP failures become coded LlmErrors', async () => {
  const { server } = await startMock()
  try {
    await setScenario(server, 'error-mid')
    const adapter = makeAdapter()
    await assert.rejects(
      collect(adapter, userRequest({ reasoningEffort: 'low' })),
      (error) => error instanceof LlmError && error.code === 'SERVER',
    )

    await setScenario(server, 'server-error')
    await assert.rejects(
      collect(adapter, userRequest({ reasoningEffort: 'low' })),
      (error) => error instanceof LlmError && error.code === 'SERVER' && error.failure.status === 500,
    )

    await setScenario(server, 'http-400')
    await assert.rejects(
      collect(adapter, userRequest({ reasoningEffort: 'low' })),
      (error) => error instanceof LlmError && error.code === 'INVALID_REQUEST',
    )
  } finally {
    server.close()
  }
})

test('done_reason length maps to max-tokens finish', async () => {
  const { server } = await startMock()
  try {
    await setScenario(server, 'length')
    const adapter = makeAdapter()
    const chunks = await collect(adapter, userRequest({ reasoningEffort: 'off' }))
    assert.deepEqual(chunks.at(-1).reason, { kind: 'max-tokens' })
  } finally {
    server.close()
  }
})

test('aborting the request surfaces an ABORTED LlmError', async () => {
  const { server } = await startMock()
  try {
    await setScenario(server, 'slow')
    const controller = new AbortController()
    const adapter = makeAdapter({ streamIdleTimeoutMs: 0 })
    const stream = adapter.stream(userRequest({ reasoningEffort: 'low', signal: controller.signal }))
    const iterator = stream[Symbol.asyncIterator]()
    const first = await iterator.next()
    assert.equal(first.done, false)
    controller.abort()
    await assert.rejects(
      (async () => {
        while (true) await iterator.next()
      })(),
      (error) => error instanceof LlmError && error.code === 'ABORTED',
    )
  } finally {
    server.close()
  }
})

test('idle stall surfaces a TIMEOUT LlmError', async () => {
  const { server } = await startMock()
  try {
    await setScenario(server, 'slow')
    const adapter = makeAdapter({ streamIdleTimeoutMs: 150 })
    await assert.rejects(
      collect(adapter, userRequest({ reasoningEffort: 'low' })),
      (error) => error instanceof LlmError && error.code === 'TIMEOUT',
    )
  } finally {
    server.close()
  }
})

test('vision capability advertises image input modality', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter()
    const vision = await adapter.resolveModel('ollama', 'qwen3.8:27b')
    assert.deepEqual(vision.inputModalities, ['text', 'image'])
    assert.equal(vision.context.contextWindow, 262144)
    const textOnly = await adapter.resolveModel('ollama', 'llama3:8b')
    assert.deepEqual(textOnly.inputModalities, ['text'])
    const forcedOff = makeAdapter({ imageCapability: 'no' })
    assert.deepEqual((await forcedOff.resolveModel('ollama', 'qwen3.8:27b')).inputModalities, ['text'])
    const forcedOn = makeAdapter({ imageCapability: 'yes' })
    assert.deepEqual((await forcedOn.resolveModel('ollama', 'llama3:8b')).inputModalities, ['text', 'image'])
  } finally {
    server.close()
  }
})

test('native strategy: user images ride the wire as base64 with the configured policy', async () => {
  const { server, state } = await startMock()
  try {
    const store = fakeAttachmentStore(new Map([
      ['att_1', { mediaType: 'image/png', bytes: [9, 8, 7] }],
    ]))
    const adapter = makeAdapter({ imageMaxPixels: 640 * 640, imageMaxBytes: 2048 }, undefined, () => store)
    const chunks = await collect(adapter, userRequest({
      reasoningEffort: 'off',
      messages: [imageMessage('What is this?', [imageRef('att_1')])],
    }))
    const userWire = state.lastChatBody.messages.at(-1)
    assert.equal(userWire.role, 'user')
    assert.equal(userWire.content, 'What is this?')
    assert.deepEqual(userWire.images, [Buffer.from([9, 8, 7]).toString('base64')])
    assert.deepEqual(store.requests, [{ attachmentId: 'att_1', policy: { maxPixels: 640 * 640, maxBytes: 2048 } }])
    assert.equal(chunks.at(-1).reason.kind, 'stop')
  } finally {
    server.close()
  }
})

test('reasoning-effort strategy: images become data-URI content parts', async () => {
  const { server, state } = await startMock()
  try {
    const store = fakeAttachmentStore(new Map([
      ['att_1', { mediaType: 'image/jpeg', bytes: [4, 5, 6] }],
    ]))
    const adapter = makeAdapter({ strategy: 'reasoning-effort' }, undefined, () => store)
    await collect(adapter, userRequest({
      reasoningEffort: 'low',
      messages: [imageMessage('Describe', [imageRef('att_1')])],
    }))
    const userWire = state.lastOpenAIBody.messages.at(-1)
    assert.equal(userWire.role, 'user')
    assert.deepEqual(userWire.content[0], { type: 'text', text: 'Describe' })
    assert.deepEqual(userWire.content[1], {
      type: 'image_url',
      image_url: { url: `data:image/jpeg;base64,${Buffer.from([4, 5, 6]).toString('base64')}` },
    })
  } finally {
    server.close()
  }
})

test('per-request image budget offloads the oldest occurrence to placeholder text', async () => {
  const { server, state } = await startMock()
  try {
    const store = fakeAttachmentStore(new Map([
      ['att_keep', { mediaType: 'image/png', bytes: [1] }],
    ]))
    const adapter = makeAdapter({ imageMaxPerRequest: 1 }, undefined, () => store)
    await collect(adapter, userRequest({
      reasoningEffort: 'off',
      messages: [imageMessage('two images', [imageRef('att_drop'), imageRef('att_keep')])],
    }))
    const userWire = state.lastChatBody.messages.at(-1)
    assert.equal(userWire.images.length, 1)
    assert.deepEqual(userWire.images, [Buffer.from([1]).toString('base64')])
    assert.ok(userWire.content.includes(offloadedImageText(imageRef('att_drop'))))
    // The offloaded reference is never read from the store.
    assert.deepEqual(store.requests.map((request) => request.attachmentId), ['att_keep'])
  } finally {
    server.close()
  }
})

test('image input without the attachment service fails with UNSUPPORTED_CONTENT', async () => {
  const { server } = await startMock()
  try {
    const adapter = makeAdapter()
    await assert.rejects(
      collect(adapter, userRequest({
        reasoningEffort: 'off',
        messages: [imageMessage('pic', [imageRef('att_1')])],
      })),
      (error) => error instanceof LlmError && error.code === 'UNSUPPORTED_CONTENT',
    )
  } finally {
    server.close()
  }
})

test('real LlmRuntime validates efforts, materializes defaults, and streams', async () => {
  const { server, state } = await startMock()
  try {
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    const efforts = parseEfforts(['off', 'low', { id: 'high', name: 'High', budget: 4096 }], logger.warn)
    const cfg = resolveConfig({
      endpoint: BASE,
      providers: ['ollama'],
      strategy: 'native',
      nativeLevels: false,
      offSentinel: 'none',
      assumeThinking: 'auto',
      defaultEffort: 'low',
      defaultContextWindow: 32768,
      defaultMaxTokens: 8192,
      streamIdleTimeoutMs: 5000,
      historyThinking: 'strip',
      models: [],
      imageCapability: 'auto',
      imageMaxPixels: 1024 * 1024,
      imageMaxBytes: 4 * 1024 * 1024,
      imageMaxPerRequest: 8,
      imageMaxRequestBytes: 32 * 1024 * 1024,
    })
    ctx.llm.registerAdapter(['ollama'], new OllamaThinkAdapter(cfg, efforts, logger))

    const messages = [createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })]

    // Unsupported effort → terminal error finish (service normalization), no provider I/O.
    const rejected = []
    for await (const chunk of ctx.llm.stream({ provider: 'ollama', model: 'qwen3:27b', reasoningEffort: 'medium', messages })) {
      rejected.push(chunk)
    }
    const failure = rejected.at(-1)
    assert.equal(failure.type, 'finish')
    assert.equal(failure.reason.kind, 'error')
    assert.equal(failure.reason.failure.code, 'UNSUPPORTED_REASONING_EFFORT')
    assert.equal(state.chatCalls, 0)

    // Omitted effort materializes the configured default (low → think:true).
    const defaulted = []
    for await (const chunk of ctx.llm.stream({ provider: 'ollama', model: 'qwen3:27b', messages })) defaulted.push(chunk)
    assert.equal(state.lastChatBody.think, true)
    assert.equal(state.chatCalls, 1)
    assert.ok(defaulted.some((chunk) => chunk.type === 'text-delta'))
    assert.equal(state.lastOpenAIBody, undefined)

    // Explicit effort flows through; blocks assemble.
    await setScenario(server, 'thinking')
    const explicit = []
    for await (const chunk of ctx.llm.stream({ provider: 'ollama', model: 'qwen3:27b', reasoningEffort: 'off', messages })) {
      explicit.push(chunk)
    }
    assert.equal(state.lastChatBody.think, false)
    assert.ok(explicit.some((chunk) => chunk.type === 'text-delta'))
    assert.deepEqual(explicit.at(-1).reason, { kind: 'stop' })
  } finally {
    server.close()
  }
})
