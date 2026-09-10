/**
 * OpenAI-compatible wire protocol: `POST /v1/chat/completions` with SSE
 * streaming, as spoken by Ollama's compatibility layer and by vLLM/SGLang.
 * The harness reasoning effort travels as `reasoning_effort` (the
 * `reasoning-effort` strategy) or as `chat_template_kwargs` with
 * `enable_thinking`/`thinking_budget` (the `template-kwarg` strategy).
 */
import { LlmError, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import { TextBlockEmitter } from './blocks.ts'
import type { WireMessage } from './messages.ts'
import { ThinkTagSplitter } from './think-tag.ts'

export interface OpenAIRequest {
  model: string
  messages: Array<Record<string, unknown>>
  stream: true
  stream_options?: { include_usage: true }
  reasoning_effort?: string
  chat_template_kwargs?: Record<string, unknown>
  temperature?: number
  max_tokens?: number
  stop?: string[]
  tools?: Array<Record<string, unknown>>
}

export interface BuildOpenAIRequestOptions {
  model: string
  messages: WireMessage[]
  tools?: Array<Record<string, unknown>>
  temperature?: number
  maxTokens?: number
  stop?: string[]
  /** `reasoning_effort` wire value; `undefined` omits the field (provider default). */
  reasoningEffort?: string
  chatTemplateKwargs?: Record<string, unknown>
}

/** OpenAI multimodal content part for one encoded image. */
function imagePart(mediaType: string, base64: string): Record<string, unknown> {
  return { type: 'image_url', image_url: { url: `data:${mediaType};base64,${base64}` } }
}

export function buildOpenAIRequest(options: BuildOpenAIRequestOptions): OpenAIRequest {
  const messages = options.messages.map((message) => {
    const hasImages = message.role === 'user' && message.images !== undefined && message.images.length > 0
    const wire: Record<string, unknown> = {
      role: message.role,
      content: hasImages
        ? [
            ...(message.content.length > 0 ? [{ type: 'text', text: message.content }] : []),
            ...message.images!.map((image) => imagePart(image.mediaType, image.base64)),
          ]
        : message.content,
    }
    if (message.role === 'tool' && message.toolCallId !== undefined) wire.tool_call_id = message.toolCallId
    if (message.role === 'assistant' && message.toolCalls) {
      wire.tool_calls = message.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: call.arguments },
      }))
    }
    return wire
  })
  const request: OpenAIRequest = {
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
  }
  if (options.reasoningEffort !== undefined) request.reasoning_effort = options.reasoningEffort
  if (options.chatTemplateKwargs !== undefined) request.chat_template_kwargs = options.chatTemplateKwargs
  if (options.temperature !== undefined) request.temperature = options.temperature
  if (options.maxTokens !== undefined) request.max_tokens = options.maxTokens
  if (options.stop && options.stop.length > 0) request.stop = options.stop
  if (options.tools && options.tools.length > 0) request.tools = options.tools
  return request
}

interface SseChunk {
  error?: { message?: string } | string
  choices?: Array<{
    delta?: {
      content?: string
      reasoning_content?: string
      reasoning?: unknown
      tool_calls?: Array<{
        index?: number
        id?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason?: string | null
  }>
  usage?: { prompt_tokens?: number; completion_tokens?: number } | null
}

interface OpenToolCall {
  id: string
  name: string
  arguments: string
  index: number
}

/**
 * Turn an SSE line iterable into harness chunks. Reasoning arrives via
 * `reasoning_content`/`reasoning` deltas or literal `<think>` tags inside
 * content; tool calls stream incrementally and close at finish. Usage is
 * emitted before the single terminal finish chunk.
 */
export async function* parseOpenAIStream(lines: AsyncIterable<string>): AsyncGenerator<StreamChunk> {
  const indices = { next: 0 }
  const blocks = new TextBlockEmitter(indices)
  const splitter = new ThinkTagSplitter()
  const toolCalls = new Map<number, OpenToolCall>()
  let finishReason: string | undefined
  let usage: { promptTokens?: number; completionTokens?: number } | undefined

  for await (const line of lines) {
    if (!line.startsWith('data:')) continue
    const payload = line.slice(5).trim()
    if (payload === '[DONE]') break
    let event: SseChunk
    try {
      event = JSON.parse(payload) as SseChunk
    } catch {
      throw new LlmError(`thinktune: unparseable SSE chunk: ${payload.slice(0, 200)}`, 'PROVIDER_HTTP_ERROR')
    }
    if (event.error) {
      const message = typeof event.error === 'string' ? event.error : event.error.message ?? 'provider error'
      throw new LlmError(`thinktune: provider stream error: ${message}`, 'SERVER')
    }

    const choice = event.choices?.[0]
    const delta = choice?.delta
    if (delta) {
      const reasoning =
        typeof delta.reasoning_content === 'string' && delta.reasoning_content.length > 0
          ? delta.reasoning_content
          : typeof delta.reasoning === 'string' && delta.reasoning.length > 0 ? delta.reasoning : ''
      if (reasoning) yield* blocks.emit(reasoning, 'reasoning')
      if (typeof delta.content === 'string' && delta.content.length > 0) {
        const split = splitter.split(delta.content)
        yield* blocks.emit(split.reasoning, 'reasoning')
        yield* blocks.emit(split.content, 'text')
      }
      if (Array.isArray(delta.tool_calls)) {
        for (const call of delta.tool_calls) {
          const key = typeof call.index === 'number' ? call.index : 0
          let entry = toolCalls.get(key)
          if (!entry) {
            const index = indices.next
            indices.next += 1
            entry = {
              id: call.id ?? `call_${key}`,
              name: call.function?.name ?? '',
              arguments: '',
              index,
            }
            toolCalls.set(key, entry)
            yield { type: 'block-start', index, blockType: 'tool-call' }
          }
          if (call.id && !call.function?.name) {
            // Continuation fragments may carry only the id; keep the first.
            entry.id = call.id
          }
          if (call.function?.name) entry.name = call.function.name
          if (call.function?.arguments) {
            entry.arguments += call.function.arguments
            yield {
              type: 'tool-call-delta',
              index: entry.index,
              id: ToolCallId(entry.id),
              ...(entry.name ? { name: entry.name } : {}),
              argumentsDelta: call.function.arguments,
            }
          }
        }
      }
    }
    if (choice?.finish_reason) finishReason = choice.finish_reason
    if (event.usage) {
      usage = {
        promptTokens: typeof event.usage.prompt_tokens === 'number' ? event.usage.prompt_tokens : undefined,
        completionTokens: typeof event.usage.completion_tokens === 'number' ? event.usage.completion_tokens : undefined,
      }
    }
  }

  yield* blocks.close()
  for (const entry of [...toolCalls.values()].sort((a, b) => a.index - b.index)) {
    yield {
      type: 'block-end',
      index: entry.index,
      block: { type: 'tool-call', id: ToolCallId(entry.id), name: entry.name, arguments: entry.arguments },
    }
  }
  const flushed = splitter.flush()
  if (flushed.reasoning) yield* blocks.emit(flushed.reasoning, 'reasoning')
  if (flushed.content) yield* blocks.emit(flushed.content, 'text')
  yield* blocks.close()
  if (usage && (usage.promptTokens !== undefined || usage.completionTokens !== undefined)) {
    const tokenUsage: TokenUsage = {
      inputTokens: usage.promptTokens ?? 0,
      outputTokens: usage.completionTokens ?? 0,
    }
    if (usage.promptTokens !== undefined && usage.completionTokens !== undefined) {
      tokenUsage.totalTokens = usage.promptTokens + usage.completionTokens
    }
    yield { type: 'usage', usage: tokenUsage }
  }
  if (finishReason === undefined) {
    throw new LlmError('thinktune: /v1/chat/completions stream ended without a finish reason', 'PROVIDER_HTTP_ERROR')
  }
  const kind = finishReason === 'length' ? 'max-tokens' : finishReason === 'tool_calls' ? 'tool-calls' : 'stop'
  yield { type: 'finish', reason: { kind } }
}
