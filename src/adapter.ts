/**
 * `OllamaThinkAdapter`: a fetch-based `LlmAdapter` for Ollama (and any
 * OpenAI-compatible local server it fronts) whose entire purpose is explicit
 * reasoning-intensity control. The harness reasoning effort selected per
 * request — `GenerateOptions.reasoningEffort`, validated by `LlmRuntime`
 * against the efforts this adapter advertises — is translated into exactly
 * one of four wire strategies chosen in plugin config.
 *
 * Vision input follows the same adapter contract as the shipped DeepSeek
 * adapter: image-capable routes declare `inputModalities`, durable image
 * references resolve through the attachment service into per-request
 * re-encoded versions (`readImageRequest`), and oldest occurrences offload to
 * deterministic placeholder text once a request budget is exceeded.
 */
import { LlmAdapter, LlmError, ReasoningEffortId, contentHasImage, offloadedImageText, offloadRequestImagesWithPolicy } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmReasoningEffortInfo,
  LlmResolvedModelInfo,
  Message,
  ModelModality,
  StreamChunk,
} from '@deepseek-ai/dsh-llm'
import type { AttachmentStore, ImageAttachmentRef, RequestImageAttachment } from '@deepseek-ai/dsh-attachment'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { EffortSpec, Strategy } from './efforts.ts'
import {
  nativeThinkValue,
  reasoningEffortValue,
  softSwitchMarker,
  templateKwargs,
  OFF_EFFORT_ID,
} from './efforts.ts'
import { applySoftSwitch, mapMessages, mapTools } from './messages.ts'
import type { ThinkTuneResolvedConfig } from './config.ts'
import { getJson, postJson, readLines, resolveBearer } from './http.ts'
import { buildNativeRequest, parseNativeStream } from './native.ts'
import { buildOpenAIRequest, parseOpenAIStream } from './openai.ts'

export interface AdapterLogger {
  warn(message: string): void
  info(message: string): void
}

interface ShowModelResponse {
  capabilities?: string[]
  model_info?: Record<string, unknown>
}

interface TagsResponse {
  models?: Array<{ name?: string; model?: string }>
}

/** Find `*.context_length` in a `/api/show` `model_info` map (arch-prefixed keys). */
function contextLengthFromShow(info: Record<string, unknown> | undefined): number | undefined {
  if (!info) return undefined
  for (const [key, value] of Object.entries(info)) {
    if (key.endsWith('context_length') && typeof value === 'number' && Number.isFinite(value) && value > 0) {
      return value
    }
  }
  return undefined
}

/** Collect image references from one request in first-seen order, deduplicated by id. */
function collectImageRefs(messages: readonly Message[]): ImageAttachmentRef[] {
  const refs: ImageAttachmentRef[] = []
  const seen = new Set<string>()
  const walk = (blocks: readonly ContentBlock[]): void => {
    for (const block of blocks) {
      if (block.type === 'image') {
        if (!seen.has(block.attachment.attachmentId)) {
          seen.add(block.attachment.attachmentId)
          refs.push(block.attachment)
        }
      } else if (block.type === 'tool-result') {
        walk(block.content)
      }
    }
  }
  for (const message of messages) walk(message.content)
  return refs
}

export class OllamaThinkAdapter extends LlmAdapter {
  private readonly cfg: ThinkTuneResolvedConfig
  private readonly efforts: readonly EffortSpec[]
  private readonly log: AdapterLogger
  private readonly resolveAttachments: () => AttachmentStore | undefined

  constructor(
    cfg: ThinkTuneResolvedConfig,
    efforts: readonly EffortSpec[],
    log: AdapterLogger,
    resolveAttachments: () => AttachmentStore | undefined = () => undefined,
  ) {
    super()
    this.cfg = cfg
    this.efforts = efforts
    this.log = log
    this.resolveAttachments = resolveAttachments
  }

  providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: `Ollama · ${this.cfg.strategy}` }
  }

  async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const overrides = new Map(this.cfg.models.map((model) => [model.id, model]))
    try {
      const tags = (await getJson(
        `${this.cfg.endpoint}/api/tags`,
        resolveBearer(this.cfg.apiKeyEnv, process.env),
      )) as TagsResponse
      const advertised: LlmModelInfo[] = []
      const seen = new Set<string>()
      for (const entry of tags.models ?? []) {
        const id = entry.model ?? entry.name
        if (!id || seen.has(id)) continue
        seen.add(id)
        const override = overrides.get(id)
        advertised.push({
          provider,
          id,
          name: override?.name ?? id,
          ...(override?.description ? { description: override.description } : {}),
        })
      }
      for (const override of this.cfg.models) {
        if (!seen.has(override.id)) {
          advertised.push({
            provider,
            id: override.id,
            name: override.name ?? override.id,
            ...(override.description ? { description: override.description } : {}),
          })
        }
      }
      return advertised
    } catch (error) {
      this.log.warn(
        `thinktune: GET /api/tags failed (${(error as Error).message}); advertising only the configured model list`,
      )
      return this.cfg.models.map((model) => ({
        provider,
        id: model.id,
        name: model.name ?? model.id,
        ...(model.description ? { description: model.description } : {}),
      }))
    }
  }

  async resolveModel(provider: string, model: string, signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    const override = this.cfg.models.find((entry) => entry.id === model)
    let show: ShowModelResponse | undefined
    try {
      show = (await postJson({
        url: `${this.cfg.endpoint}/api/show`,
        body: { model },
        bearer: resolveBearer(this.cfg.apiKeyEnv, process.env),
        signal,
      }).then((response) => response.json())) as ShowModelResponse
    } catch (error) {
      if (signal?.aborted) throw error
      this.log.warn(
        `thinktune: POST /api/show for "${model}" failed (${(error as Error).message}); ` +
          'falling back to configured thinking capability',
      )
    }
    const capabilities = show?.capabilities

    const thinkingCapable = this.cfg.assumeThinking === 'yes'
      ? true
      : this.cfg.assumeThinking === 'no'
        ? false
        : capabilities
          ? capabilities.includes('thinking')
          : true
    const imageCapable = this.cfg.imageCapability === 'yes'
      ? true
      : this.cfg.imageCapability === 'no'
        ? false
        : capabilities
          ? capabilities.includes('vision')
          : false
    const inputModalities: readonly ModelModality[] = imageCapable ? ['text', 'image'] : ['text']

    const reasoning = thinkingCapable && this.efforts.length > 0
      ? {
          efforts: this.efforts.map((effort): LlmReasoningEffortInfo => ({
            id: ReasoningEffortId(effort.id),
            name: effort.name,
            ...(effort.description ? { description: effort.description } : {}),
          })),
          ...(this.cfg.defaultEffort ? { defaultEffort: ReasoningEffortId(this.cfg.defaultEffort) } : {}),
        }
      : undefined
    const contextWindow = override?.contextWindow
      ?? contextLengthFromShow(show?.model_info)
      ?? this.cfg.defaultContextWindow
    return {
      provider,
      id: model,
      name: override?.name ?? model,
      ...(override?.description ? { description: override.description } : {}),
      inputModalities,
      context: { contextWindow },
      defaultMaxTokens: this.cfg.defaultMaxTokens,
      ...(reasoning ? { reasoning } : {}),
    }
  }

  /**
   * Resolve request-image versions through the attachment service, honoring
   * the per-image re-encode budget and the per-request offload budget.
   * Returns the messages actually sent (possibly with placeholder-substituted
   * oldest images) plus the prepared versions keyed by attachment id.
   */
  private async prepareImages(options: GenerateOptions, signal?: AbortSignal): Promise<{
    messages: readonly Message[]
    requestImages: ReadonlyMap<string, RequestImageAttachment>
  }> {
    const messages = contentHasImage(options.messages.flatMap((message) => message.content))
      ? offloadRequestImagesWithPolicy(options.messages, {
          representation: 'raw',
          maxImages: this.cfg.imageMaxPerRequest,
          maxBytes: this.cfg.imageMaxRequestBytes,
          countQuantum: 1,
          byteQuantum: Math.max(1, Math.floor(this.cfg.imageMaxRequestBytes / 8)),
          placeholder: offloadedImageText,
        })
      : options.messages
    const refs = collectImageRefs(messages)
    if (refs.length === 0) return { messages, requestImages: new Map() }
    const attachments = this.resolveAttachments()
    if (!attachments) {
      throw new LlmError(
        'thinktune: image input requires the durable attachment service (ctx.attachments), which is not mounted',
        'UNSUPPORTED_CONTENT',
      )
    }
    const versions = await Promise.all(
      refs.map((ref) =>
        attachments.readImageRequest(
          ref,
          { maxPixels: this.cfg.imageMaxPixels, maxBytes: this.cfg.imageMaxBytes },
          signal,
        ),
      ),
    )
    return { messages, requestImages: new Map(refs.map((ref, index) => [ref.attachmentId, versions[index]!])) }
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const spec = options.reasoningEffort !== undefined
      ? this.efforts.find((effort) => effort.id === options.reasoningEffort)
      : undefined
    if (options.reasoningEffort !== undefined && !spec) {
      throw new LlmError(
        `thinktune: reasoning effort "${options.reasoningEffort}" is not among the configured efforts ` +
          `[${this.efforts.map((effort) => effort.id).join(', ')}]`,
        'UNSUPPORTED_REASONING_EFFORT',
      )
    }
    const applied = spec ?? this.efforts.find((effort) => effort.id === this.cfg.defaultEffort)
    const { messages: imageProjected, requestImages } = await this.prepareImages(options, options.signal)

    const strategy = this.cfg.strategy
    const bearer = resolveBearer(this.cfg.apiKeyEnv, process.env)
    if (strategy === 'native' || strategy === 'soft-switch') {
      const messages = mapMessages(
        { ...options, messages: [...imageProjected] },
        { historyThinking: this.cfg.historyThinking, requestImages },
      )
      if (strategy === 'soft-switch' && applied) {
        applySoftSwitch(messages, softSwitchMarker(applied))
        if (applied.id === OFF_EFFORT_ID && !messages.some((message) => message.role === 'user')) {
          this.log.warn('thinktune: soft-switch strategy found no user message to carry /no_think')
        }
      }
      const request = buildNativeRequest({
        model: options.model,
        messages,
        tools: mapTools(options.tools),
        temperature: options.temperature,
        maxTokens: options.maxTokens,
        stop: options.stop,
        ...(strategy === 'native' && applied
          ? { think: nativeThinkValue(applied, this.cfg.nativeLevels) }
          : {}),
      })
      const response = await postJson({
        url: `${this.cfg.endpoint}/api/chat`,
        body: request,
        bearer,
        signal: options.signal,
      })
      yield* parseNativeStream(readLines(response, options.signal, this.cfg.streamIdleTimeoutMs))
      return
    }

    const messages = mapMessages(
      { ...options, messages: [...imageProjected] },
      { historyThinking: this.cfg.historyThinking, requestImages },
    )
    const request = buildOpenAIRequest({
      model: options.model,
      messages,
      tools: mapTools(options.tools),
      temperature: options.temperature,
      maxTokens: options.maxTokens,
      stop: options.stop,
      ...(strategy === 'reasoning-effort' && applied
        ? { reasoningEffort: reasoningEffortValue(applied, this.cfg.offSentinel) }
        : {}),
      ...(strategy === 'template-kwarg' && applied ? { chatTemplateKwargs: templateKwargs(applied) } : {}),
    })
    const response = await postJson({
      url: `${this.cfg.endpoint}/v1/chat/completions`,
      body: request,
      bearer,
      signal: options.signal,
    })
    yield* parseOpenAIStream(readLines(response, options.signal, this.cfg.streamIdleTimeoutMs))
  }
}
