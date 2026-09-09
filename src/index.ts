/**
 * dsh-thinktune — a DeepSeek Harness LLM adapter plugin for Ollama with
 * explicit thinking-intensity control.
 *
 * Mounts an `OllamaThinkAdapter` on the configured provider routes. The
 * harness reasoning effort (pickable per agent config and in the web model
 * selector) is translated into one of four common control strategies:
 *
 * - `native`          — Ollama `/api/chat` `think`: `false` | `true` | level
 * - `soft-switch`     — Qwen3 `/think` / `/no_think` marker on the last user message
 * - `reasoning-effort`— OpenAI-compatible `reasoning_effort` on `/v1/chat/completions`
 * - `template-kwarg`  — `chat_template_kwargs.enable_thinking` / `thinking_budget`
 */
import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { OllamaThinkAdapter } from './adapter.ts'
import type { ModelEntry } from './config.ts'
import { resolveConfig } from './config.ts'
import { DEFAULT_EFFORTS, parseEfforts } from './efforts.ts'
import type { EffortEntry, Strategy } from './efforts.ts'

export const name = 'thinktune-ollama'
export const inject = ['llm']

export interface Config {
  /** Provider routes this adapter registers (referenced as `provider` by agents). */
  providers: string[]
  /** Ollama base URL. */
  endpoint: string
  /** Optional env-var NAME holding a bearer token for gated endpoints; empty for none. */
  apiKeyEnv?: string
  /** Which thinking-control strategy the reasoning effort maps to. */
  strategy: Strategy
  /** `native` only: map low/medium/high to `think: "low"…` instead of `think: true`. */
  nativeLevels: boolean
  /** `reasoning-effort` only: wire value for `off` (`'none'`, `'minimal'`, or `'omit'`). */
  offSentinel: string
  /** Thinking capability: `auto` follows /api/show (assumes yes when unavailable); yes/no force it. */
  assumeThinking: 'auto' | 'yes' | 'no'
  /** Advertised reasoning efforts; defaults to off/low/medium/high. */
  efforts: (string | EffortEntry)[]
  /** Effort materialized into requests that omit one; leave unset for the provider default. */
  defaultEffort?: string
  defaultContextWindow: number
  defaultMaxTokens: number
  streamIdleTimeoutMs: number
  /** Whether assistant reasoning blocks replay to the provider in history. */
  historyThinking: 'strip' | 'keep'
  /** Advisory model catalog with optional capacity overrides. */
  models: ModelEntry[]
  /** Image input capability: `auto` follows /api/show `vision`; `yes`/`no` force it. */
  imageCapability: 'auto' | 'yes' | 'no'
  /** Aspect-preserving pixel budget per request image (width × height). */
  imageMaxPixels: number
  /** Encoded-byte target per request image after re-encoding. */
  imageMaxBytes: number
  /** Images accepted per request; oldest occurrences degrade to placeholder text. */
  imageMaxPerRequest: number
  /** Accumulated raw image bytes accepted per request before offloading. */
  imageMaxRequestBytes: number
}

const ConfigSchema = Schema.object({
  providers: Schema.array(Schema.string()).default(['ollama']).description('Provider routes to register'),
  endpoint: Schema.string().default('http://127.0.0.1:11434').description('Ollama base URL'),
  apiKeyEnv: Schema.string().default('').description('Env var NAME of an optional bearer token'),
  strategy: Schema.union<Strategy>(['native', 'soft-switch', 'reasoning-effort', 'template-kwarg'])
    .default('native')
    .description('How the selected reasoning effort reaches the wire'),
  nativeLevels: Schema.boolean().default(false)
    .description('native: send think: "low"/"medium"/"high" instead of think: true'),
  offSentinel: Schema.string().default('none')
    .description('reasoning-effort: wire value for off (or "omit" to drop the field)'),
  assumeThinking: Schema.union(['auto', 'yes', 'no']).default('auto')
    .description('Thinking capability: auto follows /api/show (assumes yes when unavailable); yes/no force it'),
  efforts: Schema.array(
    Schema.union([
      Schema.string(),
      Schema.object({
        id: Schema.string().required(),
        name: Schema.string(),
        description: Schema.string(),
        budget: Schema.natural(),
      }),
    ]),
  ).default(DEFAULT_EFFORTS.map((effort) => ({ ...effort })) as never)
    .description('Advertised reasoning efforts'),
  defaultEffort: Schema.string().description('Effort applied when a request omits one'),
  defaultContextWindow: Schema.natural().default(32768).description('Fallback context window (tokens)'),
  defaultMaxTokens: Schema.natural().default(8192).description('Per-request output cap (tokens)'),
  streamIdleTimeoutMs: Schema.natural().default(300000).description('Max provider idle per stream read (ms)'),
  historyThinking: Schema.union(['strip', 'keep']).default('strip')
    .description('Replay assistant reasoning blocks in history'),
  models: Schema.array(
    Schema.object({
      id: Schema.string().required(),
      name: Schema.string(),
      description: Schema.string(),
      contextWindow: Schema.natural(),
      maxTokens: Schema.natural(),
    }),
  ).default([]).description('Advisory model catalog and capacity overrides'),
  imageCapability: Schema.union(['auto', 'yes', 'no']).default('auto')
    .description('Image input capability: auto follows /api/show vision; yes/no force it'),
  imageMaxPixels: Schema.natural().default(1024 * 1024).description('Pixel budget per request image (w×h)'),
  imageMaxBytes: Schema.natural().default(4 * 1024 * 1024).description('Encoded-byte target per request image'),
  imageMaxPerRequest: Schema.natural().default(8).description('Images accepted per request (0 disables image input)'),
  imageMaxRequestBytes: Schema.natural().default(32 * 1024 * 1024)
    .description('Accumulated image bytes per request before offloading'),
})

export const Config: Schema<Config> = ConfigSchema as unknown as Schema<Config>

export function apply(ctx: Context, config: Config): void {
  const resolved = resolveConfig(config)
  const efforts = parseEfforts(config.efforts, (message) => ctx.logger.warn(message))
  if (efforts.length === 0) throw new Error('thinktune: efforts must not be empty after normalization')
  if (resolved.defaultEffort !== undefined && !efforts.some((effort) => effort.id === resolved.defaultEffort)) {
    throw new Error(
      `thinktune: defaultEffort "${resolved.defaultEffort}" is not among the configured efforts ` +
      `[${efforts.map((effort) => effort.id).join(', ')}]`,
    )
  }
  if (resolved.strategy === 'reasoning-effort' && resolved.offSentinel !== 'omit' && resolved.offSentinel.length === 0) {
    throw new Error('thinktune: offSentinel must be a non-empty string or "omit"')
  }
  ctx.llm.registerAdapter(resolved.providers, new OllamaThinkAdapter(
    resolved,
    efforts,
    ctx.logger,
    // The attachment service is mounted by every image-capable composition;
    // absence stays legal and rejects image input at request time.
    () => ctx.get('attachments'),
  ))
  ctx.logger.info(
    'thinktune: provider route(s) %s registered (strategy %s, efforts %s)',
    resolved.providers.join(', '),
    resolved.strategy,
    efforts.map((effort) => effort.id).join('/'),
  )
}
