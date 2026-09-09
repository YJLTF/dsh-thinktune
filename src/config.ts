/** Plugin configuration types and normalization shared by the entry point and the adapter. */
import type { Strategy } from './efforts.ts'

/** One advisory model entry: selector overrides plus per-model capacity facts. */
export interface ModelEntry {
  /** Wire model id accepted by the endpoint (e.g. `qwen3:27b`). */
  id: string
  /** Selector label; defaults to the id. */
  name?: string
  /** Optional selector detail for deployments with similar model variants. */
  description?: string
  /** Combined request+response context capacity; wins over `/api/show` and the global default. */
  contextWindow?: number
  /** Unused capacity knob reserved for parity with the catalog shape. */
  maxTokens?: number
}

/** Fully normalized plugin configuration handed to the adapter. */
export interface ThinkTuneResolvedConfig {
  endpoint: string
  providers: string[]
  apiKeyEnv: string
  strategy: Strategy
  nativeLevels: boolean
  offSentinel: string
  assumeThinking: 'auto' | 'yes' | 'no'
  defaultEffort?: string
  defaultContextWindow: number
  defaultMaxTokens: number
  streamIdleTimeoutMs: number
  historyThinking: 'strip' | 'keep'
  models: ModelEntry[]
  /** Image input capability and per-request vision budgets. */
  imageCapability: 'auto' | 'yes' | 'no'
  imageMaxPixels: number
  imageMaxBytes: number
  imageMaxPerRequest: number
  imageMaxRequestBytes: number
}

/** Validate and normalize the raw schemastery-validated config into the resolved shape. */
export function resolveConfig(raw: {
  endpoint: string
  providers: string[]
  apiKeyEnv?: string
  strategy: Strategy
  nativeLevels: boolean
  offSentinel: string
  assumeThinking: 'auto' | 'yes' | 'no'
  defaultEffort?: string
  defaultContextWindow: number
  defaultMaxTokens: number
  streamIdleTimeoutMs: number
  historyThinking: 'strip' | 'keep'
  models: ModelEntry[]
  imageCapability: 'auto' | 'yes' | 'no'
  imageMaxPixels: number
  imageMaxBytes: number
  imageMaxPerRequest: number
  imageMaxRequestBytes: number
}): ThinkTuneResolvedConfig {
  if (raw.providers.length === 0) throw new Error('thinktune: providers must list at least one route name')
  let endpoint: URL
  try {
    endpoint = new URL(raw.endpoint)
  } catch {
    throw new Error(`thinktune: endpoint "${raw.endpoint}" is not a valid URL`)
  }
  if (endpoint.protocol !== 'http:' && endpoint.protocol !== 'https:') {
    throw new Error(`thinktune: endpoint "${raw.endpoint}" must be an http(s) URL`)
  }
  if (raw.defaultEffort !== undefined && raw.defaultEffort.length === 0) {
    throw new Error('thinktune: defaultEffort must be a non-empty effort id')
  }
  if (!Number.isFinite(raw.defaultContextWindow) || raw.defaultContextWindow <= 0) {
    throw new Error('thinktune: defaultContextWindow must be a positive number')
  }
  if (!Number.isFinite(raw.defaultMaxTokens) || raw.defaultMaxTokens <= 0) {
    throw new Error('thinktune: defaultMaxTokens must be a positive number')
  }
  if (!Number.isFinite(raw.streamIdleTimeoutMs) || raw.streamIdleTimeoutMs < 0) {
    throw new Error('thinktune: streamIdleTimeoutMs must be a non-negative number')
  }
  for (const model of raw.models) {
    if (typeof model.id !== 'string' || model.id.length === 0) {
      throw new Error('thinktune: every models entry needs a non-empty id')
    }
  }
  if (!Number.isFinite(raw.imageMaxPixels) || raw.imageMaxPixels <= 0) {
    throw new Error('thinktune: imageMaxPixels must be a positive number')
  }
  if (!Number.isFinite(raw.imageMaxBytes) || raw.imageMaxBytes <= 0) {
    throw new Error('thinktune: imageMaxBytes must be a positive number')
  }
  if (!Number.isInteger(raw.imageMaxPerRequest) || raw.imageMaxPerRequest < 0) {
    throw new Error('thinktune: imageMaxPerRequest must be a non-negative integer')
  }
  if (!Number.isFinite(raw.imageMaxRequestBytes) || raw.imageMaxRequestBytes <= 0) {
    throw new Error('thinktune: imageMaxRequestBytes must be a positive number')
  }
  return {
    endpoint: raw.endpoint.replace(/\/+$/, ''),
    providers: [...raw.providers],
    apiKeyEnv: raw.apiKeyEnv ?? '',
    strategy: raw.strategy,
    nativeLevels: raw.nativeLevels,
    offSentinel: raw.offSentinel,
    assumeThinking: raw.assumeThinking,
    defaultEffort: raw.defaultEffort,
    defaultContextWindow: raw.defaultContextWindow,
    defaultMaxTokens: raw.defaultMaxTokens,
    streamIdleTimeoutMs: raw.streamIdleTimeoutMs,
    historyThinking: raw.historyThinking,
    models: raw.models.map((model) => ({ ...model })),
    imageCapability: raw.imageCapability,
    imageMaxPixels: raw.imageMaxPixels,
    imageMaxBytes: raw.imageMaxBytes,
    imageMaxPerRequest: raw.imageMaxPerRequest,
    imageMaxRequestBytes: raw.imageMaxRequestBytes,
  }
}
