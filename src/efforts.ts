/**
 * Unified reasoning-effort (thinking intensity) vocabulary and its translation
 * into each supported control strategy.
 *
 * Effort ids are adapter-owned opaque strings (see `LlmReasoningEffortInfo` in
 * @deepseek-ai/dsh-llm): `LlmRuntime` validates the caller's
 * `GenerateOptions.reasoningEffort` against exactly the ids advertised by
 * {@link ./adapter.ts} `resolveModel`, and the selected id lands here for wire
 * translation. `off` is the reserved "no thinking" id; every other id maps to
 * "thinking on" under every strategy.
 */
import type { ReasoningEffortId } from '@deepseek-ai/dsh-llm'

/** One selectable reasoning effort as configured by the user. */
export interface EffortEntry {
  /** Opaque stable value accepted by `GenerateOptions.reasoningEffort`. */
  id: string
  /** Human-readable effort name for selectors; defaults to the id. */
  name?: string
  /** Optional user-facing distinction from otherwise similar efforts. */
  description?: string
  /** Thinking-budget token cap, used by the `template-kwarg` strategy (`thinking_budget`). */
  budget?: number
}

/** Resolved, validated effort spec. */
export interface EffortSpec {
  id: string
  name: string
  description?: string
  budget?: number
}

/** The reserved "no thinking" effort id. */
export const OFF_EFFORT_ID = 'off'

/** Which control mechanism the selected reasoning effort maps to. */
export type Strategy = 'native' | 'soft-switch' | 'reasoning-effort' | 'template-kwarg'

/** Wire values Ollama accepts as native `think` levels. */
export const NATIVE_THINK_LEVELS = ['low', 'medium', 'high', 'max']

export const DEFAULT_EFFORTS: EffortSpec[] = [
  { id: 'off', name: 'Off', description: 'Disable thinking entirely' },
  { id: 'low', name: 'Low', description: 'Light thinking pass' },
  { id: 'medium', name: 'Medium', description: 'Balanced thinking' },
  { id: 'high', name: 'High', description: 'Extended thinking' },
]

/**
 * Normalize user-configured effort entries: fill names, drop duplicates with a
 * warning, and validate budgets. Returns the built-in default set when the
 * input is empty or omitted.
 */
export function parseEfforts(
  input: readonly (string | EffortEntry)[] | undefined,
  warn: (message: string) => void,
): EffortSpec[] {
  if (!input || input.length === 0) return DEFAULT_EFFORTS.map((effort) => ({ ...effort }))
  const specs: EffortSpec[] = []
  const seen = new Set<string>()
  for (const entry of input) {
    const candidate: EffortEntry = typeof entry === 'string' ? { id: entry } : entry
    const id = candidate.id
    if (typeof id !== 'string' || id.trim().length === 0) {
      throw new Error(`thinktune: effort entry ${JSON.stringify(entry)} has an empty id`)
    }
    if (candidate.budget !== undefined && (!Number.isInteger(candidate.budget) || candidate.budget <= 0)) {
      throw new Error(`thinktune: effort "${id}" has a non-positive-integer budget ${candidate.budget}`)
    }
    if (seen.has(id)) {
      warn(`thinktune: duplicate effort id "${id}" ignored`)
      continue
    }
    seen.add(id)
    specs.push({
      id,
      name: candidate.name ?? id,
      description: candidate.description,
      budget: candidate.budget,
    })
  }
  return specs
}

export function findEffort(specs: readonly EffortSpec[], id: ReasoningEffortId | string): EffortSpec | undefined {
  return specs.find((spec) => spec.id === id)
}

/** `native` strategy: the top-level `think` value of an Ollama `/api/chat` request. */
export function nativeThinkValue(spec: EffortSpec, nativeLevels: boolean): boolean | string {
  if (spec.id === OFF_EFFORT_ID) return false
  if (nativeLevels && NATIVE_THINK_LEVELS.includes(spec.id)) return spec.id
  return true
}

/** `soft-switch` strategy: the Qwen3 soft switch appended to the last user message. */
export function softSwitchMarker(spec: EffortSpec): string {
  return spec.id === OFF_EFFORT_ID ? '/no_think' : '/think'
}

/**
 * `reasoning-effort` strategy: the OpenAI-compatible `reasoning_effort` field.
 * `offSentinel` is the wire value for `off`; `'omit'` drops the field so the
 * endpoint keeps its own default.
 */
export function reasoningEffortValue(spec: EffortSpec, offSentinel: string): string | undefined {
  if (spec.id === OFF_EFFORT_ID) return offSentinel === 'omit' ? undefined : offSentinel
  return spec.id
}

/** `template-kwarg` strategy: `chat_template_kwargs` for vLLM/SGLang-style endpoints. */
export function templateKwargs(spec: EffortSpec): Record<string, unknown> {
  if (spec.id === OFF_EFFORT_ID) return { enable_thinking: false }
  const kwargs: Record<string, unknown> = { enable_thinking: true }
  if (spec.budget !== undefined) kwargs.thinking_budget = spec.budget
  return kwargs
}
