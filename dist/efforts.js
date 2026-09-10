/** The reserved "no thinking" effort id. */
export const OFF_EFFORT_ID = 'off';
/** Wire values Ollama accepts as native `think` levels. */
export const NATIVE_THINK_LEVELS = ['low', 'medium', 'high', 'max'];
export const DEFAULT_EFFORTS = [
    { id: 'off', name: 'Off', description: 'Disable thinking entirely' },
    { id: 'low', name: 'Low', description: 'Light thinking pass' },
    { id: 'medium', name: 'Medium', description: 'Balanced thinking' },
    { id: 'high', name: 'High', description: 'Extended thinking' },
];
/**
 * Normalize user-configured effort entries: fill names, drop duplicates with a
 * warning, and validate budgets. Returns the built-in default set when the
 * input is empty or omitted.
 */
export function parseEfforts(input, warn) {
    if (!input || input.length === 0)
        return DEFAULT_EFFORTS.map((effort) => ({ ...effort }));
    const specs = [];
    const seen = new Set();
    for (const entry of input) {
        const candidate = typeof entry === 'string' ? { id: entry } : entry;
        const id = candidate.id;
        if (typeof id !== 'string' || id.trim().length === 0) {
            throw new Error(`thinktune: effort entry ${JSON.stringify(entry)} has an empty id`);
        }
        if (candidate.budget !== undefined && (!Number.isInteger(candidate.budget) || candidate.budget <= 0)) {
            throw new Error(`thinktune: effort "${id}" has a non-positive-integer budget ${candidate.budget}`);
        }
        if (seen.has(id)) {
            warn(`thinktune: duplicate effort id "${id}" ignored`);
            continue;
        }
        seen.add(id);
        specs.push({
            id,
            name: candidate.name ?? id,
            description: candidate.description,
            budget: candidate.budget,
        });
    }
    return specs;
}
export function findEffort(specs, id) {
    return specs.find((spec) => spec.id === id);
}
/** `native` strategy: the top-level `think` value of an Ollama `/api/chat` request. */
export function nativeThinkValue(spec, nativeLevels) {
    if (spec.id === OFF_EFFORT_ID)
        return false;
    if (nativeLevels && NATIVE_THINK_LEVELS.includes(spec.id))
        return spec.id;
    return true;
}
/** `soft-switch` strategy: the Qwen3 soft switch appended to the last user message. */
export function softSwitchMarker(spec) {
    return spec.id === OFF_EFFORT_ID ? '/no_think' : '/think';
}
/**
 * `reasoning-effort` strategy: the OpenAI-compatible `reasoning_effort` field.
 * `offSentinel` is the wire value for `off`; `'omit'` drops the field so the
 * endpoint keeps its own default.
 */
export function reasoningEffortValue(spec, offSentinel) {
    if (spec.id === OFF_EFFORT_ID)
        return offSentinel === 'omit' ? undefined : offSentinel;
    return spec.id;
}
/** `template-kwarg` strategy: `chat_template_kwargs` for vLLM/SGLang-style endpoints. */
export function templateKwargs(spec) {
    if (spec.id === OFF_EFFORT_ID)
        return { enable_thinking: false };
    const kwargs = { enable_thinking: true };
    if (spec.budget !== undefined)
        kwargs.thinking_budget = spec.budget;
    return kwargs;
}
