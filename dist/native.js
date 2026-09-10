/**
 * Ollama-native wire protocol: `POST /api/chat` with NDJSON streaming. The
 * `think` field (boolean, or a level string under `nativeLevels`) is the
 * harness reasoning effort's native carrier; the `soft-switch` strategy rides
 * the same protocol with no `think` field and a `/think`|`/no_think` marker
 * already appended to the last user message.
 */
import { LlmError, ToolCallId } from '@deepseek-ai/dsh-llm';
import { ThinkTagSplitter } from "./think-tag.js";
function toolCallArguments(wire) {
    try {
        return JSON.parse(wire.arguments);
    }
    catch {
        // Malformed stored arguments stay a string; Ollama tolerates it worse than
        // an empty object, but a broken history must not silently vanish.
        return {};
    }
}
export function buildNativeRequest(options) {
    const messages = options.messages.map((message) => {
        const wire = { role: message.role, content: message.content };
        if (message.thinking !== undefined)
            wire.thinking = message.thinking;
        if (message.images && message.images.length > 0) {
            wire.images = message.images.map((image) => image.base64);
        }
        if (message.role === 'assistant' && message.toolCalls) {
            wire.tool_calls = message.toolCalls.map((call) => ({
                function: { name: call.name, arguments: toolCallArguments(call) },
            }));
        }
        return wire;
    });
    const request = {
        model: options.model,
        messages,
        stream: true,
    };
    if (options.think !== undefined)
        request.think = options.think;
    if (options.tools && options.tools.length > 0)
        request.tools = options.tools;
    const ollamaOptions = {};
    if (options.temperature !== undefined)
        ollamaOptions.temperature = options.temperature;
    if (options.maxTokens !== undefined)
        ollamaOptions.num_predict = options.maxTokens;
    if (options.stop && options.stop.length > 0)
        ollamaOptions.stop = options.stop;
    if (Object.keys(ollamaOptions).length > 0)
        request.options = ollamaOptions;
    return request;
}
function openBlockBlock(block) {
    return {
        type: 'block-end',
        index: block.index,
        block: block.kind === 'text'
            ? { type: 'text', text: block.text }
            : { type: 'reasoning', text: block.text },
    };
}
/**
 * Turn NDJSON `/api/chat` lines into harness chunks. Handles thinking deltas,
 * content deltas (including literal `<think>` tags), atomic tool calls,
 * token usage, and the terminal finish. Throws a coded LlmError on in-stream
 * error objects and malformed lines.
 */
export async function* parseNativeStream(lines, contentSplitter = new ThinkTagSplitter()) {
    let open;
    let nextIndex = 0;
    let toolCounter = 0;
    let finished = false;
    const emitText = async function* (text, kind) {
        if (text.length === 0)
            return;
        if (open && open.kind !== kind) {
            yield openBlockBlock(open);
            open = undefined;
        }
        if (!open) {
            open = { index: nextIndex, kind, text: '' };
            yield { type: 'block-start', index: nextIndex, blockType: kind };
            nextIndex += 1;
        }
        open.text += text;
        yield kind === 'text'
            ? { type: 'text-delta', index: open.index, text }
            : { type: 'reasoning-delta', index: open.index, text };
    };
    const closeOpen = async function* () {
        if (open) {
            yield openBlockBlock(open);
            open = undefined;
        }
    };
    for await (const line of lines) {
        let event;
        try {
            event = JSON.parse(line);
        }
        catch {
            throw new LlmError(`thinktune: unparseable /api/chat line: ${line.slice(0, 200)}`, 'PROVIDER_HTTP_ERROR');
        }
        if (event.error) {
            throw new LlmError(`thinktune: ollama stream error: ${event.error}`, 'SERVER');
        }
        const message = event.message;
        if (message) {
            if (typeof message.thinking === 'string' && message.thinking.length > 0) {
                yield* emitText(message.thinking, 'reasoning');
            }
            if (typeof message.content === 'string' && message.content.length > 0) {
                // Legacy stacks inline the trace as think tags inside content.
                const split = contentSplitter.split(message.content);
                yield* emitText(split.reasoning, 'reasoning');
                yield* emitText(split.content, 'text');
            }
            if (Array.isArray(message.tool_calls)) {
                yield* closeOpen();
                for (const call of message.tool_calls) {
                    const name = call.function?.name;
                    if (typeof name !== 'string' || name.length === 0) {
                        throw new LlmError('thinktune: tool call without a function name', 'PROVIDER_HTTP_ERROR');
                    }
                    const raw = call.function?.arguments;
                    const argsJson = raw === undefined || raw === null
                        ? '{}'
                        : typeof raw === 'string' ? raw : JSON.stringify(raw);
                    const id = typeof call.id === 'string' && call.id.length > 0 ? call.id : `call_${toolCounter}`;
                    toolCounter += 1;
                    const index = nextIndex;
                    nextIndex += 1;
                    yield { type: 'block-start', index, blockType: 'tool-call' };
                    yield {
                        type: 'tool-call-delta',
                        index,
                        id: ToolCallId(id),
                        name,
                        argumentsDelta: argsJson,
                    };
                    yield {
                        type: 'block-end',
                        index,
                        block: { type: 'tool-call', id: ToolCallId(id), name, arguments: argsJson },
                    };
                }
            }
        }
        if (event.done) {
            yield* closeOpen();
            const inputTokens = typeof event.prompt_eval_count === 'number' ? event.prompt_eval_count : undefined;
            const outputTokens = typeof event.eval_count === 'number' ? event.eval_count : undefined;
            if (inputTokens !== undefined || outputTokens !== undefined) {
                const usage = {
                    inputTokens: inputTokens ?? 0,
                    outputTokens: outputTokens ?? 0,
                };
                if (inputTokens !== undefined && outputTokens !== undefined)
                    usage.totalTokens = inputTokens + outputTokens;
                yield { type: 'usage', usage };
            }
            yield { type: 'finish', reason: { kind: event.done_reason === 'length' ? 'max-tokens' : 'stop' } };
            finished = true;
        }
    }
    if (!finished) {
        throw new LlmError('thinktune: /api/chat stream ended without a done event', 'PROVIDER_HTTP_ERROR');
    }
}
