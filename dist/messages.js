/**
 * Harness message → provider-neutral wire message mapping, shared by the
 * Ollama-native and OpenAI-compatible protocols. Durable `FileBlock`
 * references are projected to deterministic handle text by harness request
 * assembly before an adapter sees them; `ImageBlock` references resolve
 * through the attachment service into per-request re-encoded versions that
 * both wire shapes carry (native `images` arrays and OpenAI `image_url`
 * data URIs). Images nested inside tool results degrade to text placeholders:
 * Ollama vision input only exists on user messages.
 */
import { LlmError } from '@deepseek-ai/dsh-llm';
function textOfBlocks(blocks, nested = false) {
    const parts = [];
    for (const block of blocks) {
        if (block.type === 'text')
            parts.push(block.text);
        else if (block.type === 'reasoning')
            parts.push(block.text);
        else if (block.type === 'tool-call')
            parts.push(nested ? `[tool call ${block.name}]` : '');
        else if (block.type === 'image')
            parts.push('[image]');
        else if (block.type === 'file')
            parts.push(`[file ${block.attachment.attachmentId}]`);
        else if (block.type === 'tool-result')
            parts.push(`[tool result for ${block.toolCallId}]`);
    }
    return parts.filter((part) => part.length > 0).join('\n');
}
function encodeImage(attachment) {
    return {
        mediaType: attachment.mediaType,
        base64: Buffer.from(attachment.data).toString('base64'),
    };
}
function mapMessage(message, opts) {
    if (message.role === 'system') {
        const content = textOfBlocks(message.content);
        return [{ role: 'system', content }];
    }
    if (message.role === 'user') {
        const wired = [];
        const textBlocks = [];
        const images = [];
        for (const block of message.content) {
            if (block.type === 'tool-result') {
                wired.push({
                    role: 'tool',
                    content: textOfBlocks(block.content, true),
                    toolCallId: block.toolCallId,
                });
            }
            else if (block.type === 'image') {
                const version = opts.requestImages.get(block.attachment.attachmentId);
                if (!version) {
                    throw new LlmError(`thinktune: no prepared request image for attachment ${block.attachment.attachmentId}`, 'UNSUPPORTED_CONTENT');
                }
                images.push(encodeImage(version));
            }
            else if (block.type === 'file') {
                throw new LlmError('thinktune: file blocks cannot reach the provider; harness request assembly should have projected them to text', 'UNSUPPORTED_CONTENT');
            }
            else {
                textBlocks.push(block);
            }
        }
        const text = textOfBlocks(textBlocks);
        if (text.length > 0 || images.length > 0) {
            wired.push({ role: 'user', content: text, ...(images.length > 0 ? { images } : {}) });
        }
        return wired;
    }
    // Assistant message.
    const toolCalls = message.content.some((block) => block.type === 'tool-call')
        ? message.content
            .filter((block) => block.type === 'tool-call')
            .map((block) => ({ id: block.id, name: block.name, arguments: block.arguments }))
        : undefined;
    const thinking = opts.historyThinking === 'keep'
        ? textOfBlocks(message.content.filter((block) => block.type === 'reasoning'))
        : undefined;
    const content = textOfBlocks(message.content.filter((block) => block.type !== 'reasoning' && block.type !== 'tool-call'));
    return [{ role: 'assistant', content, ...(toolCalls ? { toolCalls } : {}), ...(thinking ? { thinking } : {}) }];
}
/** Map the harness request history (plus the one-shot `system` prompt) to wire messages. */
export function mapMessages(options, opts) {
    const wired = [];
    if (options.system !== undefined && options.system.length > 0) {
        wired.push({ role: 'system', content: options.system });
    }
    for (const message of options.messages)
        wired.push(...mapMessage(message, opts));
    return wired;
}
/** Map harness tool schemas to the shared OpenAI-style tools entry (empty → undefined). */
export function mapTools(tools) {
    if (!tools || tools.length === 0)
        return undefined;
    return tools.map((tool) => ({
        type: 'function',
        function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
        },
    }));
}
/**
 * Append the Qwen3 soft switch (`/think` or `/no_think`) to the last user
 * message so the chat template steers thinking mode. Mutates the mapped list.
 */
export function applySoftSwitch(messages, marker) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
        const message = messages[index];
        if (message && message.role === 'user') {
            message.content = `${message.content}\n${marker}`;
            return;
        }
    }
}
