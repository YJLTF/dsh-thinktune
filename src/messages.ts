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
import { LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message, ToolSchema } from '@deepseek-ai/dsh-llm'
import type { RequestImageAttachment } from '@deepseek-ai/dsh-attachment'

/** One Ollama/OpenAI chat message in the intermediate shape both serializers accept. */
export interface WireMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  /** Assistant-side tool invocations (`function.arguments` stays a raw JSON string). */
  toolCalls?: WireToolCall[]
  /** Tool-side correlation with the originating tool call. */
  toolCallId?: string
  /** Assistant-side reasoning trace, retained only when `historyThinking` is `'keep'`. */
  thinking?: string
  /** User-side request images in wire order; each carries its encoded payload. */
  images?: WireImage[]
}

/** One encoded request-image payload. */
export interface WireImage {
  /** Verified media type of the encoded request version. */
  mediaType: string
  /** Canonical base64 of the encoded request bytes. */
  base64: string
}

export interface WireToolCall {
  id: string
  name: string
  arguments: string
}

export interface MapMessagesOptions {
  /** Whether assistant `reasoning` blocks replay to the provider. */
  historyThinking: 'strip' | 'keep'
  /** Prepared request-image versions keyed by attachment id. */
  requestImages: ReadonlyMap<string, RequestImageAttachment>
}

function textOfBlocks(blocks: readonly ContentBlock[], nested = false): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type === 'text') parts.push(block.text)
    else if (block.type === 'reasoning') parts.push(block.text)
    else if (block.type === 'tool-call') parts.push(nested ? `[tool call ${block.name}]` : '')
    else if (block.type === 'image') parts.push('[image]')
    else if (block.type === 'file') parts.push(`[file ${block.attachment.attachmentId}]`)
    else if (block.type === 'tool-result') parts.push(`[tool result for ${block.toolCallId}]`)
  }
  return parts.filter((part) => part.length > 0).join('\n')
}

function encodeImage(attachment: RequestImageAttachment): WireImage {
  return {
    mediaType: attachment.mediaType,
    base64: Buffer.from(attachment.data).toString('base64'),
  }
}

function mapMessage(message: Message, opts: MapMessagesOptions): WireMessage[] {
  if (message.role === 'system') {
    const content = textOfBlocks(message.content)
    return [{ role: 'system', content }]
  }

  if (message.role === 'user') {
    const wired: WireMessage[] = []
    const textBlocks: ContentBlock[] = []
    const images: WireImage[] = []
    for (const block of message.content) {
      if (block.type === 'tool-result') {
        wired.push({
          role: 'tool',
          content: textOfBlocks(block.content, true),
          toolCallId: block.toolCallId,
        })
      } else if (block.type === 'image') {
        const version = opts.requestImages.get(block.attachment.attachmentId)
        if (!version) {
          throw new LlmError(
            `thinktune: no prepared request image for attachment ${block.attachment.attachmentId}`,
            'UNSUPPORTED_CONTENT',
          )
        }
        images.push(encodeImage(version))
      } else if (block.type === 'file') {
        throw new LlmError(
          'thinktune: file blocks cannot reach the provider; harness request assembly should have projected them to text',
          'UNSUPPORTED_CONTENT',
        )
      } else {
        textBlocks.push(block)
      }
    }
    const text = textOfBlocks(textBlocks)
    if (text.length > 0 || images.length > 0) {
      wired.push({ role: 'user', content: text, ...(images.length > 0 ? { images } : {}) })
    }
    return wired
  }

  // Assistant message.
  const toolCalls: WireToolCall[] | undefined = message.content.some((block) => block.type === 'tool-call')
    ? message.content
        .filter((block): block is Extract<ContentBlock, { type: 'tool-call' }> => block.type === 'tool-call')
        .map((block) => ({ id: block.id, name: block.name, arguments: block.arguments }))
    : undefined
  const thinking = opts.historyThinking === 'keep'
    ? textOfBlocks(message.content.filter((block): block is Extract<ContentBlock, { type: 'reasoning' }> => block.type === 'reasoning'))
    : undefined
  const content = textOfBlocks(message.content.filter((block) => block.type !== 'reasoning' && block.type !== 'tool-call'))
  return [{ role: 'assistant', content, ...(toolCalls ? { toolCalls } : {}), ...(thinking ? { thinking } : {}) }]
}

/** Map the harness request history (plus the one-shot `system` prompt) to wire messages. */
export function mapMessages(
  options: Pick<GenerateOptions, 'system'> & { messages: readonly Message[] },
  opts: MapMessagesOptions,
): WireMessage[] {
  const wired: WireMessage[] = []
  if (options.system !== undefined && options.system.length > 0) {
    wired.push({ role: 'system', content: options.system })
  }
  for (const message of options.messages) wired.push(...mapMessage(message, opts))
  return wired
}

/** Map harness tool schemas to the shared OpenAI-style tools entry (empty → undefined). */
export function mapTools(tools: GenerateOptions['tools']): Array<Record<string, unknown>> | undefined {
  if (!tools || tools.length === 0) return undefined
  return tools.map((tool: ToolSchema) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))
}

/**
 * Append the Qwen3 soft switch (`/think` or `/no_think`) to the last user
 * message so the chat template steers thinking mode. Mutates the mapped list.
 */
export function applySoftSwitch(messages: WireMessage[], marker: string): void {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message && message.role === 'user') {
      message.content = `${message.content}\n${marker}`
      return
    }
  }
}
