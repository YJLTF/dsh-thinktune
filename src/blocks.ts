/**
 * Shared streaming text-block state for both wire parsers. Reasoning and
 * visible-text deltas accumulate into one open block that opens with
 * `block-start`, closes on kind switches and at stream end, and allocates its
 * block indices from the same counter as tool-call blocks so harness-side
 * ordering stays consistent.
 */
import type { StreamChunk } from '@deepseek-ai/dsh-llm'

type TextKind = 'reasoning' | 'text'

interface OpenTextBlock {
  index: number
  kind: TextKind
  text: string
}

export class TextBlockEmitter {
  private open?: OpenTextBlock
  private readonly indices: { next: number }

  constructor(indices: { next: number }) {
    this.indices = indices
  }

  /** Feed one delta; yields the block transitions plus the delta chunk. */
  *emit(text: string, kind: TextKind): Generator<StreamChunk> {
    if (text.length === 0) return
    if (this.open && this.open.kind !== kind) yield* this.close()
    let block = this.open
    if (!block) {
      const index = this.indices.next
      this.indices.next += 1
      block = { index, kind, text: '' }
      this.open = block
      yield { type: 'block-start', index, blockType: kind }
    }
    block.text += text
    yield kind === 'text'
      ? { type: 'text-delta', index: block.index, text }
      : { type: 'reasoning-delta', index: block.index, text }
  }

  /** Close the open block, if any. */
  *close(): Generator<StreamChunk> {
    const block = this.open
    if (!block) return
    this.open = undefined
    yield {
      type: 'block-end',
      index: block.index,
      block: block.kind === 'text'
        ? { type: 'text', text: block.text }
        : { type: 'reasoning', text: block.text },
    }
  }
}
