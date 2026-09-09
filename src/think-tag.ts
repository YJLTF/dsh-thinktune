/**
 * Incremental `<think>…</think>` splitter for content streams. Some serving
 * stacks (older Ollama OpenAI-compat layers, llama.cpp, vLLM) deliver the
 * reasoning trace as literal think tags inside `content` instead of a
 * dedicated reasoning field; the harness wants it as `reasoning-delta`. Holds
 * back a partial tag at chunk boundaries and flushes on stream end.
 */
export class ThinkTagSplitter {
  private inThink = false
  private holdback = ''

  private static readonly OPEN = '<think>'
  private static readonly CLOSE = '</think>'

  /** Feed one content chunk; returns the reasoning and visible text it contains. */
  split(text: string): { reasoning: string; content: string } {
    let data = this.holdback + text
    this.holdback = ''
    let reasoning = ''
    let content = ''
    while (data.length > 0) {
      if (this.inThink) {
        const close = data.indexOf(ThinkTagSplitter.CLOSE)
        if (close >= 0) {
          reasoning += data.slice(0, close)
          data = data.slice(close + ThinkTagSplitter.CLOSE.length)
          this.inThink = false
          continue
        }
        const hold = holdbackLength(data, ThinkTagSplitter.CLOSE)
        if (hold > 0) {
          reasoning += data.slice(0, data.length - hold)
          this.holdback = data.slice(data.length - hold)
          data = ''
          break
        }
        reasoning += data
        data = ''
        break
      }
      const open = data.indexOf(ThinkTagSplitter.OPEN)
      const close = data.indexOf(ThinkTagSplitter.CLOSE)
      if (close >= 0 && (open === -1 || close < open)) {
        // An unmatched close tag: tolerate servers that close without opening.
        data = data.slice(close + ThinkTagSplitter.CLOSE.length)
        continue
      }
      if (open >= 0) {
        content += data.slice(0, open)
        data = data.slice(open + ThinkTagSplitter.OPEN.length)
        this.inThink = true
        continue
      }
      const lt = data.indexOf('<')
      if (lt === -1) {
        content += data
        data = ''
        break
      }
      const tail = data.slice(lt)
      const hold = holdbackLength(tail, ThinkTagSplitter.OPEN)
      if (hold > 0) {
        content += data.slice(0, lt)
        this.holdback = data.slice(lt)
        data = ''
        break
      }
      // A literal '<' that starts no tag: emit it and keep scanning.
      content += data.slice(0, lt + 1)
      data = data.slice(lt + 1)
    }
    return { reasoning, content }
  }

  /** Flush any held-back partial tag at end of stream. */
  flush(): { reasoning: string; content: string } {
    const data = this.holdback
    this.holdback = ''
    return this.inThink ? { reasoning: data, content: '' } : { reasoning: '', content: data }
  }
}

/** Length of the longest suffix of `data` that is a proper prefix of `tag`. */
function holdbackLength(data: string, tag: string): number {
  const max = Math.min(data.length, tag.length - 1)
  for (let length = max; length > 0; length -= 1) {
    if (tag.startsWith(data.slice(data.length - length))) return length
  }
  return 0
}
