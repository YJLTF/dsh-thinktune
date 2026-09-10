export class TextBlockEmitter {
    open;
    indices;
    constructor(indices) {
        this.indices = indices;
    }
    /** Feed one delta; yields the block transitions plus the delta chunk. */
    *emit(text, kind) {
        if (text.length === 0)
            return;
        if (this.open && this.open.kind !== kind)
            yield* this.close();
        let block = this.open;
        if (!block) {
            const index = this.indices.next;
            this.indices.next += 1;
            block = { index, kind, text: '' };
            this.open = block;
            yield { type: 'block-start', index, blockType: kind };
        }
        block.text += text;
        yield kind === 'text'
            ? { type: 'text-delta', index: block.index, text }
            : { type: 'reasoning-delta', index: block.index, text };
    }
    /** Close the open block, if any. */
    *close() {
        const block = this.open;
        if (!block)
            return;
        this.open = undefined;
        yield {
            type: 'block-end',
            index: block.index,
            block: block.kind === 'text'
                ? { type: 'text', text: block.text }
                : { type: 'reasoning', text: block.text },
        };
    }
}
