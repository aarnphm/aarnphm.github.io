import type { Code, Effects, Token } from 'micromark-util-types'

/** Link one text chunk per physical line for micromark's subtokenizer. */
export function inlineChunks(effects: Effects) {
  let previous: Token | undefined
  let open = false
  function close() {
    if (open) effects.exit('chunkText')
    open = false
  }
  return {
    close,
    consume(code: Code) {
      if (!open) {
        const current = effects.enter('chunkText', { contentType: 'text', previous })
        if (previous) previous.next = current
        previous = current
        open = true
      }
      effects.consume(code)
      if (code === -5 || code === -4 || code === -3) close()
    },
  }
}
