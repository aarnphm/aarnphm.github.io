/**
 * Display TeX from OCR'd pdfs, made something KaTeX will typeset. A page cut can leave a stray
 * `}` or a `\right|` whose `\left` sits in the row before, and a multi-line equation can carry a
 * `\tag` per line, where KaTeX takes one per display. The fix is local: unmatched closers drop,
 * what is still open at the end is closed, and the last number stands for the equation.
 */

// `x _ {a} _ {b}`: OCR stacks a second subscript (or superscript) where the page set a nested
// one; an empty group between them is what TeX needs
const DOUBLE = /(([_^])\s*(?:\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\\[a-zA-Z]+|[a-zA-Z0-9]))\s*(?=\2)/g

const TAG = /\\tag(\*?)\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g

// the tokens that open or close something: environments, \left/\right with their delimiter,
// row and cell breaks, any other control symbol (so `\{` and `\}` are not braces), braces
const TOKEN =
  /\\(?:begin|end)\s*\{[^{}]*\}|\\(?:left|right)(?![a-zA-Z])\s*(?:\\[a-zA-Z]+|\\.|[^\s])|\\\\|\\[a-zA-Z]+|\\.|[{}&]/g

interface Frame {
  /** what ends this group: `}` or `\end{name}`; the root has none */
  close: string
  /** `\left`s in this group still waiting for their `\right` */
  lefts: number
}

const shut = (frame: Frame) => '\\right.'.repeat(frame.lefts)

/** Unmatched `}`, `\end{…}` and `\right` drop; open groups and `\left`s close at the end. */
export function balanceTex(tex: string): string {
  const frames: Frame[] = [{ close: '', lefts: 0 }]
  let out = ''
  let last = 0
  for (const match of tex.matchAll(TOKEN)) {
    out += tex.slice(last, match.index)
    last = match.index + match[0].length
    const token = match[0]
    const top = frames[frames.length - 1]
    if (token === '{') {
      frames.push({ close: '}', lefts: 0 })
      out += token
    } else if (token === '}') {
      if (top.close !== '}') continue
      out += shut(top) + token
      frames.pop()
    } else if (token.startsWith('\\begin')) {
      frames.push({ close: token.replace(/^\\begin\s*/, '\\end'), lefts: 0 })
      out += token
    } else if (token.startsWith('\\end')) {
      const name = token.replace(/^\\end\s*/, '\\end')
      if (!frames.some(f => f.close === name)) continue
      while (frames[frames.length - 1].close !== name) {
        const frame = frames.pop()!
        out += shut(frame) + frame.close
      }
      out += shut(frames.pop()!) + token
    } else if (token === '\\\\' || token === '&') {
      // a \left cannot reach past its row or cell
      out += shut(top) + token
      top.lefts = 0
    } else if (/^\\left(?![a-zA-Z])/.test(token)) {
      top.lefts += 1
      out += token
    } else if (/^\\right(?![a-zA-Z])/.test(token)) {
      if (top.lefts > 0) {
        top.lefts -= 1
        out += token
      } else out += token.replace(/^\\right\s*/, ' ')
    } else out += token
  }
  out += tex.slice(last)
  while (frames.length > 1) {
    const frame = frames.pop()!
    out += shut(frame) + frame.close
  }
  return out + shut(frames[0])
}

/**
 * A display equation's TeX for KaTeX: lines joined; `#` escaped (OCR'd assembly listings write
 * `#3` for an immediate, which TeX reads as a macro parameter); stacked scripts split; of several
 * `\tag`s the last kept, at the end; braces and delimiters balanced. TeX that already typesets
 * comes out the same.
 */
export function displayTex(raw: string): string {
  let tex = raw
    .replace(/\s*\n\s*/g, ' ')
    .replace(/(?<!\\)#/g, '\\#')
    .replace(DOUBLE, '$1{}')
    .trim()
  const tags = [...tex.matchAll(TAG)]
  if (tags.length > 1) {
    const [, star, label] = tags[tags.length - 1]
    // the row the tag sat on often holds only it (and a full stop): drop the break before \end
    const body = tex
      .replace(TAG, ' ')
      .replace(/\\\\\s*(\\end\s*\{)/g, '$1')
      .trim()
    tex = `${body} \\tag${star}{${label.trim()}}`
  }
  return balanceTex(tex)
}
