import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'

// The telescopic-text model (https://github.com/jackyzha0/telescopic-text) and its markup,
// shared by the TelescopicText transformer and the latent reader. Each caller renders the inline
// text of a line its own way (html fragments on note pages, the reader's markdown runs there).

interface Line {
  og: string // the original string to replace
  new: string // the replacement string
  replacements: Line[] // nested replacements to apply on the resultant line afterwards
}

export interface Content {
  text: string // Original string content in the line
  replacements: Line[] // Sections of the original text to replace/expand
}

interface NewContent {
  text: string
  expansions?: NewContent[]
  separator?: string
}

type TelescopicOutput = NewContent[]

interface TelescopeNode {
  depth: number
  telescopicOut: TelescopicOutput
}

/** inline markup of one piece of a line */
export type TelescopicInline = (value: string) => ElementContent[]

const BulletSeparators = ['*', '-', '+']
const RegexEscapedBulletSeparators = ['\\*', '-', '\\+']

// Parses the input string and returns the output as a structured data format.
function parseMarkdown(mdContent: string, onInvalid?: (line: string) => void): TelescopicOutput {
  const lines = mdContent.split('\n')
  // NOTE: this should handle normalizing the depth (if its an indented list)
  const root: TelescopicOutput = []
  const nodeStack: TelescopeNode[] = [{ depth: 0, telescopicOut: root }]

  // This is essentially a trie data structure to parse out all the bullet points
  // The algorithm works by assuming that any time you encounter a longer depth than the current one,
  // you are moving onto the next line.
  const firstNonEmptyLine = lines.find(l => l.trim().length > 0)
  const defaultDepth =
    // @ts-ignore
    firstNonEmptyLine?.match(`^\\s*(${RegexEscapedBulletSeparators.join('|')})`)?.[0]?.length - 1 ||
    0
  for (const line of lines) {
    const trimmedLine = line.trim()
    if (!trimmedLine.length) {
      continue
    }

    // validate that the trimmed line starts with the bullet indicators.
    if (!BulletSeparators.some(sep => trimmedLine.startsWith(sep))) {
      onInvalid?.(line)
      continue
    }

    // count all spaces in front to get current depth
    const currentDepth =
      line.match(`^\\s*(${RegexEscapedBulletSeparators.join('|')})`)![0].length - 1

    // if greater depth, attach on to the last one
    // else you can pop back up to one with 1 less depth
    while (
      nodeStack.length - 1 >= 0 &&
      currentDepth <= nodeStack[nodeStack.length - 1].depth &&
      nodeStack[nodeStack.length - 1].depth > 0
    ) {
      nodeStack.pop()
    }

    const { telescopicOut, ...restLastNode } = nodeStack[nodeStack.length - 1]
    const strippedLine = trimmedLine.substring(1).replace(/^\s+/, '')
    // Add current content / node to the stack
    const currentContent: NewContent = { text: strippedLine, expansions: [] }
    if (currentDepth === defaultDepth) {
      telescopicOut.push(currentContent)
      nodeStack[nodeStack.length - 1] = { ...restLastNode, telescopicOut }
    } else {
      // a deeper first line has no line above it to expand
      const last = telescopicOut[telescopicOut.length - 1]
      if (!last) {
        onInvalid?.(line)
        continue
      }
      last.expansions!.push(currentContent)
      // add this current one as a replacement to the last upper level one.
      const newNode = { depth: currentDepth, telescopicOut: [currentContent] }
      nodeStack.push(newNode)
    }
  }

  return root
}

// Ideally this would not be needed (just used to convert between data structures currently).
function outputToContent(output: TelescopicOutput, separator: string = ' '): Content {
  const parseReplacementsFromOutput = (out: TelescopicOutput): Line[] => {
    return out.flatMap((line: NewContent) => {
      if (!line.expansions?.length) {
        return []
      }
      const newText = line.expansions.map(line => line.text).join(separator)

      return [
        {
          og: line.text,
          new: newText,
          replacements: line.expansions?.length ? parseReplacementsFromOutput(line.expansions) : [],
        },
      ]
    })
  }
  const text = output.map((line: NewContent) => line.text).join(separator)
  const replacements: Line[] = parseReplacementsFromOutput(output)

  return { text, replacements }
}

/** a telescopic fence's bullet list as nested replacements */
export function telescopicContent(
  mdContent: string,
  separator: string = ' ',
  onInvalid?: (line: string) => void,
): Content {
  return outputToContent(parseMarkdown(mdContent, onInvalid), separator)
}

const emptyText = (): ElementContent => ({ type: 'text', value: '' })

/**
 * `div.telescope`: each expandable piece is `span.details.close` holding its summary (a focusable
 * span, so it stays in the text flow) and the expansion; telescopic.inline.ts opens them.
 */
export function telescopeHast(
  content: Content,
  inline: TelescopicInline,
  expand: 'click' | 'hover' = 'click',
): Element {
  function processContent(line: Content): ElementContent[] {
    let lastIndex = 0
    const nodes: ElementContent[] = []
    let lineText = line.text

    for (let i = 0; i < line.replacements.length; i++) {
      const replacement = line.replacements[i]

      const index = lineText.indexOf(replacement.og, lastIndex)
      const [before, ...after] = lineText.split(replacement.og)
      lineText = after.join(replacement.og)

      // Add text before replacement
      nodes.push(...inline(before))

      // Create telescopic node
      const summaryChildren = inline(replacement.og)
      const expandedChildren = processContent({
        text: replacement.new,
        replacements: replacement.replacements,
      })

      nodes.push(
        h('span', { class: 'details close' }, [
          // A span keeps the summary in the text flow; a button is an atomic inline-block box.
          h(
            'span',
            { class: 'summary', role: 'button', tabindex: 0, 'aria-expanded': 'false' },
            summaryChildren.length ? summaryChildren : [emptyText()],
          ),
          // Expanded content
          h(
            'span',
            { class: 'expanded' },
            expandedChildren.length ? expandedChildren : [emptyText()],
          ),
        ]),
      )
      lastIndex = index + replacement.og.length
    }

    // Add remaining text
    if (lastIndex < lineText.length) {
      nodes.push(...inline(lineText.slice(lastIndex)))
    }

    return nodes
  }

  return h('div.telescope', { 'data-expand': expand }, processContent(content))
}
