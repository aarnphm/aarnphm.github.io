import {
  MarkdownRenderChild,
  MarkdownRenderer,
  type MarkdownPostProcessorContext,
  type Plugin,
} from 'obsidian'
import type { ParsedSidenote, SidenoteMatch, SidenoteProperties } from './types'
import { findSidenotes } from './parser'

const OPEN = '{{sidenotes'

export function createSidenote(
  parsed: ParsedSidenote,
  number: number,
  plugin: Plugin,
  sourcePath: string,
  resized: () => void = () => {},
  renderedContent?: DocumentFragment,
): { element: HTMLElement; child: MarkdownRenderChild; ready: Promise<void> } {
  const element = document.createElement('span')
  element.className = 'sidenote garden-plugin-ui'
  const child = new MarkdownRenderChild(element)
  let active = true
  child.register(() => {
    active = false
  })

  const label = element.createEl('button', { cls: 'sidenote-label', attr: { type: 'button' } })
  const rawLabel = parsed.label?.trim()
  const wikilink = rawLabel && /^\[\[([^\]]+)\]\]$/.exec(rawLabel)
  label.textContent = wikilink
    ? (wikilink[1].split('|')[1] ?? wikilink[1])
    : rawLabel || String(number)
  if (!rawLabel) label.dataset.auto = 'true'

  const content = element.createEl('span', { cls: 'sidenote-content' })
  content.id = `garden-sidenote-${crypto.randomUUID()}`
  applyLayoutClasses(parsed.properties, element, content)
  content.hidden = true
  label.setAttribute('aria-controls', content.id)
  label.setAttribute('aria-expanded', 'false')

  const setOpen = (open: boolean) => {
    element.dataset.sidenoteOpen = String(open)
    element.classList.toggle('open', open)
    content.hidden = !open
    label.setAttribute('aria-expanded', String(open))
    resized()
  }
  child.registerDomEvent(label, 'click', event => {
    event.preventDefault()
    setOpen(content.hidden)
  })
  child.registerDomEvent(element, 'keydown', event => {
    if (event.key === 'Escape' && !content.hidden) {
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
      label.focus()
    }
  })

  const ready = (async () => {
    if (renderedContent) content.append(renderedContent)
    else await MarkdownRenderer.render(plugin.app, parsed.content, content, sourcePath, child)
    if (!active) return
    const links = normalizeInternal(parsed.properties?.internal)
    if (links.length) {
      const linked = content.createEl('span', { cls: 'sidenote-internal' })
      await MarkdownRenderer.render(
        plugin.app,
        `linked notes: ${links.join(', ')}`,
        linked,
        sourcePath,
        child,
      )
    }
    if (active) resized()
  })()
  return { element, child, ready }
}

export async function processSidenotes(
  el: HTMLElement,
  ctx: MarkdownPostProcessorContext,
  plugin: Plugin,
): Promise<void> {
  if (el.closest('.sidenote, .cm-editor')) return
  const section = ctx.getSectionInfo(el)
  const source = section?.text
    .split('\n')
    .slice(section.lineStart, section.lineEnd + 1)
    .join('\n')
  const sourceNotes = source ? findSidenotes(source) : []
  const preceding = section ? section.text.split('\n').slice(0, section.lineStart).join('\n') : ''
  let number = findSidenotes(preceding).length

  // Obsidian has already rendered links, emphasis, embeds, and math. Preserve their rendered markup.
  for (;;) {
    const nodes = collectTextNodes(el)
    const text = nodes.map(node => node.data).join('')
    const original = sourceNotes[0]
    const match = original ? renderedMatch(text, original.data) : findSidenotes(text)[0]
    if (!match) return
    const range = rangeAt(nodes, match.from, match.to)
    const contentRange = rangeAt(nodes, match.contentFrom, match.to - 2)
    if (!range || !contentRange) return
    const parsed = sourceNotes.shift()?.data ?? match.data
    const rendered = createSidenote(
      parsed,
      ++number,
      plugin,
      ctx.sourcePath,
      undefined,
      contentRange.cloneContents(),
    )
    ctx.addChild(rendered.child)
    range.deleteContents()
    range.insertNode(rendered.element)
    await rendered.ready
  }
}

function renderedMatch(text: string, data: ParsedSidenote): SidenoteMatch | undefined {
  let from = text.indexOf(OPEN)
  if (from === -1) return
  for (;;) {
    const next = text.indexOf(OPEN, from + OPEN.length)
    const close = text.indexOf('}}', from + OPEN.length)
    if (next === -1 || (close !== -1 && close < next)) break
    from = next
  }
  let header = from + OPEN.length
  if (text[header] === '<') {
    const end = text.indexOf('>', header)
    if (end === -1) return
    header = end + 1
  }
  // Obsidian may consume the opening label brackets as part of a wikilink.
  const separator =
    data.label === undefined ? text.indexOf(':', header) : text.indexOf(']:', header)
  if (separator === -1) return
  const contentFrom = separator + (data.label === undefined ? 1 : 2)
  const end = text.indexOf('}}', contentFrom)
  if (end === -1) return
  return { from, to: end + 2, contentFrom, data }
}

function collectTextNodes(root: HTMLElement): Text[] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const nodes: Text[] = []
  let node: Node | null
  while ((node = walker.nextNode())) {
    if (!(node instanceof Text)) continue
    if (node.parentElement?.closest('.sidenote, code, pre, mjx-container, .math, .frontmatter'))
      continue
    nodes.push(node)
  }
  return nodes
}

function rangeAt(nodes: Text[], from: number, to: number): Range | null {
  const range = document.createRange()
  let offset = 0
  let started = false
  for (const [index, node] of nodes.entries()) {
    const end = offset + node.length
    if (!started && from >= offset && from <= end) {
      range.setStart(node, from - offset)
      started = true
    }
    if (started && (to < end || (to === end && index === nodes.length - 1))) {
      range.setEnd(node, to - offset)
      return range
    }
    offset = end
  }
  return null
}

function applyLayoutClasses(
  props: SidenoteProperties | undefined,
  wrapper: HTMLElement,
  content: HTMLElement,
): void {
  const enabled = (value: string | string[] | undefined, fallback = false) => {
    const normalized = (Array.isArray(value) ? value[0] : value)?.trim().toLowerCase()
    if (normalized === undefined) return fallback
    return ['true', '1', 'yes', 'on', 'inline'].includes(normalized)
  }
  const left = enabled(props?.left, true)
  const right = enabled(props?.right, true)
  const inline = enabled(props?.inline) || enabled(props?.dropdown) || (!left && !right)
  wrapper.dataset.allowLeft = String(left)
  wrapper.dataset.allowRight = String(right)
  content.classList.add(
    inline ? 'sidenote-inline' : left && !right ? 'sidenote-left' : 'sidenote-right',
  )
}

function normalizeInternal(value: string | string[] | undefined): string[] {
  if (Array.isArray(value)) return value
  return (
    value
      ?.split(',')
      .map(link => link.trim())
      .filter(Boolean) ?? []
  )
}
