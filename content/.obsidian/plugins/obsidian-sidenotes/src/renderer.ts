import { MarkdownPostProcessorContext, MarkdownRenderer, Plugin } from "obsidian"
import { extractSegments } from "./parser"
import type { ParsedSidenote, SidenoteProperties } from "./types"

const KEYWORD = "{{sidenotes"

export async function processSidenotes(
  el: HTMLElement,
  ctx: MarkdownPostProcessorContext,
  plugin: Plugin,
): Promise<void> {
  let nodes = collectTextNodes(el)
  if (nodes.length === 0) return

  let counter = 0
  let mutated = true

  while (mutated) {
    mutated = false

    outer: for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]
      const value = node.nodeValue || ""
      let searchIndex = value.indexOf(KEYWORD)

      while (searchIndex !== -1) {
        const nextCounter = counter + 1
        const replaced = await parseAcrossNodes(nodes, i, searchIndex, ctx, plugin, nextCounter)
        if (replaced) {
          counter = nextCounter
          mutated = true
          nodes = collectTextNodes(el)
          break outer
        }
        searchIndex = value.indexOf(KEYWORD, searchIndex + KEYWORD.length)
      }
    }
  }
}

function shouldSkip(node: Text): boolean {
  const parent = node.parentElement
  if (!parent) return false
  if (parent.closest(".sidenote")) return true
  return Boolean(parent.closest("code, pre, .cm-editor, mjx-container, .math"))
}

function deriveLabel(rawLabel: string | undefined, fallback: number): { text: string; auto: boolean } {
  if (rawLabel && rawLabel.trim().length > 0) {
    const trimmed = rawLabel.trim()
    const wikilinkMatch = /^\[\[([^\]]+)\]\]$/.exec(trimmed)
    if (wikilinkMatch) {
      const parts = wikilinkMatch[1].split("|")
      const alias = parts[1] ?? parts[0]
      return { text: alias, auto: false }
    }
    return { text: trimmed, auto: false }
  }

  return { text: String(fallback), auto: true }
}

async function buildSidenoteElement(
  parsed: ParsedSidenote,
  label: { text: string; auto: boolean },
  sourcePath: string,
  plugin: Plugin,
): Promise<HTMLElement> {
  const wrapper = document.createElement("span")
  wrapper.classList.add("sidenote")

  const labelEl = document.createElement("span")
  labelEl.classList.add("sidenote-label")
  if (label.auto) {
    labelEl.dataset.auto = "true"
  }
  labelEl.setAttribute("role", "button")
  labelEl.tabIndex = 0
  labelEl.textContent = label.text

  const contentEl = document.createElement("span")
  contentEl.classList.add("sidenote-content")
  applyLayoutClasses(parsed.properties, contentEl)
  contentEl.hidden = true

  await renderMarkdownInto(parsed.content, contentEl, sourcePath, plugin)

  const internalLinks = normalizeInternal(parsed.properties?.internal)
  if (internalLinks.length > 0) {
    const internalEl = document.createElement("div")
    internalEl.classList.add("sidenote-internal")
    const markdown = `linked notes: ${internalLinks.join(", ")}`
    await renderMarkdownInto(markdown, internalEl, sourcePath, plugin)
    contentEl.appendChild(internalEl)
  }

  wrapper.appendChild(labelEl)
  wrapper.appendChild(contentEl)
  attachToggle(wrapper, labelEl, contentEl)
  return wrapper
}

async function renderMarkdownInto(
  markdown: string,
  target: HTMLElement,
  sourcePath: string,
  plugin: Plugin,
): Promise<void> {
  const container = document.createElement("div")
  await MarkdownRenderer.renderMarkdown(markdown, container, sourcePath, plugin)
  const nodes = unwrap(container)
  nodes.forEach((child) => target.appendChild(child))
}

function unwrap(container: HTMLElement): Node[] {
  if (container.childElementCount === 1 && container.firstElementChild?.tagName === "P") {
    return Array.from(container.firstElementChild.childNodes)
  }
  return Array.from(container.childNodes)
}

function applyLayoutClasses(props: SidenoteProperties | undefined, el: HTMLElement): void {
  const inline = isTruthy(props?.inline) || isTruthy(props?.dropdown)
  const allowLeft = parseBoolean(props?.left, true)
  const allowRight = parseBoolean(props?.right, true)

  if (inline || (!allowLeft && !allowRight)) {
    el.classList.add("sidenote-inline")
    return
  }

  if (allowLeft && !allowRight) {
    el.classList.add("sidenote-left")
    return
  }

  if (!allowLeft && allowRight) {
    el.classList.add("sidenote-right")
    return
  }

  el.classList.add("sidenote-right")
}

function isTruthy(value: string | string[] | undefined): boolean {
  if (Array.isArray(value)) {
    value = value[0]
  }
  if (value === undefined) return false
  const normalized = String(value).trim().toLowerCase()
  return normalized === "true" || normalized === "1" || normalized === "yes" || normalized === "inline"
}

function parseBoolean(value: string | string[] | undefined, defaultValue: boolean): boolean {
  if (Array.isArray(value)) {
    value = value[0]
  }
  if (value === undefined) return defaultValue
  const normalized = String(value).trim().toLowerCase()
  if (["false", "0", "no", "off"].includes(normalized)) return false
  if (["true", "1", "yes", "on"].includes(normalized)) return true
  return defaultValue
}

function normalizeInternal(value: string | string[] | undefined): string[] {
  if (Array.isArray(value)) return value
  if (!value) return []
  return value
    .split(",")
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
}

function attachToggle(wrapper: HTMLElement, labelEl: HTMLElement, contentEl: HTMLElement): void {
  let open = false

  const setOpen = (next: boolean) => {
    open = next
    wrapper.classList.toggle("open", open)
    contentEl.hidden = !open
    labelEl.setAttribute("aria-expanded", String(open))
  }

  const toggle = () => setOpen(!open)

  labelEl.addEventListener("click", (event) => {
    event.preventDefault()
    toggle()
  })

  labelEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault()
      toggle()
    }
  })

  setOpen(false)
}

function collectTextNodes(root: HTMLElement): Text[] {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const nodes: Text[] = []
  let current: Node | null
  while ((current = walker.nextNode())) {
    const textNode = current as Text
    if (!textNode.nodeValue || !textNode.nodeValue.includes(KEYWORD)) continue
    if (shouldSkip(textNode)) continue
    nodes.push(textNode)
  }
  return nodes
}

async function parseAcrossNodes(
  nodes: Text[],
  startIndex: number,
  startOffset: number,
  ctx: MarkdownPostProcessorContext,
  plugin: Plugin,
  counter: number,
): Promise<boolean> {
  const startText = nodes[startIndex].nodeValue || ""
  let aggregated = startText.slice(startOffset)
  const lengths: number[] = [startText.length - startOffset]

  const boundary = getBlockAncestor(nodes[startIndex])

  let idx = startIndex
  while (aggregated.indexOf("}}") === -1 && idx + 1 < nodes.length) {
    idx += 1
    if (getBlockAncestor(nodes[idx]) !== boundary) break
    const text = nodes[idx].nodeValue || ""
    aggregated += text
    lengths.push(text.length)
  }

  const segments = extractSegments(aggregated)
  const sidenote = segments.find((s) => s.type === "sidenote")
  if (!sidenote || sidenote.type !== "sidenote") return false

  const rawLen = sidenote.data.raw.length
  let remaining = rawLen

  let endNodeIndex = startIndex
  let endOffset = startOffset

  if (remaining <= lengths[0]) {
    endOffset = startOffset + remaining
  } else {
    remaining -= lengths[0]
    for (let j = 1; j < lengths.length; j++) {
      if (remaining <= lengths[j]) {
        endNodeIndex = startIndex + j
        endOffset = remaining
        break
      }
      remaining -= lengths[j]
    }
  }

  const range = document.createRange()
  range.setStart(nodes[startIndex], startOffset)
  range.setEnd(nodes[endNodeIndex], endOffset)

  const label = deriveLabel(sidenote.data.label, counter)
  const fragment = document.createDocumentFragment()
  const rendered = await buildSidenoteElement(sidenote.data, label, ctx.sourcePath, plugin)
  fragment.appendChild(rendered)

  range.deleteContents()
  range.insertNode(fragment)
  return true
}

function getBlockAncestor(node: Node): Element | null {
  let el: Element | null = node.parentElement
  while (el) {
    const tag = el.tagName?.toLowerCase()
    if (["p", "li", "blockquote", "div", "section", "article", "td", "th"].includes(tag)) {
      return el
    }
    el = el.parentElement
  }
  return null
}
