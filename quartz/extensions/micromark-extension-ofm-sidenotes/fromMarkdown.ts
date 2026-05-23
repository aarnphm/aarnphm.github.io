import type { PhrasingContent, Paragraph } from 'mdast'
import type { Extension as MdastExtension, CompileContext, Token } from 'mdast-util-from-markdown'
import type { Extension as MicromarkExtension } from 'micromark-util-types'
import { fromMarkdown } from 'mdast-util-from-markdown'
import type { Sidenote, SidenoteReference, SidenoteDefinition } from './types'

export interface FromMarkdownOptions {
  micromarkExtensions?: MicromarkExtension[]
  mdastExtensions?: MdastExtension[]
}

export function sidenoteFromMarkdown(options: FromMarkdownOptions = {}): MdastExtension {
  const micromarkExts = options.micromarkExtensions || []
  const mdastExts = options.mdastExtensions || []

  return {
    enter: {
      sidenote: enterSidenote,
      sidenoteProperties: enterProperties,
      sidenoteLabel: enterLabel,
      sidenoteContent: enterContent,
      sidenoteReference: enterReference,
      sidenoteReferenceLabel: enterLabel,
      sidenoteDefinition: enterDefinition,
      sidenoteDefinitionLabel: enterLabel,
    },
    exit: {
      sidenote: exitSidenote,
      sidenotePropertiesChunk: exitPropertiesChunk,
      sidenoteProperties: exitProperties,
      sidenoteLabelChunk: exitLabelChunk,
      sidenoteLabel: exitLabel,
      sidenoteContentChunk: exitContentChunk,
      sidenoteContent: exitContent,
      sidenoteReference: exitReference,
      sidenoteReferenceLabelChunk: exitReferenceLabelChunk,
      sidenoteReferenceLabel: exitReferenceLabel,
      sidenoteDefinition: exitDefinition,
      sidenoteDefinitionLabelChunk: exitDefinitionLabelChunk,
      sidenoteDefinitionLabel: exitDefinitionLabel,
    },
  }

  function enterSidenote(this: CompileContext, token: Token): undefined {
    const node: Sidenote = {
      type: 'sidenote',
      value: '',
      children: [],
      data: { sidenoteParsed: { raw: '', content: '' } },
    }
    this.enter(node as any, token)
    return undefined
  }

  function exitSidenote(this: CompileContext, token: Token): undefined {
    const node = this.stack[this.stack.length - 1] as any as Sidenote

    if (node) {
      node.value = this.sliceSerialize(token)

      if (node.data?.sidenoteParsed) {
        node.data.sidenoteParsed.raw = node.value
      }
    }

    this.exit(token)
    return undefined
  }

  function enterProperties(this: CompileContext): undefined {
    return undefined
  }

  function exitPropertiesChunk(this: CompileContext, token: Token): undefined {
    const node = this.stack[this.stack.length - 1] as any as Sidenote
    if (!node || !node.data?.sidenoteParsed) return undefined

    const raw = this.sliceSerialize(token)
    const parsed = parseProperties(raw)
    node.data.sidenoteParsed.properties = parsed
    return undefined
  }

  function exitProperties(this: CompileContext): undefined {
    return undefined
  }

  function enterLabel(this: CompileContext): undefined {
    return undefined
  }

  function exitLabelChunk(this: CompileContext, token: Token): undefined {
    const node = this.stack[this.stack.length - 1] as any as Sidenote
    if (!node || !node.data?.sidenoteParsed) return undefined

    const raw = this.sliceSerialize(token)
    node.data.sidenoteParsed.label = raw
    return undefined
  }

  function exitLabel(this: CompileContext): undefined {
    const node = this.stack[this.stack.length - 1] as any as Sidenote
    const labelRaw = node.data?.sidenoteParsed?.label || ''

    if (node.data?.sidenoteParsed) {
      try {
        const labelTree = fromMarkdown(labelRaw, {
          extensions: micromarkExts,
          mdastExtensions: mdastExts,
        })

        const nodes: PhrasingContent[] = []
        for (const child of labelTree.children) {
          if (child.type === 'paragraph') {
            nodes.push(...((child as Paragraph).children as PhrasingContent[]))
          } else if (isPhrasingContent(child)) {
            nodes.push(child as PhrasingContent)
          }
        }

        node.data.sidenoteParsed.labelNodes = nodes
      } catch {
        node.data.sidenoteParsed.labelNodes = [{ type: 'text', value: labelRaw }]
      }
    }

    return undefined
  }

  function enterContent(this: CompileContext): undefined {
    return undefined
  }

  function exitContentChunk(this: CompileContext, token: Token): undefined {
    const node = this.stack[this.stack.length - 1] as any as Sidenote
    if (!node || !node.data?.sidenoteParsed) return undefined

    const contentRaw = this.sliceSerialize(token)
    node.data.sidenoteParsed.content = contentRaw
    return undefined
  }

  function exitContent(this: CompileContext): undefined {
    const node = this.stack[this.stack.length - 1] as any as Sidenote
    const contentRaw = node.data?.sidenoteParsed?.content || ''

    if (node.data?.sidenoteParsed) {
      try {
        const contentTree = fromMarkdown(contentRaw, {
          extensions: micromarkExts,
          mdastExtensions: mdastExts,
        })

        const children: PhrasingContent[] = []
        for (const child of contentTree.children) {
          if (child.type === 'paragraph') {
            children.push(...((child as Paragraph).children as PhrasingContent[]))
          } else if (isPhrasingContent(child)) {
            children.push(child as PhrasingContent)
          }
        }

        node.children = children
      } catch {
        node.children = [{ type: 'text', value: contentRaw }]
      }
    }

    return undefined
  }

  // Reference Handlers
  function enterReference(this: CompileContext, token: Token): undefined {
    const node: SidenoteReference = { type: 'sidenoteReference', label: '', children: [] }
    this.enter(node as any, token)
    return undefined
  }

  function exitReference(this: CompileContext, token: Token): undefined {
    this.exit(token)
    return undefined
  }

  function exitReferenceLabelChunk(this: CompileContext, token: Token): undefined {
    const node = this.stack[this.stack.length - 1] as any as SidenoteReference
    if (!node) return undefined
    const raw = this.sliceSerialize(token)
    node.label = raw.startsWith('^') ? raw.slice(1) : raw
    return undefined
  }

  function exitReferenceLabel(this: CompileContext): undefined {
    const node = this.stack[this.stack.length - 1] as any as SidenoteReference
    parseLabelNodes(node, micromarkExts, mdastExts)
    return undefined
  }

  // Definition Handlers
  function enterDefinition(this: CompileContext, token: Token): undefined {
    const node: SidenoteDefinition = { type: 'sidenoteDefinition', label: '', children: [] }
    this.enter(node as any, token)
    return undefined
  }

  function exitDefinition(this: CompileContext, token: Token): undefined {
    this.exit(token)
    return undefined
  }

  function exitDefinitionLabelChunk(this: CompileContext, token: Token): undefined {
    const node = this.stack[this.stack.length - 1] as any as SidenoteDefinition
    if (!node) return undefined
    node.label = this.sliceSerialize(token)
    return undefined
  }

  function exitDefinitionLabel(this: CompileContext): undefined {
    const node = this.stack[this.stack.length - 1] as any as SidenoteDefinition
    parseLabelNodes(node, micromarkExts, mdastExts)
    return undefined
  }
}

function parseLabelNodes(
  node: SidenoteReference | SidenoteDefinition,
  micromarkExts: MicromarkExtension[],
  mdastExts: MdastExtension[],
) {
  const labelRaw = node.label || ''
  try {
    const labelTree = fromMarkdown(labelRaw, {
      extensions: micromarkExts,
      mdastExtensions: mdastExts,
    })

    const nodes: PhrasingContent[] = []
    for (const child of labelTree.children) {
      if (child.type === 'paragraph') {
        nodes.push(...((child as Paragraph).children as PhrasingContent[]))
      } else if (isPhrasingContent(child)) {
        nodes.push(child as PhrasingContent)
      }
    }
    node.labelNodes = nodes
  } catch {
    node.labelNodes = [{ type: 'text', value: labelRaw }]
  }
}

function parseProperties(raw: string): Record<string, string | string[]> {
  const props: Record<string, string | string[]> = {}
  let index = 0

  const isNameChar = (char: string | undefined) => char !== undefined && /[A-Za-z0-9_]/.test(char)
  const skipSpace = () => {
    while (index < raw.length && /\s/.test(raw[index])) index += 1
  }
  const hasNextKey = (start: number): boolean => {
    let cursor = start
    while (cursor < raw.length && /\s/.test(raw[cursor])) cursor += 1
    if (!isNameChar(raw[cursor])) return false
    while (cursor < raw.length && isNameChar(raw[cursor])) cursor += 1
    while (cursor < raw.length && /\s/.test(raw[cursor])) cursor += 1
    return raw[cursor] === ':'
  }
  const extractWikilinks = (value: string): string[] => {
    const links: string[] = []
    let cursor = 0
    while (cursor < value.length) {
      const start = value.indexOf('[[', cursor)
      if (start === -1) break
      const end = value.indexOf(']]', start + 2)
      if (end === -1) break
      links.push(value.slice(start, end + 2))
      cursor = end + 2
    }
    return links
  }

  while (index < raw.length) {
    skipSpace()
    if (raw[index] === ',') {
      index += 1
      continue
    }

    const keyStart = index
    while (index < raw.length && isNameChar(raw[index])) index += 1
    const key = raw.slice(keyStart, index).trim()
    skipSpace()
    if (!key || raw[index] !== ':') break
    index += 1
    skipSpace()

    const valueStart = index
    let wikilinkDepth = 0
    while (index < raw.length) {
      if (raw.startsWith('[[', index)) {
        wikilinkDepth += 1
        index += 2
        continue
      }
      if (raw.startsWith(']]', index) && wikilinkDepth > 0) {
        wikilinkDepth -= 1
        index += 2
        continue
      }
      if (raw[index] === ',' && wikilinkDepth === 0 && hasNextKey(index + 1)) break
      index += 1
    }

    const value = raw.slice(valueStart, index).trim()
    const wikilinks = extractWikilinks(value)
    props[key] = wikilinks.length > 0 ? wikilinks : value
  }

  return props
}

function isPhrasingContent(node: any): boolean {
  const phrasingTypes = new Set([
    'text',
    'emphasis',
    'strong',
    'delete',
    'inlineCode',
    'break',
    'link',
    'image',
    'linkReference',
    'imageReference',
    'html',
    'inlineMath',
  ])
  return phrasingTypes.has(node.type)
}
