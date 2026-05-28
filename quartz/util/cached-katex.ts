import type { Element, ElementContent, Root, RootContent } from 'hast'
import type { VFile } from 'vfile'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { toText } from 'hast-util-to-text'
import katex, { type KatexOptions } from 'katex'

type Options = Omit<KatexOptions, 'displayMode' | 'throwOnError'>
type Macros = NonNullable<Options['macros']>
type MacroValue = Macros[string]

const emptyOptions: Readonly<Options> = {}
const emptyClasses: ReadonlyArray<unknown> = []
const maxCacheEntries = 4096
const cache = new Map<string, ElementContent[]>()
const settingsIds = new WeakMap<Readonly<Options>, number>()
let nextSettingsId = 0

interface RenderContext {
  settingsId: number
  cacheable: boolean
}

function cloneNodes(nodes: ElementContent[]): ElementContent[] {
  return structuredClone(nodes)
}

function settingsId(settings: Readonly<Options>): number {
  const cached = settingsIds.get(settings)
  if (cached !== undefined) return cached
  const id = nextSettingsId
  nextSettingsId += 1
  settingsIds.set(settings, id)
  return id
}

function cacheKey(context: RenderContext, displayMode: boolean, value: string): string {
  return `${context.settingsId}\u0000${displayMode ? '1' : '0'}\u0000${value}`
}

function cachedNodes(key: string): ElementContent[] | undefined {
  const nodes = cache.get(key)
  if (!nodes) return undefined
  cache.delete(key)
  cache.set(key, nodes)
  return cloneNodes(nodes)
}

function setCachedNodes(key: string, nodes: ElementContent[]): void {
  cache.set(key, cloneNodes(nodes))
  if (cache.size <= maxCacheEntries) return
  const first = cache.keys().next().value
  if (first) cache.delete(first)
}

function parseKatexHtml(html: string): ElementContent[] {
  const root = fromHtmlIsomorphic(html, { fragment: true })
  return root.children.filter(
    (node: RootContent): node is ElementContent => node.type !== 'doctype',
  )
}

function cloneSettings(settings: Readonly<Options>): Options {
  const macros = settings.macros
  if (!macros) return { ...settings }
  return { ...settings, macros: { ...macros } }
}

function macroSnapshot(macros: Macros | undefined): Map<string, MacroValue> | undefined {
  if (!macros) return undefined
  const snapshot = new Map<string, MacroValue>()
  for (const key of Object.keys(macros)) {
    snapshot.set(key, macros[key])
  }
  return snapshot
}

function macrosChanged(
  macros: Macros | undefined,
  snapshot: Map<string, MacroValue> | undefined,
): boolean {
  if (!snapshot) return macros !== undefined
  if (!macros) return true
  const keys = Object.keys(macros)
  if (keys.length !== snapshot.size) return true
  for (const key of keys) {
    if (!snapshot.has(key) || snapshot.get(key) !== macros[key]) return true
  }
  return false
}

function renderToString(
  value: string,
  settings: Options,
  displayMode: boolean,
  throwOnError: boolean,
  context: RenderContext,
): string {
  const snapshot = macroSnapshot(settings.macros)
  try {
    return katex.renderToString(value, { ...settings, displayMode, throwOnError })
  } finally {
    if (macrosChanged(settings.macros, snapshot)) {
      context.cacheable = false
    }
  }
}

function renderKatex(
  value: string,
  settings: Options,
  displayMode: boolean,
  file: VFile,
  element: Element,
  context: RenderContext,
): ElementContent[] {
  const key = context.cacheable ? cacheKey(context, displayMode, value) : undefined
  if (key) {
    const cached = cachedNodes(key)
    if (cached) return cached
  }

  try {
    const nodes = parseKatexHtml(renderToString(value, settings, displayMode, true, context))
    if (key && context.cacheable) setCachedNodes(key, nodes)
    return nodes
  } catch (error) {
    const cause = error as Error
    const ruleId = cause.name.toLowerCase()

    file.message('Could not render math with KaTeX', {
      ancestors: [element],
      cause,
      place: element.position,
      ruleId,
      source: 'rehype-katex',
    })

    try {
      return parseKatexHtml(
        renderToString(value, { ...settings, strict: 'ignore' }, displayMode, false, context),
      )
    } catch {
      return [
        {
          type: 'element',
          tagName: 'span',
          properties: {
            className: ['katex-error'],
            style: `color:${settings.errorColor || '#cc0000'}`,
            title: String(error),
          },
          children: [{ type: 'text', value }],
        },
      ]
    }
  }
}

function elementClasses(element: Element): ReadonlyArray<unknown> {
  return Array.isArray(element.properties.className) ? element.properties.className : emptyClasses
}

function mathCodeChild(element: Element): Element | undefined {
  if (element.tagName !== 'pre') return undefined
  return element.children.find(
    (child): child is Element =>
      child.type === 'element' &&
      child.tagName === 'code' &&
      elementClasses(child).includes('language-math'),
  )
}

function transformChild(
  parent: Root | Element,
  index: number,
  child: Element,
  settings: Options,
  file: VFile,
  context: RenderContext,
): boolean {
  const preCode = mathCodeChild(child)
  if (preCode) {
    const result = renderKatex(
      toText(child, { whitespace: 'pre' }),
      settings,
      true,
      file,
      preCode,
      context,
    )
    parent.children.splice(index, 1, ...result)
    return true
  }

  const classes = elementClasses(child)
  const mathDisplay = classes.includes('math-display')
  const mathInline = classes.includes('math-inline')
  if (!mathDisplay && !mathInline) {
    return false
  }

  const result = renderKatex(
    toText(child, { whitespace: 'pre' }),
    settings,
    mathDisplay,
    file,
    child,
    context,
  )
  parent.children.splice(index, 1, ...result)
  return true
}

function transformChildren(
  parent: Root | Element,
  settings: Options,
  file: VFile,
  context: RenderContext,
): void {
  for (let index = 0; index < parent.children.length; index += 1) {
    const child = parent.children[index]
    if (child.type !== 'element') continue
    if (transformChild(parent, index, child, settings, file, context)) {
      continue
    }
    transformChildren(child, settings, file, context)
  }
}

export default function cachedKatex(options?: Readonly<Options> | null) {
  const settings = options || emptyOptions
  const id = settingsId(settings)

  return function transform(tree: Root, file: VFile): undefined {
    transformChildren(tree, cloneSettings(settings), file, { settingsId: id, cacheable: true })
    return undefined
  }
}
