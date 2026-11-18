import type { Element, ElementContent, Root as HtmlRoot } from 'hast'
import type { Parents as MdastParents, PhrasingContent, Root as MdRoot } from 'mdast'
import type { Handler, Options as ToHastOptions, State } from 'mdast-util-to-hast'
import { toHast } from 'mdast-util-to-hast'
import rehypeRaw from 'rehype-raw'
import remarkParse from 'remark-parse'
import {
  type Pluggable,
  type PluggableList,
  type Plugin,
  type PluginTuple,
  type Preset,
  type Transformer,
  type Processor,
  unified,
} from 'unified'
import { visit, EXIT } from 'unist-util-visit'
import { VFile } from 'vfile'
import type {
  Sidenote,
  SidenoteDefinition,
  SidenoteReference,
} from '../extensions/micromark-extension-ofm-sidenotes'
import type { BuildCtx } from '../util/ctx'
import { logBuildSpan, PerfTimer } from '../util/perf'
import { attachHastReferences, stripReferenceTransport } from '../util/references/hast'
import { referenceResolutionSignature } from '../util/references/markdown'
import { referenceHastHandler } from '../util/references/render'
import { isRecord } from '../util/type-guards'

export type QuartzMdProcessor = Processor<MdRoot, MdRoot, MdRoot>
export type QuartzHtmlProcessor = Processor<HtmlRoot, HtmlRoot, HtmlRoot>

function oneAsElementChildren(
  state: State,
  nodes: PhrasingContent[] | undefined,
  parent: MdastParents,
): ElementContent[] {
  if (!nodes) return []
  const children: ElementContent[] = []
  for (const node of nodes) {
    const result = state.one(node, parent)
    if (Array.isArray(result)) {
      children.push(...result)
    } else if (result) {
      children.push(result)
    }
  }
  return children
}

function isSidenote(node: unknown): node is Sidenote {
  return typeof node === 'object' && node !== null && 'type' in node && node.type === 'sidenote'
}

function isSidenoteReference(node: unknown): node is SidenoteReference {
  return (
    typeof node === 'object' && node !== null && 'type' in node && node.type === 'sidenoteReference'
  )
}

function isSidenoteDefinition(node: unknown): node is SidenoteDefinition {
  return (
    typeof node === 'object' &&
    node !== null &&
    'type' in node &&
    node.type === 'sidenoteDefinition'
  )
}

function sidenoteElement(state: State, node: Sidenote): Element {
  const content = state.all(node)
  const label = oneAsElementChildren(state, node.data?.sidenoteParsed?.labelNodes, node)

  return {
    type: 'element',
    tagName: 'span',
    properties: {
      className: ['sidenote-placeholder'],
      dataType: 'sidenote',
      sidenoteId: node.data?.sidenoteId,
      baseId: node.data?.baseId,
      forceInline: node.data?.forceInline,
      allowLeft: node.data?.allowLeft,
      allowRight: node.data?.allowRight,
      label: node.data?.label,
      internal: node.data?.internal,
    },
    children: [
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-label-hast'] },
        children: label,
      },
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-content-hast'] },
        children: content,
      },
    ],
  }
}

function sidenoteReferenceElement(state: State, node: SidenoteReference): Element {
  const label = oneAsElementChildren(state, node.labelNodes, node)

  return {
    type: 'element',
    tagName: 'span',
    properties: {
      className: ['sidenote-ref-placeholder'],
      dataType: 'sidenote-ref',
      label: node.label,
      sidenoteId: node.data?.sidenoteId,
      baseId: node.data?.baseId,
    },
    children: [
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-label-hast'] },
        children: label,
      },
    ],
  }
}

function sidenoteDefinitionElement(state: State, node: SidenoteDefinition): Element {
  const content = state.all(node)
  const label = oneAsElementChildren(state, node.labelNodes, node)

  return {
    type: 'element',
    tagName: 'div',
    properties: {
      className: ['sidenote-def-placeholder'],
      dataType: 'sidenote-def',
      label: node.label,
    },
    children: [
      {
        type: 'element',
        tagName: 'span',
        properties: { className: ['sidenote-label-hast'] },
        children: label,
      },
      {
        type: 'element',
        tagName: 'div',
        properties: { className: ['sidenote-content-hast'] },
        children: content,
      },
    ],
  }
}

const sidenoteHandler: Handler = (state, node) =>
  isSidenote(node) ? sidenoteElement(state, node) : undefined

const sidenoteReferenceHandler: Handler = (state, node) =>
  isSidenoteReference(node) ? sidenoteReferenceElement(state, node) : undefined

const sidenoteDefinitionHandler: Handler = (state, node) =>
  isSidenoteDefinition(node) ? sidenoteDefinitionElement(state, node) : undefined

const astToHastOptions: ToHastOptions = {
  allowDangerousHtml: true,
  handlers: {
    wikilink: referenceHastHandler,
    sidenote: sidenoteHandler,
    sidenoteReference: sidenoteReferenceHandler,
    sidenoteDefinition: sidenoteDefinitionHandler,
  },
}

export function mdastToHastRoot(ast: MdRoot): HtmlRoot {
  const tree = toHast(ast, astToHastOptions)
  if (tree.type !== 'root') {
    throw new Error('Expected Markdown root to convert to HTML root')
  }
  return tree
}

export function createMdProcessor(ctx: BuildCtx): QuartzMdProcessor {
  const transformers = ctx.cfg.plugins.transformers

  return (
    unified()
      // base Markdown -> MD AST
      .use(remarkParse)
      // MD AST -> MD AST transforms
      .use(
        transformers.flatMap(plugin => plugin.markdownPlugins?.(ctx) ?? []),
      ) as unknown as QuartzMdProcessor
    //  ^ sadly the typing of `use` is not smart enough to infer the correct type from our plugin list
  )
}

export function createHtmlProcessor(ctx: BuildCtx): QuartzHtmlProcessor {
  const transformers = ctx.cfg.plugins.transformers
  const rawProcessor = unified().use(rehypeRaw)
  return unified()
    .use(
      transformers.flatMap(plugin =>
        timedHtmlPlugins(ctx, `${plugin.name}:source`, plugin.htmlSourcePlugins?.(ctx) ?? []),
      ),
    )
    .use(() => async (tree: HtmlRoot, file: VFile) => {
      let raw = false
      visit(tree, 'raw', () => {
        raw = true
        return EXIT
      })
      const normalized = raw ? await rawProcessor.run(tree, file) : tree
      attachHastReferences(normalized, file)
      return normalized
    })
    .use(
      transformers.flatMap(plugin =>
        timedHtmlPlugins(ctx, plugin.name, plugin.htmlPlugins?.(ctx) ?? []),
      ),
    )
    .use(() => (tree: HtmlRoot, file: VFile) => {
      attachHastReferences(tree, file)
      stripReferenceTransport(tree, file)
      file.data.referenceResolutionSignature = referenceResolutionSignature(ctx, file)
    }) as unknown as QuartzHtmlProcessor
}

function timedHtmlPlugins(
  ctx: BuildCtx,
  transformerName: string,
  plugins: PluggableList,
): PluggableList {
  return plugins.map((plugin, index) => timedPluggable(ctx, `${transformerName}:${index}`, plugin))
}

function timedPluggable(ctx: BuildCtx, label: string, plugin: Pluggable): Pluggable {
  if (Array.isArray(plugin)) {
    const [attacher, ...parameters] = plugin
    return [timedPlugin(ctx, label, attacher), ...parameters] as PluginTuple
  }

  if (typeof plugin === 'function') {
    return timedPlugin(ctx, label, plugin)
  }

  if (isPreset(plugin)) {
    return {
      ...plugin,
      plugins: plugin.plugins?.map((item, index) => timedPluggable(ctx, `${label}.${index}`, item)),
    }
  }

  return plugin
}

function isPreset(plugin: Pluggable): plugin is Preset {
  return isRecord(plugin)
}

function timedPlugin(ctx: BuildCtx, label: string, plugin: Plugin<unknown[]>): Plugin<unknown[]> {
  return function wrappedPlugin(this: Processor, ...parameters: unknown[]) {
    const transformer = plugin.apply(this, parameters)
    if (!transformer) return transformer
    return timedTransformer(ctx, label, transformer)
  }
}

function timedTransformer(ctx: BuildCtx, label: string, transformer: Transformer): Transformer {
  return (tree, file) => {
    const perf = new PerfTimer()
    let logged = false
    const log = (targetFile: VFile) => {
      if (logged) return
      logged = true
      const slug = typeof targetFile.data.slug === 'string' ? targetFile.data.slug : targetFile.path
      logBuildSpan(ctx.argv, `hast:${label}`, slug, perf.elapsedMs())
    }

    const result = transformer(tree, file, () => undefined)
    if (result instanceof Promise) {
      return result.then(
        value => {
          log(file)
          return value
        },
        error => {
          log(file)
          throw error
        },
      )
    }

    log(file)
    return result
  }
}
