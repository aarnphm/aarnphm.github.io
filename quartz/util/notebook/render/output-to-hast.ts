import type { Element, ElementContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import katex from 'katex'
import type { ErrorOutput, MimeBundle, Output, StreamOutput } from '../types'
import { customMacros, katexOptions } from '../../../cfg'
import { escapeHTML } from '../../escape'
import {
  NOTEBOOK_IMAGE_BINARY_MIME_TYPES,
  NOTEBOOK_OUTPUT_MIME_PRIORITY,
  type NotebookImageBinaryMimeType,
} from '../mime'

export type NotebookOutputHtml = { label: string; html: string }

const ipythonDisplayPlaceholder = /^<IPython\.core\.display\.[A-Za-z0-9_]+ object>$/

const ansiPattern = new RegExp(`${String.fromCharCode(27)}\\[[0-?]*[ -/]*[@-~]`, 'g')

function stripAnsi(value: string): string {
  return value.replace(ansiPattern, '')
}

function classToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, '-') || 'output'
}

function trimTrailing(value: string): string {
  return value.replace(/\s+$/, '')
}

function escapedPreText(value: string): string {
  return escapeHTML(value).replaceAll('\r\n', '\n').replaceAll('\r', '\n').replaceAll('\n', '&#10;')
}

function mimeText(value: string | readonly string[] | undefined): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.join('')
  return ''
}

function preSamp(classes: string[], name: string, text: string): ElementContent {
  return h(
    'pre',
    { className: classes, 'data-output-name': name },
    h('samp', {}, trimTrailing(text)),
  )
}

function outputName(node: ElementContent): string {
  if (node.type !== 'element') return 'output'
  const value = node.properties?.dataOutputName ?? node.properties?.['data-output-name']
  return typeof value === 'string' && value.trim() ? value : 'output'
}

function classNames(node: Element): string[] {
  const value = node.properties?.className
  if (typeof value === 'string') return value.split(/\s+/).filter(Boolean)
  if (!Array.isArray(value)) return []
  return value.filter((item): item is string => typeof item === 'string')
}

function textContent(node: ElementContent): string | undefined {
  if (node.type === 'text') return node.value
}

function serializedPreSamp(node: ElementContent): string | undefined {
  if (node.type !== 'element' || node.tagName !== 'pre') return undefined
  const classes = classNames(node)
  if (!classes.includes('notebook-output')) return undefined
  if (node.children.length !== 1) return undefined
  const samp = node.children[0]
  if (samp.type !== 'element' || samp.tagName !== 'samp') return undefined
  const text = samp.children.map(textContent)
  if (text.some(value => value === undefined)) return undefined
  const classAttribute = escapeHTML(classes.join(' '))
  const nameAttribute = escapeHTML(outputName(node))
  return `<pre class="${classAttribute}" data-output-name="${nameAttribute}"><samp>${escapedPreText(text.join(''))}</samp></pre>`
}

function renderOutputNodeHtml(node: ElementContent): string {
  return serializedPreSamp(node) ?? toHtml(node, { allowDangerousHtml: true })
}

function rawBlock(classes: string[], rawHtml: string): ElementContent {
  const parsed = fromHtml(rawHtml, { fragment: true }).children as ElementContent[]
  return h('div', { className: classes }, [
    { type: 'text', value: '\n' },
    ...parsed,
    { type: 'text', value: '\n' },
  ])
}

function renderStream(output: StreamOutput): ElementContent[] {
  if (!output.text.trim()) return []
  return [
    preSamp(
      [
        'notebook-output',
        'notebook-output-stream',
        `notebook-output-stream-${classToken(output.name)}`,
      ],
      output.name,
      output.text,
    ),
  ]
}

function renderError(output: ErrorOutput): ElementContent[] {
  const traceback = stripAnsi(output.traceback.join('\n'))
  const header = [output.ename, output.evalue].filter(Boolean).join(': ')
  const text = traceback || header
  if (!text.trim()) return []
  return [preSamp(['notebook-output', 'notebook-output-error'], 'exit 1', text)]
}

function latexSource(value: string): { source: string; displayMode: boolean } {
  const trimmed = value.trim()
  const blockMath = trimmed.match(/^\$\$([\s\S]*)\$\$$/)
  if (blockMath) return { source: blockMath[1].trim(), displayMode: true }
  const bracketMath = trimmed.match(/^\\\[([\s\S]*)\\\]$/)
  if (bracketMath) return { source: bracketMath[1].trim(), displayMode: true }
  const inlineMath = trimmed.match(/^\$([\s\S]*)\$$/)
  if (inlineMath) return { source: inlineMath[1].trim(), displayMode: false }
  return { source: trimmed, displayMode: true }
}

function renderLatex(value: string): ElementContent | undefined {
  if (!value.trim()) return undefined
  const latex = latexSource(value)
  try {
    const rendered = katex.renderToString(latex.source, {
      output: 'htmlAndMathml',
      macros: customMacros,
      ...katexOptions,
      displayMode: latex.displayMode,
    })
    const innerNodes = fromHtml(rendered, { fragment: true }).children as ElementContent[]
    return h(
      'div',
      { className: ['notebook-output', 'notebook-output-latex'], 'data-output-name': 'result' },
      innerNodes,
    )
  } catch {
    return preSamp(['notebook-output', 'notebook-output-text'], 'result', value)
  }
}

function renderMimeBundle(data: MimeBundle): ElementContent[] {
  const html = mimeText(data['text/html'])
  if (html.trim()) {
    return [rawBlock(['notebook-output', 'notebook-output-html'], html)]
  }

  const markdown = mimeText(data['text/markdown'])
  if (markdown.trim()) {
    return [{ type: 'raw', value: markdown } as unknown as ElementContent]
  }

  const latex = mimeText(data['text/latex'])
  const renderedLatex = renderLatex(latex)
  if (renderedLatex) return [renderedLatex]

  const svg = mimeText(data['image/svg+xml'])
  if (svg.trim()) {
    return [rawBlock(['notebook-output', 'notebook-output-svg'], svg)]
  }

  for (const mime of NOTEBOOK_IMAGE_BINARY_MIME_TYPES as readonly NotebookImageBinaryMimeType[]) {
    const image = mimeText(data[mime])
    if (image.trim()) {
      return [
        h('p', { className: ['notebook-output', 'notebook-output-image'] }, [
          h('img', {
            src: `data:${mime};base64,${image.replace(/\s/g, '')}`,
            alt: 'notebook output',
          }),
        ]),
      ]
    }
  }

  const jsonValue = data['application/json']
  const json =
    typeof jsonValue === 'string' ? jsonValue : Array.isArray(jsonValue) ? jsonValue.join('') : ''
  if (json.trim()) {
    return [
      preSamp(['notebook-output', 'notebook-output-text', 'notebook-output-json'], 'result', json),
    ]
  }

  const text = mimeText(data['text/plain'])
  if (!text.trim() || ipythonDisplayPlaceholder.test(text.trim())) return []
  return [preSamp(['notebook-output', 'notebook-output-text'], 'result', text)]
}

export function renderOutput(output: Output): ElementContent[] {
  if (output.kind === 'stream') return renderStream(output)
  if (output.kind === 'error') return renderError(output)
  return renderMimeBundle(output.data)
}

export function renderSuccessMarker(): ElementContent {
  return h('div', {
    className: ['notebook-output', 'notebook-output-success'],
    'data-output-name': 'exit 0',
  })
}

export function renderOutputHtml(output: Output): string {
  return renderOutput(output).map(renderOutputNodeHtml).join('')
}

export function renderOutputHtmlBlocks(output: Output): NotebookOutputHtml[] {
  return renderOutput(output).map(node => ({
    label: outputName(node),
    html: renderOutputNodeHtml(node),
  }))
}

export function renderSuccessMarkerHtml(): string {
  return toHtml(renderSuccessMarker())
}

export function renderSuccessMarkerHtmlBlock(): NotebookOutputHtml {
  const node = renderSuccessMarker()
  return { label: outputName(node), html: toHtml(node) }
}

export const NOTEBOOK_OUTPUT_MIME_LIST: readonly string[] = NOTEBOOK_OUTPUT_MIME_PRIORITY
