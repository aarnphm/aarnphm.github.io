import type { Element, Root as HtmlRoot } from 'hast'
import type { Root } from 'mdast'
import { toString } from 'mdast-util-to-string'
import { visit } from 'unist-util-visit'
import type { QuartzTransformerPlugin } from '../../types/plugin'
// @ts-ignore Inline browser modules are bundled into script strings.
import script from '../../components/scripts/speech.inline'
import style from '../../components/styles/speech.inline.scss'
import { remarkSpeech } from '../../extensions/micromark-extension-speech'

interface Options {
  modelBaseUrl?: string
}

function hasClass(node: Element, name: string): boolean {
  const classes = node.properties.className
  return Array.isArray(classes) ? classes.includes(name) : classes === name
}

function hasBlank(node: Element): boolean {
  return (
    hasClass(node, 'cloze-blank') ||
    node.children.some(child => child.type === 'element' && hasBlank(child))
  )
}

const interactiveTags = new Set(['a', 'button', 'input', 'select', 'textarea', 'summary'])

function hasInteractiveContent(node: Element): boolean {
  return node.children.some(
    child =>
      child.type === 'element' &&
      (interactiveTags.has(child.tagName) || hasInteractiveContent(child)),
  )
}

function playButton(text: string, children: Element['children']): Element {
  return {
    type: 'element',
    tagName: 'button',
    properties: {
      type: 'button',
      disabled: true,
      className: ['speech-play'],
      ariaLabel: `Écouter : ${text}`,
      title: 'Écouter',
    },
    children,
  }
}

function addControls(tree: HtmlRoot): void {
  function walk(node: Element, interactiveAncestor: boolean): void {
    const isPhrase = hasClass(node, 'speech-phrase')
    const text = node.properties.dataSpeechText
    if (isPhrase && typeof text === 'string') {
      if (
        interactiveAncestor ||
        hasBlank(node) ||
        hasInteractiveContent(node) ||
        text.length === 0
      ) {
        delete node.properties.dataSpeechText
      } else {
        node.children = [playButton(text, node.children)]
      }
    }
    const interactive = interactiveAncestor || interactiveTags.has(node.tagName)
    for (const child of node.children) {
      if (child.type === 'element') walk(child, interactive)
    }
  }
  for (const child of tree.children) {
    if (child.type === 'element') walk(child, false)
  }
}

export const Speech: QuartzTransformerPlugin<Options> = (options = {}) => ({
  name: 'Speech',
  markdownPlugins: () => [
    remarkSpeech,
    () => (tree: Root) => {
      visit(tree, 'speechPhrase', node => {
        const text = toString(node, { includeHtml: false }).trim().replace(/\s+/g, ' ')
        node.data = {
          ...node.data,
          hName: 'span',
          hProperties: {
            className: ['speech-phrase'],
            dataSpeechText: text,
            lang: 'fr',
            ...(options.modelBaseUrl ? { dataSpeechModelBaseUrl: options.modelBaseUrl } : {}),
          },
        }
      })
    },
  ],
  htmlPlugins: () => [() => (tree: HtmlRoot) => addControls(tree)],
  externalResources: () => ({
    css: [{ content: style, inline: true, spaPreserve: true }],
    js: [{ script, loadTime: 'afterDOMReady', contentType: 'inline', spaPreserve: true }],
  }),
})
