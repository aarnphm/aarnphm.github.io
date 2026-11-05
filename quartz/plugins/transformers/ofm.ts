import { Element, Root as HtmlRoot } from 'hast'
// @ts-ignore
import calloutScript from '../../components/scripts/callout.inline.ts'
//@ts-ignore
import checkboxScript from '../../components/scripts/checkbox.inline'
// @ts-ignore
import embedScript from '../../components/scripts/embed.inline'
// @ts-ignore
import noteStackScript from '../../components/scripts/note-stack.inline'
// @ts-ignore
import peleScript from '../../components/scripts/pele.inline'
import peleStyle from '../../components/styles/pele.inline.scss'
import transcriptStyle from '../../components/styles/youtube-transcript.scss'
import noteStackStyle from '../../styles/note-stack.scss'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { CSSResource, JSResource } from '../../util/resources'
import { createHtmlPlugins, enableMarkdownTaskCheckboxes } from './ofm/html'
import { createMarkdownPlugins } from './ofm/markdown'
export { externalLinkRegex } from './ofm/html'
export interface Options {
  comments: boolean
  highlight: boolean
  wikilinks: boolean
  callouts: boolean
  mermaid: boolean
  parseTags: boolean
  parseArrows: boolean
  parseBlockReferences: boolean
  enableCheckbox: boolean
  enableInHtmlEmbed: boolean
  enableYouTubeEmbed: boolean
  enableInlineFootnotes: boolean
  enableImageGrid: boolean
  enableMarker: boolean
  enableCalloutIcon: boolean
}

const defaultOptions: Options = {
  comments: true,
  highlight: true,
  wikilinks: true,
  callouts: true,
  mermaid: true,
  parseTags: true,
  parseArrows: true,
  parseBlockReferences: true,
  enableCheckbox: true,
  enableInHtmlEmbed: false,
  enableYouTubeEmbed: true,
  enableInlineFootnotes: true,
  enableImageGrid: true,
  enableMarker: true,
  enableCalloutIcon: false,
}

export const ObsidianFlavoredMarkdown: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: 'ObsidianFlavoredMarkdown',
    cacheKey: JSON.stringify(opts),
    markdownPlugins() {
      return createMarkdownPlugins(opts)
    },
    htmlSourcePlugins() {
      return opts.enableCheckbox ? [() => enableMarkdownTaskCheckboxes] : []
    },
    htmlPlugins() {
      return createHtmlPlugins(opts)
    },
    externalResources() {
      const js: JSResource[] = []
      const css: CSSResource[] = []

      if (opts.wikilinks) {
        js.push({ script: noteStackScript, loadTime: 'afterDOMReady', contentType: 'inline' })
        css.push({ content: noteStackStyle, inline: true })
      }

      if (opts.enableCheckbox) {
        js.push({ script: checkboxScript, loadTime: 'afterDOMReady', contentType: 'inline' })
      }

      if (opts.enableYouTubeEmbed) {
        js.push({ script: embedScript, loadTime: 'afterDOMReady', contentType: 'inline' })
        css.push({ content: transcriptStyle, inline: true })
      }

      if (opts.callouts) {
        js.push({ script: calloutScript, loadTime: 'afterDOMReady', contentType: 'inline' })
      }
      if (opts.mermaid) {
        js.push({
          script: peleScript,
          loadTime: 'afterDOMReady',
          contentType: 'inline',
          moduleType: 'module',
        })
        css.push({ content: peleStyle, inline: true })
      }

      return { js, css }
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    images: Record<string, { count: number; el: Element }>
    blocks: Record<string, Element>
    htmlAst: HtmlRoot
  }
}
