import type { ComponentChild } from 'preact'
import { fromHtmlIsomorphic } from 'hast-util-from-html-isomorphic'
import { visit } from 'unist-util-visit'
import type { QuartzComponentProps } from '../../types/component'
import { fromHtmlStringToArenaJsx } from '../../util/arena'
import { renderInlineReferenceHtml } from '../../util/description'
import { referenceCatalogFor, referenceSourcePath } from '../../util/references/context'

export function arenaReferenceText(
  text: string,
  props: QuartzComponentProps,
  options: { links?: boolean } = {},
): ComponentChild {
  const slug = props.fileData.slug
  if (!text || !slug) return text
  const html = renderInlineReferenceHtml(text, slug, {
    catalog: referenceCatalogFor(props.ctx),
    origin: referenceSourcePath(props.fileData),
  })
  const tree = fromHtmlIsomorphic(html, { fragment: true })
  visit(tree, 'element', node => {
    if (node.tagName !== 'a') return
    if (options.links === false) {
      node.tagName = 'span'
      delete node.properties.href
      delete node.properties.target
      return
    }
    node.properties.dataNoPopover = true
    const href = node.properties.href
    if (typeof href === 'string' && href.startsWith('/') && !href.startsWith('//'))
      node.properties.dataSlug = href.slice(1).split('#')[0]
  })
  return fromHtmlStringToArenaJsx(props.fileData.filePath ?? '', tree, slug, props)
}
