import { Root } from 'hast'
import type { TranscludeOptions } from '../plugins/transformers/frontmatter'
import { QuartzComponentProps } from '../types/component'
import { FullSlug } from '../util/path'
import { renderPageWithMetadata } from './renderPage/render'
import { resolveTransclusions } from './renderPage/transclude'
export { pageResources } from './renderPage/page-resources'
export { CuriusContent, CuriusFriends, CuriusNavigation } from './renderPage/curius'
export { renderPageWithMetadata } from './renderPage/render'

export function transcludeFinal(
  root: Root,
  componentData: QuartzComponentProps,
  state: { visited: Set<FullSlug> },
  userOpts?: Partial<TranscludeOptions>,
): Root {
  return resolveTransclusions(root, componentData, state, userOpts).tree
}

export function renderPage(...args: Parameters<typeof renderPageWithMetadata>): string {
  return renderPageWithMetadata(...args).html
}
