import { Root } from 'hast'
import { h } from 'hastscript'
import path from 'path'
import type { JsonCanvas } from '../transformers/jcast/types'
import { defaultContentPageLayout, sharedPageComponents } from '../../../quartz.layout'
import { FullPageLayout } from '../../cfg'
import { Content } from '../../components'
import CanvasComponent from '../../components/Canvas'
import { pageResources, renderPage } from '../../components/renderPage'
import { QuartzComponentProps } from '../../types/component'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { contentDataFor } from '../../util/ctx'
import { buildCanvasPreview, documentPreviewSlug } from '../../util/document-preview'
import { pathToRoot, simplifySlug, SimpleSlug, slugifyFilePath } from '../../util/path'
import { collectCanvasMeta } from '../transformers/canvas'
import { QuartzPluginData } from '../vfile'
import { write } from './helpers'

export const CanvasPage: QuartzEmitterPlugin<Partial<FullPageLayout>> = userOpts => {
  const header = sharedPageComponents.header.filter(component => {
    const name = component.displayName || component.name || ''
    return name !== 'Breadcrumbs' && name !== 'StackedNotes'
  })

  const opts: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    ...userOpts,
    pageBody: Content(),
    header,
    beforeBody: [],
    afterBody: [CanvasComponent()],
  }

  const {
    head: Head,
    header: Header,
    beforeBody: BeforeBody,
    pageBody,
    afterBody,
    sidebar,
    footer: Footer,
  } = opts

  return {
    name: 'CanvasPage',
    getQuartzComponents() {
      return [Head, ...Header, ...BeforeBody, pageBody, ...afterBody, ...sidebar, Footer]
    },
    async *emit(ctx, content, resources) {
      const { cfg } = ctx

      const allFiles = contentDataFor(content)

      for (const [_tree, file] of content) {
        // Only process files marked as canvas files
        if (!file.data.jsonCanvas || !file.data.canvas || !file.data.canvasContent) continue

        const slug = slugifyFilePath(file.data.relativePath!, true)
        const jcast = file.data.canvas

        const jscastMetadata = collectCanvasMeta(jcast)

        yield write({ ctx, content: JSON.stringify(jscastMetadata), slug, ext: '.meta.json' })

        const displayNames = new Map<string, string>()
        for (const [id, node] of jcast.data.nodeMap) {
          const displayName = node.data?.resolved?.displayName
          if (displayName) displayNames.set(id, displayName)
        }
        const preview = buildCanvasPreview({
          slug,
          title: path.basename(file.data.filePath!, '.canvas'),
          canvas: JSON.parse(file.data.canvasContent) as JsonCanvas,
          displayNames,
        })
        yield write({
          ctx,
          content: JSON.stringify(preview),
          slug: documentPreviewSlug(slug),
          ext: '.json',
        })

        // default canvas configuration
        const defaultConfig = {
          drag: true,
          zoom: true,
          forceStrength: 0.3,
          linkDistance: 150,
          collisionRadius: 50,
          useManualPositions: true,
          showInlineContent: false,
          showPreviewOnHover: true,
          previewMaxLength: 300,
        }

        const resourceBase = slug.startsWith('/') ? slug : `/${slug}`

        const canvasElement = h(
          'div.canvas-container',
          {
            'data-canvas': `${resourceBase}.canvas`,
            'data-meta': `${resourceBase}.meta.json`,
            'data-cfg': JSON.stringify(defaultConfig),
            'data-canvas-bounds': JSON.stringify(jcast.data.bounds),
            'data-canvas-title': path.basename(file.data.filePath!, '.canvas'),
            style: `position: relative;`,
          },
          [h('div.canvas-loading', 'Loading canvas...')],
        )

        const tree: Root = { type: 'root', children: [canvasElement] }

        //@ts-ignore
        const linkedSlugs = (file.data.links ?? []).map(simplifySlug) as SimpleSlug[]

        const fileData: QuartzPluginData = {
          ...file.data,
          slug,
          frontmatter: { title: `${slug}.canvas`, tags: ['canvas'], pageLayout: 'default' },
          description: `Canvas of ${slug}`,
          htmlAst: tree,
          text: file.data.text ?? '',
          links: linkedSlugs,
        }

        const externalResources = pageResources(pathToRoot(slug), resources, ctx)
        const componentData: QuartzComponentProps = {
          ctx,
          fileData,
          externalResources,
          cfg: cfg.configuration,
          children: [],
          tree,
          allFiles,
        }

        const content = renderPage(ctx, slug, componentData, opts, externalResources, false)

        // write HTML page to output
        yield write({ ctx, content, slug, ext: '.html' })
      }
    },
  }
}
