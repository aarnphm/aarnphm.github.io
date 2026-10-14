import type { FullPageLayout } from '../../cfg'
import type { QuartzComponent } from '../../types/component'
import type { QuartzEmitterPluginInstance } from '../../types/plugin'
import { sharedPageComponents } from '../../../quartz.layout'
import {
  DisabledPluginNotice,
  disabledPluginNoticeStyle,
} from '../../components/DisabledPluginNotice'
import HeaderConstructor from '../../components/Header'
import { pageResources, renderPage } from '../../components/renderPage'
import { i18n } from '../../i18n'
import { contentDataFor } from '../../util/ctx'
import { isFullSlug, pathToRoot, type FullSlug } from '../../util/path'
import { defaultProcessedContent, type QuartzPluginData } from '../vfile'
import { removeWritten, write } from './helpers'

type Options = {
  label: string
  source: string | ((file: QuartzPluginData) => boolean)
  routes: readonly string[] | ((file: QuartzPluginData) => readonly string[])
  setting?: string
}

export function DisabledPage(options: Options): QuartzEmitterPluginInstance {
  const routeSlug = (route: string): FullSlug => {
    if (!isFullSlug(route)) throw new Error(`Invalid disabled-page route: ${route}`)
    return route
  }
  const fixedRoutes = typeof options.routes === 'function' ? [] : options.routes.map(routeSlug)
  const writtenRoutes = new Map<string, Set<FullSlug>>()
  const Body: QuartzComponent = ({ cfg, fileData }) => (
    <article class="popover-hint disabled-plugin-page">
      <DisabledPluginNotice
        label={options.label}
        setting={options.setting}
        locale={cfg.locale}
        homeHref={pathToRoot(fileData.slug!)}
      />
    </article>
  )
  Body.css = disabledPluginNoticeStyle
  const layout: FullPageLayout = {
    ...sharedPageComponents,
    pageBody: Body,
    beforeBody: [],
    sidebar: [],
    afterBody: [],
  }
  const Header = HeaderConstructor()

  return {
    name: 'DisabledPage',
    getQuartzComponents() {
      return [layout.head, Header, ...layout.header, Body, layout.footer]
    },
    async *emit(ctx, content, resources) {
      const sourceSlugs = new Set(content.map(([, file]) => file.data.slug))
      const routes = new Set<FullSlug>()
      for (const [, file] of content) {
        const matches =
          typeof options.source === 'function'
            ? options.source(file.data)
            : file.data.slug === options.source
        if (!matches) continue
        const selected =
          typeof options.routes === 'function'
            ? options.routes(file.data).map(routeSlug)
            : fixedRoutes
        for (const slug of selected) {
          if (!sourceSlugs.has(slug)) routes.add(slug)
        }
      }
      const previous = writtenRoutes.get(ctx.argv.output) ?? new Set(fixedRoutes)
      for (const slug of previous) {
        if (!routes.has(slug) && !sourceSlugs.has(slug)) await removeWritten(ctx, slug, '.html')
      }
      writtenRoutes.set(ctx.argv.output, routes)
      const allFiles = contentDataFor(content)
      for (const slug of routes) {
        const message = i18n(ctx.cfg.configuration.locale).components.disabledPlugin.message
        const [tree, file] = defaultProcessedContent({
          slug,
          text: message,
          description: message,
          frontmatter: { title: options.label, tags: [], pageLayout: 'default' },
        })
        const externalResources = pageResources(pathToRoot(slug), resources, ctx)
        yield write({
          ctx,
          slug,
          ext: '.html',
          content: renderPage(
            ctx,
            slug,
            {
              ctx,
              cfg: ctx.cfg.configuration,
              fileData: file.data,
              children: [],
              tree,
              allFiles,
              externalResources,
            },
            layout,
            externalResources,
          ),
        })
      }
    },
  }
}
