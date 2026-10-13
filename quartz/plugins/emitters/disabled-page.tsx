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
import { isFullSlug, pathToRoot } from '../../util/path'
import { defaultProcessedContent } from '../vfile'
import { removeWritten, write } from './helpers'

type Options = { label: string; source: string; routes: readonly string[]; setting?: string }

export function DisabledPage(options: Options): QuartzEmitterPluginInstance {
  const routes = [...new Set(options.routes)].map(route => {
    if (!isFullSlug(route)) throw new Error(`Invalid disabled-page route: ${route}`)
    return route
  })
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
      if (!content.some(([, file]) => file.data.slug === options.source)) {
        for (const slug of routes) await removeWritten(ctx, slug, '.html')
        return
      }
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
              allFiles: contentDataFor(content),
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
