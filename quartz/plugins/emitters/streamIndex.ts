import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"
import { groupStreamEntries } from "../../util/stream"
import { renderStreamEntry, formatStreamDate, buildOnPath } from "../../components/stream/Entry"
import StreamPageComponent from "../../components/pages/StreamPage"
import { pageResources, renderPage } from "../../components/renderPage"
import { sharedPageComponents, defaultContentPageLayout } from "../../../quartz.layout"
import type { QuartzComponentProps } from "../../components/types"
import type { FullPageLayout } from "../../cfg"
import { pathToRoot, FullSlug, normalizeHastElement } from "../../util/path"
import { render } from "preact-render-to-string"
import type { QuartzPluginData } from "../vfile"
import type { StaticResources } from "../../util/resources"
import type { Root, Element, ElementContent } from "hast"
import type { VNode } from "preact"
import { BuildCtx } from "../../util/ctx"

const formatIsoAsYMD = (iso?: string | null): string | null => {
  if (!iso) return null
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return null
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, "0")
  const day = String(date.getUTCDate()).padStart(2, "0")
  return `${year}/${month}/${day}`
}

const isElement = (node: ElementContent): node is Element => node.type === "element"

async function* processStreamIndex(
  ctx: BuildCtx,
  fileData: QuartzPluginData,
  tree: Root,
  allFiles: QuartzPluginData[],
  resources: StaticResources,
) {
  const filteredHeader = sharedPageComponents.header.filter((component) => {
    const name = component.displayName || component.name || ""
    return name !== "Breadcrumbs" && name !== "StackedNotes"
  })
  const filteredBefore = defaultContentPageLayout.beforeBody.filter(
    (c) => c.displayName !== "Byline" || c.name !== "Byline",
  )

  const layout: FullPageLayout = {
    ...sharedPageComponents,
    ...defaultContentPageLayout,
    header: filteredHeader,
    beforeBody: filteredBefore,
    afterBody: [],
    pageBody: StreamPageComponent(),
  }

  const groups = groupStreamEntries(fileData!.streamData!.entries)
  if (groups.length === 0) return

  const lines = groups.map((group) => {
    const isoSource =
      group.isoDate ??
      group.entries.find((entry) => entry.date)?.date ??
      (group.timestamp ? new Date(group.timestamp).toISOString() : null)

    const path = buildOnPath(isoSource!) ?? null
    const entries = group.entries.map((entry) => {
      const vnode = renderStreamEntry(entry, fileData!.filePath!, {
        groupId: group.id,
        timestampValue: group.timestamp,
        showDate: true,
        resolvedIsoDate: entry.date ?? group.isoDate,
      })

      return {
        id: entry.id,
        html: render(vnode as VNode<any>),
        metadata: entry.metadata,
        isoDate: entry.date ?? group.isoDate ?? null,
        displayDate:
          formatIsoAsYMD(entry.date ?? group.isoDate ?? isoSource) ??
          formatStreamDate(entry.date ?? group.isoDate) ??
          null,
      }
    })

    return JSON.stringify({
      groupId: group.id,
      timestamp: group.timestamp ?? null,
      isoDate: group.isoDate ?? null,
      groupSize: group.entries.length,
      path,
      entries,
    })
  })

  const payload = lines.join("\n")

  yield write({
    ctx,
    slug: "streams" as FullSlug,
    ext: ".jsonl",
    content: payload,
  })

  for (const group of groups) {
    const isoSource =
      group.isoDate ??
      group.entries.find((entry) => entry.date)?.date ??
      (group.timestamp ? new Date(group.timestamp).toISOString() : null)

    const onPath = buildOnPath(isoSource!)
    if (!onPath) continue

    const slug = onPath.replace(/^\//, "") as FullSlug
    const titleDate = formatIsoAsYMD(isoSource) ?? formatIsoAsYMD(group.isoDate)
    const title = titleDate ?? fileData!.frontmatter?.title ?? "stream"
    const sourceSlug = fileData.slug! as FullSlug
    const rebasedEntries = group.entries.map((entry) => ({
      ...entry,
      content: entry.content.map((node) =>
        isElement(node)
          ? (normalizeHastElement(node, slug, sourceSlug) as ElementContent)
          : node,
      ),
    }))

    const fileDataForGroup: QuartzPluginData = {
      ...fileData,
      slug,
      streamData: { entries: rebasedEntries },
      frontmatter: {
        ...fileData!.frontmatter,
        title,
        streamCanonical: "/stream",
        pageLayout: "default",
      },
    }

    const externalResources = pageResources(pathToRoot(slug), resources, ctx)
    const componentData: QuartzComponentProps = {
      ctx,
      fileData: fileDataForGroup,
      externalResources,
      cfg: ctx.cfg.configuration,
      children: [],
      tree,
      allFiles,
    }

    const html = renderPage(ctx, slug, componentData, layout, externalResources, false)

    yield write({
      ctx,
      slug,
      ext: ".html",
      content: html,
    })
  }
}

export const StreamIndex: QuartzEmitterPlugin = () => {
  return {
    name: "StreamIndex",
    async *emit(ctx, content, resources) {
      const allFiles = content.map(([, file]) => file.data as QuartzPluginData)

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        if (data.slug !== "stream" || !data.streamData) continue

        yield* processStreamIndex(ctx, data, tree, allFiles, resources)
      }
    },
    async *partialEmit(ctx, content, resources, changeEvents) {
      const allFiles = content.map(([, file]) => file.data as QuartzPluginData)
      const changedSlugs = new Set<string>()

      for (const changeEvent of changeEvents) {
        if (changeEvent.file) {
          if (changeEvent.type === "add" || changeEvent.type === "change") {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
          continue
        }

        if (changeEvent.type === "add" || changeEvent.type === "change") {
          const changedPath = changeEvent.path
          for (const [_, vf] of content) {
            const deps = (vf.data.codeDependencies as string[] | undefined) ?? []
            if (deps.includes(changedPath)) {
              changedSlugs.add(vf.data.slug!)
            }
          }
        }
      }

      if (!changedSlugs.has("stream")) return

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        const slug = data.slug!
        if (slug !== "stream" || !data.streamData) continue
        yield* processStreamIndex(ctx, data, tree, allFiles, resources)
      }
    },
  }
}
