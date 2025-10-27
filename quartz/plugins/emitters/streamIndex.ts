import { QuartzEmitterPlugin } from "../types"
import { write } from "./helpers"
import { groupStreamEntries } from "../../util/stream"
import { renderStreamEntry, formatStreamDate, buildOnPath } from "../../components/stream/Entry"
import StreamPageComponent from "../../components/pages/StreamPage"
import { pageResources, renderPage } from "../../components/renderPage"
import { sharedPageComponents, defaultContentPageLayout } from "../../../quartz.layout"
import type { QuartzComponentProps } from "../../components/types"
import type { FullPageLayout } from "../../cfg"
import { pathToRoot, FullSlug } from "../../util/path"
import { render } from "preact-render-to-string"
import type { QuartzPluginData } from "../vfile"
import type { StaticResources } from "../../util/resources"
import type { Node } from "unist"

export const StreamIndex: QuartzEmitterPlugin = () => {
  const formatIsoAsYMD = (iso?: string | null): string | null => {
    if (!iso) return null
    const date = new Date(iso)
    if (Number.isNaN(date.getTime())) return null
    const year = date.getUTCFullYear()
    const month = String(date.getUTCMonth() + 1).padStart(2, "0")
    const day = String(date.getUTCDate()).padStart(2, "0")
    return `${year}/${month}/${day}`
  }

  return {
    name: "StreamIndex",
    async *emit(ctx, content, resources: StaticResources) {
      const allFiles = content.map(([, file]) => file.data as QuartzPluginData)

      let streamFile: QuartzPluginData | undefined
      let streamTree: Node | undefined

      for (const [tree, file] of content) {
        const data = file.data as QuartzPluginData
        if (data.slug === "stream" && data.streamData) {
          streamFile = data
          streamTree = tree
          break
        }
      }

      if (!streamFile?.streamData || !streamFile.filePath || !streamTree) {
        return
      }

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

      const groups = groupStreamEntries(streamFile.streamData.entries)
      if (groups.length === 0) {
        return
      }

      const lines = groups.map((group) => {
        const isoSource =
          group.isoDate ??
          group.entries.find((entry) => entry.date)?.date ??
          (group.timestamp ? new Date(group.timestamp).toISOString() : null)

        const path = buildOnPath(isoSource) ?? null
        const entries = group.entries.map((entry) => {
          const vnode = renderStreamEntry(entry, streamFile!.filePath!, {
            groupId: group.id,
            timestampValue: group.timestamp,
            showDate: true,
            resolvedIsoDate: entry.date ?? group.isoDate,
          })

          return {
            id: entry.id,
            html: render(vnode),
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
        slug: "stream-timestamps" as FullSlug,
        ext: ".jsonl",
        content: payload,
      })

      for (const group of groups) {
        const isoSource =
          group.isoDate ??
          group.entries.find((entry) => entry.date)?.date ??
          (group.timestamp ? new Date(group.timestamp).toISOString() : null)

        const onPath = buildOnPath(isoSource)
        if (!onPath) {
          continue
        }

        const slug = onPath.replace(/^\//, "") as FullSlug
        const titleDate = formatIsoAsYMD(isoSource) ?? formatIsoAsYMD(group.isoDate)
        const title = titleDate ?? streamFile.frontmatter?.title ?? "stream"

        const existingTags = Array.isArray(streamFile.frontmatter?.tags)
          ? (streamFile.frontmatter!.tags as unknown[]).map((value) => String(value))
          : []
        if (!existingTags.includes("on")) {
          existingTags.push("on")
        }

        const fileDataForGroup: QuartzPluginData = {
          ...streamFile,
          slug,
          streamData: {
            entries: [...group.entries],
          },
          frontmatter: {
            ...streamFile.frontmatter,
            title,
            streamCanonical: "/stream",
            pageLayout: "default",
            tags: existingTags,
          },
        }

        const externalResources = pageResources(pathToRoot(slug), resources, ctx)
        const componentData: QuartzComponentProps = {
          ctx,
          fileData: fileDataForGroup,
          externalResources,
          cfg: ctx.cfg.configuration,
          children: [],
          tree: streamTree,
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
    },
  }
}
