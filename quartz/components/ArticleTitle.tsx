import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
import { resolveRelative, FullSlug } from "../util/path"
import { createWikilinkRegex, parseWikilink, resolveWikilinkTarget } from "../util/wikilinks"
import { ArenaData, ArenaChannel } from "../plugins/transformers/arena"
import { toArenaHeadingJsx } from "../util/arena"
import type { JSX } from "preact"

export default (() => (componentData: QuartzComponentProps) => {
  const { fileData, displayClass } = componentData
  const title = fileData.frontmatter?.title
  const slug = fileData.slug!
  const isArenaIndex = slug === "arena"
  const isArenaChannel = slug.startsWith("arena/") && slug !== "arena"

  const renderDescription = (description: string | undefined) => {
    if (!description) {
      return []
    }

    const nodes: (string | JSX.Element)[] = []
    const regex = createWikilinkRegex()
    let lastIndex = 0
    let match: RegExpExecArray | null

    while ((match = regex.exec(description)) !== null) {
      const matchText = match[0]
      const start = match.index

      if (start > lastIndex) {
        nodes.push(description.slice(lastIndex, start))
      }

      const parsed = parseWikilink(matchText)
      const resolved = parsed ? resolveWikilinkTarget(parsed, slug as FullSlug) : null

      if (parsed && resolved) {
        const hrefBase = resolveRelative(slug as FullSlug, resolved.slug)
        const href = parsed.anchor ? `${hrefBase}${parsed.anchor}` : hrefBase
        nodes.push(
          <a
            href={href}
            class="internal"
            data-no-popover
            data-slug={resolved.slug}
            key={`${href}-${nodes.length}`}
          >
            {parsed.alias ?? parsed.target ?? matchText}
          </a>,
        )
      } else {
        nodes.push(parsed?.alias ?? parsed?.target ?? matchText)
      }

      lastIndex = regex.lastIndex
    }

    if (lastIndex < description.length) {
      nodes.push(description.slice(lastIndex))
    }

    return nodes.length > 0 ? nodes : [description]
  }

  if (isArenaIndex) {
    const arenaData = fileData.arenaData as ArenaData | undefined

    return (
      <hgroup class={classNames(displayClass, "title-col", "arena-title-block")} data-article-title>
        <h1 class="article-title">are.na</h1>
        <p class="description">
          {arenaData
            ? `${arenaData.channels.length} channels · ${arenaData.channels.reduce((sum, ch) => sum + ch.blocks.length, 0)} blocks`
            : ""}
        </p>
      </hgroup>
    )
  }

  if (isArenaChannel) {
    const channel = fileData.arenaChannel as ArenaChannel | undefined
    const arenaRootSlug = "arena" as FullSlug

    return (
      <hgroup class={classNames(displayClass, "title-col", "arena-title-block")} data-article-title>
        <h1 class="article-title">
          <a
            href={resolveRelative(slug, arenaRootSlug)}
            class="internal"
            data-no-popover
            data-slug={arenaRootSlug}
            style={{ background: "transparent" }}
          >
            are.na
          </a>
          {" / "}
          {channel?.titleHtmlNode
            ? toArenaHeadingJsx(
                fileData.filePath!,
                channel.titleHtmlNode,
                fileData.slug! as FullSlug,
                `arena/${channel.slug}` as FullSlug,
                componentData,
              )
            : channel?.name || title}
        </h1>
        <p class="description">{channel ? `${channel.blocks.length} blocks` : ""}</p>
      </hgroup>
    )
  }

  if (title) {
    return (
      <hgroup class={classNames(displayClass, "title-col")} data-article-title>
        <h1 class="article-title">{title}</h1>
        <p class="description">{renderDescription(fileData.description)}</p>
      </hgroup>
    )
  }

  return <></>
}) satisfies QuartzComponentConstructor
