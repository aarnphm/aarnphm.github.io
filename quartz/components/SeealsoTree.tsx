import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/seealsoTree.scss"
import { classNames } from "../util/lang"
import { FullSlug, resolveRelative } from "../util/path"
import type { FrontmatterLink } from "../plugins/transformers/frontmatter"
import { JSX } from "preact"

const MAX_DEPTH = 5
const MAX_CHILDREN_PER_NODE = 5

function getDisplayTitle(
  slug: FullSlug,
  file: QuartzComponentProps["fileData"] | undefined,
  alias?: string,
): string {
  if (alias && alias.trim().length > 0) {
    return alias.trim()
  }

  const frontmatterTitle = file?.frontmatter?.title
  if (typeof frontmatterTitle === "string" && frontmatterTitle.length > 0) {
    return frontmatterTitle
  }

  const fragment = slug.split("/").pop() || slug
  return fragment.replace(/\.[^/.]+$/, "").replace(/-/g, " ")
}

export default (() => {
  const SeealsoTree: QuartzComponent = ({
    fileData,
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    const fmLinks = fileData.frontmatterLinks as Record<string, FrontmatterLink[]> | undefined

    const rootLinks = fmLinks?.["seealso"]

    if (!rootLinks || rootLinks.length === 0) {
      return null
    }

    const slugToFile = new Map<FullSlug, QuartzComponentProps["fileData"]>()
    for (const data of allFiles) {
      const slug = data.slug as FullSlug | undefined
      if (slug) {
        slugToFile.set(slug, data)
      }
    }

    const seealsoBySlug = new Map<FullSlug, FrontmatterLink[]>()
    for (const data of allFiles) {
      const slug = data.slug as FullSlug | undefined
      if (!slug) continue
      const links = (data.frontmatterLinks as Record<string, FrontmatterLink[]> | undefined)?.[
        "seealso"
      ]
      if (links && links.length > 0) {
        seealsoBySlug.set(slug, links)
      }
    }

    const currentSlug = fileData.slug as FullSlug | undefined
    if (!currentSlug) {
      return null
    }

    const visited = new Set<FullSlug>([currentSlug])
    const lines: JSX.Element[] = []
    const nbsp = "\u00a0"
    const padAfterLabel = nbsp.repeat(2)
    const segmentPad = nbsp.repeat(3)
    const segmentWithBar = `│${segmentPad}`
    const segmentEmpty = `${nbsp}${segmentPad}`

    const formatReadingLabel = (minutes?: number): string => {
      let value = 0
      if (typeof minutes === "number" && Number.isFinite(minutes) && minutes > 0) {
        value = Math.ceil(minutes)
      }
      if (value < 10) {
        // e.g. "[ 0m]" to keep width aligned with "[12m]"
        return `[${nbsp}${value}m]`
      }
      return `[${value}m]`
    }

    const addBranch = (
      link: FrontmatterLink,
      depth: number,
      isLast: boolean,
      ancestorHasSibling: boolean[],
    ): void => {
      const targetSlug = link.slug
      if (visited.has(targetSlug)) {
        return
      }
      visited.add(targetSlug)

      const targetFile = slugToFile.get(targetSlug)
      const title = getDisplayTitle(targetSlug, targetFile, link.alias)
      const href = resolveRelative(currentSlug, targetSlug)

      const minutes = targetFile?.readingTime?.minutes

      const rawChildren = depth < MAX_DEPTH ? (seealsoBySlug.get(targetSlug) ?? []) : []
      const children = rawChildren.slice(0, MAX_CHILDREN_PER_NODE)

      const segments: string[] = []
      for (const hasSibling of ancestorHasSibling) {
        segments.push(hasSibling ? segmentWithBar : segmentEmpty)
      }
      const branchGlyph = isLast ? "└── " : "├── "
      const prefix = segments.join("") + branchGlyph

      const nextAncestors = [...ancestorHasSibling, !isLast]

      const labelText = formatReadingLabel(minutes)

      lines.push(
        <>
          {prefix}
          {labelText}
          {padAfterLabel}
          <a class="internal" href={href} data-slug={targetSlug}>
            {title}
          </a>
          <br />
        </>,
      )

      if (children && children.length > 0) {
        children.forEach((child, idx) =>
          addBranch(child, depth + 1, idx === children.length - 1, nextAncestors),
        )
      }
    }

    const topLevel = rootLinks.slice(0, MAX_CHILDREN_PER_NODE)
    topLevel.forEach((link, idx) => addBranch(link, 0, idx === topLevel.length - 1, []))

    return (
      <section class={classNames(displayClass, "seealso-tree", "main-col")}>
        <p class="seealso-tree-lines">{lines}</p>
      </section>
    )
  }

  SeealsoTree.css = style

  return SeealsoTree
}) satisfies QuartzComponentConstructor
