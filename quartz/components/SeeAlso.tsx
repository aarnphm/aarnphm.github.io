import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
import style from "./styles/seealsoTree.scss"
import { classNames } from "../util/lang"
import { FullSlug, resolveRelative } from "../util/path"
import type { FrontmatterLink } from "../plugins/transformers/frontmatter"
import { JSX } from "preact"
import fs from "fs"
import path from "path"
import { Cite } from "@citation-js/core"
import "@citation-js/plugin-bibtex"

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

let loadedBib: Cite | null = null
function getCitationTitle(bibKey: string): string | undefined {
  if (!loadedBib) {
    const bibPath = path.join(process.cwd(), "content/References.bib")
    if (fs.existsSync(bibPath)) {
      const bibContent = fs.readFileSync(bibPath, "utf8")
      loadedBib = new Cite(bibContent, { generateGraph: false })
    }
  }

  if (loadedBib) {
    const entry = loadedBib.data.find((item: any) => item.id === bibKey)
    return entry?.title
  }
  return undefined
}

export default (() => {
  const SeeAlso: QuartzComponent = ({ fileData, allFiles, displayClass }: QuartzComponentProps) => {
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

    const visited = new Set<string>([currentSlug])
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
      const isCitation = targetSlug.startsWith("@")
      const uniqueId = isCitation ? targetSlug : targetSlug

      if (visited.has(uniqueId)) {
        return
      }
      visited.add(uniqueId)

      let title = link.alias || targetSlug
      let href = "#"
      let minutes: number | undefined
      let children: FrontmatterLink[] = []

      if (isCitation) {
        const bibKey = targetSlug.substring(1)
        const citeTitle = getCitationTitle(bibKey)
        if (citeTitle) {
          title = citeTitle
        }
        href = `#bib-${bibKey.toLowerCase()}`
      } else {
        const targetFile = slugToFile.get(targetSlug)
        title = getDisplayTitle(targetSlug, targetFile, link.alias)
        href = resolveRelative(currentSlug, targetSlug)
        minutes = targetFile?.readingTime?.minutes
        const rawChildren = depth < MAX_DEPTH ? (seealsoBySlug.get(targetSlug) ?? []) : []
        children = rawChildren.slice(0, MAX_CHILDREN_PER_NODE)
      }

      const segments: string[] = []
      for (const hasSibling of ancestorHasSibling) {
        segments.push(hasSibling ? segmentWithBar : segmentEmpty)
      }
      const branchGlyph = isLast ? "└── " : "├── "
      const prefix = segments.join("") + branchGlyph

      const nextAncestors = [...ancestorHasSibling, !isLast]

      const labelText = isCitation ? "[cite]" : formatReadingLabel(minutes)

      lines.push(
        <>
          {prefix}
          <span class="seealso-label">{labelText}</span>
          {padAfterLabel}
          <a
            class={isCitation ? "external" : "internal"}
            href={href}
            data-slug={isCitation ? undefined : targetSlug}
          >
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

  SeeAlso.css = style

  return SeeAlso
}) satisfies QuartzComponentConstructor
