import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import breadcrumbsStyle from "./styles/breadcrumbs.scss"
import { FullSlug, SimpleSlug, joinSegments, resolveRelative } from "../util/path"
import { QuartzPluginData } from "../plugins/vfile"
import { classNames } from "../util/lang"

type CrumbData = {
  displayName: string
  path: string
}

type CrumbStyle = "full" | "letter" | "unique"

interface BreadcrumbOptions {
  /**
   * Maximum length for each breadcrumb title. Set to 0 or negative to disable truncation
   */
  titleLength: number
  /**
   * Symbol between crumbs
   */
  spacerSymbol: string
  /**
   * Name of first crumb
   */
  rootName: string
  /**
   * Maximum number of breadcrumbs to show before collapsing the middle ones
   * Set to 0 or negative to show all breadcrumbs
   */
  maxItems: number
  /**
   * Whether to look up frontmatter title for folders (could cause performance problems with big vaults)
   */
  resolveFrontmatterTitle: boolean
  /**
   * Whether to display breadcrumbs on root `index.md`
   */
  hideOnRoot: boolean
  /**
   * Whether to display the current page in the breadcrumbs.
   */
  showCurrentPage: boolean
  /**
   * Set a style for breadcrumbs. The following are supported:
   * - full (default): show the full path of the breadcrumb.
   * - letter: works like full, but will write every folder name using first letter only. The last folder will be displayed in full. For example:
   *   - `folder` will be shorten to `f`
   *   - `.config` will be shorten to `.c`
   * - unique: works like `letter`, but will make sure every folder name is shortest unique value. For example:
   *   - `path/path/path/to/file.md` with `unique` will set `p/pa/pat/path/to/file.md`.
   *   - However, uniqueness does not refer different folder at the same level. For example: `path1/file.md` and `path2/file.md` will both show `p/file.md`
   */
  style: CrumbStyle
}

const defaultOptions: BreadcrumbOptions = {
  titleLength: 20,
  spacerSymbol: "❯",
  rootName: "Home",
  resolveFrontmatterTitle: true,
  hideOnRoot: false,
  showCurrentPage: true,
  style: "full",
  maxItems: 0,
}

function formatCrumb(
  displayName: string,
  baseSlug: FullSlug,
  currentSlug: SimpleSlug,
  maxLength?: number,
): CrumbData {
  let title = displayName.replaceAll("-", " ")
  if (maxLength && maxLength > 0 && title.length > maxLength) {
    title = title.slice(0, maxLength) + "..."
  }
  return {
    displayName: title,
    path: resolveRelative(baseSlug, currentSlug),
  }
}

export default ((userOpts?: Partial<BreadcrumbOptions>) => {
  // Merge options with defaults
  const opts: BreadcrumbOptions = { ...defaultOptions, ...userOpts }

  // computed index of folder name to its associated file data
  let folderIndex: Map<string, QuartzPluginData> | undefined

  const Breadcrumbs: QuartzComponent = ({
    fileData,
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    if (opts.hideOnRoot && fileData.slug === "index") return <></>

    // Format entry for root element
    const firstEntry = formatCrumb(opts.rootName, fileData.slug!, "/" as SimpleSlug)
    const crumbs: CrumbData[] = [firstEntry]

    if (!folderIndex && opts.resolveFrontmatterTitle) {
      folderIndex = new Map()
      // construct the index for the first time
      for (const file of allFiles) {
        const folderParts = file.slug?.split("/")
        if (folderParts?.at(-1) === "index") {
          folderIndex.set(folderParts.slice(0, -1).join("/"), file)
        }
      }
    }

    // Split slug into hierarchy/parts
    const slugParts = fileData.slug?.split("/")
    if (slugParts) {
      // is tag breadcrumb?
      const isTagPath = slugParts[0] === "tags"

      // full path until current part
      let currentPath = ""

      // Map to store the shortened names for each path segment
      const shortenedNames: Map<string, string> = new Map()

      for (let i = 0; i < slugParts.length - 1; i++) {
        let curPathSegment = slugParts[i]

        // Try to resolve frontmatter folder title
        const currentFile = folderIndex?.get(slugParts.slice(0, i + 1).join("/"))
        if (currentFile) {
          const title = currentFile.frontmatter!.title
          if (title !== "index") {
            curPathSegment = title
          }
        }

        // Add current slug to full path
        currentPath = joinSegments(currentPath, slugParts[i])
        const includeTrailingSlash = !isTagPath || i < 1

        switch (opts.style) {
          case "letter":
            if (curPathSegment.startsWith(".")) {
              curPathSegment = curPathSegment.slice(0, 2)
            } else {
              curPathSegment = curPathSegment.charAt(0)
            }
            break
          case "unique":
            let shortenedName = curPathSegment.charAt(0)
            let uniqueName = shortenedName
            let counter = 1

            while (shortenedNames.has(uniqueName)) {
              uniqueName = curPathSegment.slice(0, counter + 1)
              counter++
            }

            shortenedNames.set(uniqueName, currentPath)
            curPathSegment = uniqueName
            break
        }

        // Format and add current crumb
        const crumb = formatCrumb(
          curPathSegment,
          fileData.slug!,
          (currentPath + (includeTrailingSlash ? "/" : "")) as SimpleSlug,
        )
        crumbs.push(crumb)
      }

      // Add current file to crumb (can directly use frontmatter title)
      if (opts.showCurrentPage && slugParts.at(-1) !== "index") {
        const formatted = formatCrumb(
          isTagPath ? (slugParts.at(-1) ?? "") : fileData.frontmatter!.title,
          fileData.slug!,
          "" as SimpleSlug,
        )
        crumbs.push({
          displayName: formatted.displayName,
          path: "",
        })
      }
    }

    let displayCrumbs = [...crumbs]
    if (opts.maxItems > 0 && crumbs.length > opts.maxItems) {
      const first = displayCrumbs[0]
      const last = displayCrumbs.slice(
        displayCrumbs.length - opts.maxItems! + 1,
        displayCrumbs.length,
      )
      displayCrumbs = [first, { displayName: "...", path: crumbs.at(-2)!.path }, ...last]
    }

    return (
      <nav class={classNames(displayClass, "breadcrumb-container")} aria-label="breadcrumbs">
        {displayCrumbs.map((crumb, index) => (
          <div class="breadcrumb-element">
            <a href={crumb.path} data-breadcrumbs>
              {crumb.displayName}
            </a>
            {index !== displayCrumbs.length - 1 && <p>{` ${opts.spacerSymbol} `}</p>}
          </div>
        ))}
      </nav>
    )
  }
  Breadcrumbs.css = breadcrumbsStyle

  return Breadcrumbs
}) satisfies QuartzComponentConstructor
