import path from "path"
import fs from "node:fs/promises"
import crypto from "node:crypto"
import { Root } from "hast"
import { GlobalConfiguration } from "../../cfg"
import { formatDate, getDate } from "../../components/Date"
import { escapeHTML } from "../../util/escape"
import { FilePath, FullSlug, SimpleSlug, joinSegments, simplifySlug } from "../../util/path"
import { QuartzEmitterPlugin } from "../types"
import { toHtml } from "hast-util-to-html"
import { write } from "./helpers"
import { i18n } from "../../i18n"
import { QuartzPluginData } from "../vfile"
import { version } from "../../../package.json"
import { ReadTimeResults } from "reading-time"

export type ContentIndexMap = Map<FullSlug, ContentDetails>
export type ContentLayout = "default" | "letter" | "technical" | "reflection" | "letter-poem"
export type ContentDetails = {
  slug: string
  title: string
  filePath: FilePath
  links: SimpleSlug[]
  aliases: string[]
  tags: string[]
  layout: ContentLayout
  content: string
  fileName: FilePath
  richContent?: string
  fileData?: QuartzPluginData
  date?: Date
  readingTime?: Partial<ReadTimeResults>
  description?: string
}

interface Options {
  enableSiteMap: boolean
  enableRSS: boolean
  enableAtom: boolean
  rssLimit?: number
  rssSlug: string
  includeEmptyFiles: boolean
}

const defaultOptions: Options = {
  enableSiteMap: true,
  enableRSS: true,
  enableAtom: true,
  rssLimit: 10,
  rssSlug: "index",
  includeEmptyFiles: true,
}

function generateSiteMap(cfg: GlobalConfiguration, idx: ContentIndexMap): string {
  const base = cfg.baseUrl ?? ""
  const createURLEntry = (slug: SimpleSlug, content: ContentDetails): string => {
    let modifiedDate = content.date
    if (!modifiedDate && content.fileData!.frontmatter?.modified) {
      modifiedDate = new Date(content.fileData!.frontmatter.modified)
    }
    return `<url>
    <loc>https://${joinSegments(base, encodeURI(slug))}</loc>
    <lastmod>${modifiedDate?.toISOString()}</lastmod>
  </url>`
  }

  const urls = Array.from(idx)
    .map(([slug, content]) => createURLEntry(simplifySlug(slug), content))
    .join("")
  return `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">${urls}</urlset>`
}

function generateRSSFeed(cfg: GlobalConfiguration, idx: ContentIndexMap, limit?: number): string {
  const base = cfg.baseUrl ?? "example.com"

  const createURLEntry = (slug: SimpleSlug, content: ContentDetails): string => {
    return `<item>
    <title>${escapeHTML(content.title)}</title>
    <link>https://${joinSegments(base, encodeURI(slug))}</link>
    <guid>https://${joinSegments(base, encodeURI(slug))}</guid>
    <description><![CDATA[ ${content.description} ]]></description>
    <author>contact@aarnphm.xyz</author>
    <pubDate>${content.date?.toUTCString()}</pubDate>
    ${content.tags.map((el) => `<category domain="https://${joinSegments(base, "tags", el)}">${el}</category>`).join("\n")}
    <content:encoded>${content.richContent}</content:encoded>
  </item>`
  }

  const items = Array.from(idx)
    .sort(([_, f1], [__, f2]) => {
      if (f1.date && f2.date) {
        return f2.date.getTime() - f1.date.getTime()
      } else if (f1.date && !f2.date) {
        return -1
      } else if (!f1.date && f2.date) {
        return 1
      }

      return f1.title.localeCompare(f2.title)
    })
    .filter(([_, content]) => {
      if (
        !content.fileData ||
        !content.fileData.frontmatter ||
        !content.fileData.frontmatter.noindex
      )
        return true
      return content.fileData.frontmatter.noindex !== true
    })
    .map(([slug, content]) => createURLEntry(simplifySlug(slug), content))
    .slice(0, limit ?? idx.size)
    .join("")

  return `<?xml version="1.0" encoding="UTF-8" ?>
<?xml-stylesheet href="/static/rss.xsl" type="text/xsl" ?>
<rss version="2.0">
  <channel>
    <title>${escapeHTML(cfg.pageTitle)}</title>
    <link>https://${base}</link>
    <description>${!!limit ? i18n(cfg.locale).pages.rss.lastFewNotes({ count: limit }) : i18n(cfg.locale).pages.rss.recentNotes} on ${escapeHTML(
      cfg.pageTitle,
    )}</description>
    <generator>Quartz v${version} -- quartz.jzhao.xyz</generator>
    <managingEditor>contact@aarnphm.xyz (Aaron Pham)</managingEditor>
    <webMaster>contact@aarnphm.xyz (Aaron Pham)</webMaster>
    ${items}
  </channel>
</rss>`
}

function generateAtomFeed(cfg: GlobalConfiguration, idx: ContentIndexMap, limit?: number): string {
  const base = cfg.baseUrl ?? "example.com"
  const createURLEntry = (slug: SimpleSlug, content: ContentDetails): string => {
    let modifiedDate = content.date
    if (!modifiedDate && content.fileData!.frontmatter?.modified) {
      modifiedDate = new Date(content.fileData!.frontmatter.modified)
    }
    return `<entry>
    <title>${escapeHTML(content.title)}</title>
    <link href="https://${joinSegments(base, encodeURI(slug))}" />
    <link rel="alternate" type="text/markdown" href="https://${joinSegments(base, encodeURI(slug))}.html.md" />
    <summary>${content.description}</summary>
    <published>${content.date?.toISOString()}</published>
    <updated>${modifiedDate?.toISOString()}</updated>
    <publishedTime>${formatDate(content.date!, cfg.locale)}</publishedTime>
    <updatedTime>${formatDate(new Date(modifiedDate!), cfg.locale)}</updatedTime>
    ${content.tags.map((el) => `<category term="${el}" label="${el}" />`).join("\n")}
    <author>
      <name>Aaron Pham</name>
      <email>contact@aarnphm.xyz</email>
    </author>
    <content type="html">${content.richContent}</content>
  </entry>`
  }

  const items = Array.from(idx)
    .sort(([_, f1], [__, f2]) => {
      if (f1.date && f2.date) {
        return f2.date.getTime() - f1.date.getTime()
      } else if (f1.date && !f2.date) {
        return -1
      } else if (!f1.date && f2.date) {
        return 1
      }

      return f1.title.localeCompare(f2.title)
    })
    .filter(([_, content]) => {
      if (
        !content.fileData ||
        !content.fileData.frontmatter ||
        !content.fileData.frontmatter.noindex
      )
        return true
      return content.fileData.frontmatter.noindex !== true
    })
    .map(([slug, content]) => createURLEntry(simplifySlug(slug), content))
    .slice(0, limit ?? idx.size)
    .join("")

  return `<?xml version="1.0" encoding="UTF-8" ?>
<?xml-stylesheet href="/static/feed.xsl" type="text/xsl" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>${escapeHTML(cfg.pageTitle)}</title>
  <subtitle>${!!limit ? i18n(cfg.locale).pages.rss.lastFewNotes({ count: limit }) : i18n(cfg.locale).pages.rss.recentNotes} on ${escapeHTML(
    cfg.pageTitle,
  )}</subtitle>
  <link href="https://${base}" />
  <link rel="alternate" type="text/html" href="https://${base}" />
  <category term="evergreen" />
  <id>https://${base}</id>
  <updated>${idx.get("index" as FullSlug)!.date?.toISOString()}</updated>
  <contributor>
    <name>Aaron Pham</name>
    <email>contact@aarnphm.xyz</email>
  </contributor>
  <logo>https://${base}/icon.png</logo>
  <icon>https://${base}/icon.png</icon>
  <generator>Quartz v${version} -- quartz.jzhao.xyz</generator>
  <rights type="html">${escapeHTML(`&amp;copy; ${new Date().getFullYear()} Aaron Pham`)}</rights>
  ${items}
</feed>`
}

export const ContentIndex: QuartzEmitterPlugin<Partial<Options>> = (opts) => {
  opts = { ...defaultOptions, ...opts }
  return {
    name: "ContentIndex",
    async *emit(ctx, content, _resources) {
      const cfg = ctx.cfg.configuration

      const linkIndex: ContentIndexMap = new Map()
      for (const [tree, file] of content) {
        const slug = file.data.slug!
        const date = getDate(ctx.cfg.configuration, file.data) ?? new Date()

        if (opts?.includeEmptyFiles || (file.data.text && file.data.text !== "")) {
          // Filter out PDF links and links to noindex pages
          const links = (file.data.links ?? []).filter((link) => {
            // Skip links to pages with noindex: true
            // @ts-ignore
            const targetFile = content.find(([_, f]) => f.data.slug === link)?.[1]
            if (targetFile?.data.frontmatter?.noindex === true) return false

            return true
          })

          file.data.links = links

          linkIndex.set(slug, {
            slug,
            title: file.data.frontmatter?.title!,
            links,
            filePath: file.data.filePath!,
            fileName: file.data
              .filePath!.replace(".md", "")
              .substring(ctx.argv.directory.length + 1) as FilePath,
            tags: file.data.frontmatter?.tags ?? [],
            aliases: file.data.frontmatter?.aliases ?? [],
            content: file.data.text ?? "",
            richContent: escapeHTML(toHtml(tree as Root, { allowDangerousHtml: true })),
            date: date,
            readingTime: {
              minutes: Math.ceil(file.data.readingTime?.minutes!),
              words: Math.ceil(file.data.readingTime?.words!),
            },
            fileData: file.data,
            layout: file.data.frontmatter!.pageLayout,
            description: file.data.description,
          })
        }
      }

      yield write({
        ctx,
        content: `
User-agent: PerplexityBot
Disallow: /

User-agent: Perplexity-User
Disallow: /

User-agent: *
Allow: /
Host: https://${cfg.baseUrl}
Sitemap: https://${joinSegments(cfg.baseUrl ?? "https://example.com", "sitemap.xml")}
`,
        slug: "robots" as FullSlug,
        ext: ".txt",
      })

      if (opts?.enableSiteMap) {
        yield write({
          ctx,
          content: generateSiteMap(cfg, linkIndex),
          slug: "sitemap" as FullSlug,
          ext: ".xml",
        })
      }

      if (opts?.enableRSS) {
        yield write({
          ctx,
          content: generateRSSFeed(cfg, linkIndex, opts.rssLimit),
          slug: (opts?.rssSlug ?? "index") as FullSlug,
          ext: ".xml",
        })
      }

      if (opts?.enableAtom) {
        yield write({
          ctx,
          content: generateAtomFeed(cfg, linkIndex, opts.rssLimit),
          slug: "feed" as FullSlug,
          ext: ".xml",
        })
      }

      const fp = joinSegments("static", "contentIndex") as FullSlug
      const simplifiedIndex = Object.fromEntries(
        Array.from(linkIndex).map(([slug, content]) => {
          // remove richContent and fileData from content index as nothing downstream
          // actually uses it. we only keep it in the index as we need it
          // for the RSS feed
          delete content.fileData
          delete content.richContent
          return [slug, content]
        }),
      )

      yield write({
        ctx,
        content: JSON.stringify(simplifiedIndex),
        slug: fp,
        ext: ".json",
      })

      // inform Chrome to yield correct information
      if (process.env.NODE_ENV === "development" || process.env.NODE_ENV === "test") {
        // https://chromium.googlesource.com/devtools/devtools-frontend/+/main/docs/ecosystem/automatic_workspace_folders.md
        const slug = joinSegments(".well-known", "appspecific", "com.chrome.devtools") as FullSlug
        const root = path.resolve(path.dirname(ctx.argv.directory)) as FilePath
        const dir = path.dirname(joinSegments(ctx.argv.output, slug)) as FilePath
        await fs.mkdir(dir, { recursive: true })
        yield write({
          ctx,
          content: JSON.stringify({
            workspace: {
              root,
              uuid: crypto
                .createHash("sha256")
                .update(root)
                .digest("hex")
                .replace(/(.{8})(.{4})(.{4})(.{4})(.{12})/, "$1-$2-$3-$4-$5")
                .substring(0, 36),
            },
          }),
          slug,
          ext: ".json",
        })
      }
    },
    externalResources: ({ cfg }) => {
      const additionalHead = []
      if (opts?.enableRSS) {
        additionalHead.push(
          <link
            rel="alternate"
            type="application/rss+xml"
            title="RSS feed"
            href={`https://${cfg.configuration.baseUrl}/index.xml`}
          />,
        )
      }

      if (opts?.enableAtom) {
        additionalHead.push(
          <link
            rel="alternate"
            type="application/atom+xml"
            title="atom feed"
            href={`https://${cfg.configuration.baseUrl}/feed.xml`}
          />,
        )
      }

      return { additionalHead }
    },
  }
}
