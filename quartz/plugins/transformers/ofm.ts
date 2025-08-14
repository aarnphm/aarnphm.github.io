import { QuartzTransformerPlugin } from "../types"
import {
  Root,
  Html,
  BlockContent,
  DefinitionContent,
  Paragraph,
  Code,
  PhrasingContent,
} from "mdast"
import { Element, Literal, Root as HtmlRoot } from "hast"
import { ReplaceFunction, findAndReplace as mdastFindReplace } from "mdast-util-find-and-replace"
import rehypeRaw from "rehype-raw"
import { SKIP, visit } from "unist-util-visit"
import path from "path"
import { splitAnchor } from "../../util/path"
import { CSSResource, JSResource } from "../../util/resources"
// @ts-ignore
import calloutScript from "../../components/scripts/callout.inline.ts"
import { FilePath, pathToRoot, slugTag, slugifyFilePath } from "../../util/path"
import { toHast } from "mdast-util-to-hast"
import { toHtml } from "hast-util-to-html"
import { toString } from "mdast-util-to-string"
import { capitalize } from "../../util/lang"
import { PluggableList } from "unified"
import { h, s } from "hastscript"
import { whitespace } from "hast-util-whitespace"
import { remove } from "unist-util-remove"
import { svgOptions } from "../../components/svg"

export interface Options {
  comments: boolean
  highlight: boolean
  wikilinks: boolean
  callouts: boolean
  mermaid: boolean
  parseTags: boolean
  parseArrows: boolean
  parseBlockReferences: boolean
  enableInHtmlEmbed: boolean
  enableYouTubeEmbed: boolean
  enableVideoEmbed: boolean
  enableInlineFootnotes: boolean
  enableImageGrid: boolean
  enableBrokenWikilinks: boolean
}

const defaultOptions: Options = {
  comments: true,
  highlight: true,
  wikilinks: true,
  callouts: true,
  mermaid: true,
  parseTags: true,
  parseArrows: true,
  parseBlockReferences: true,
  enableInHtmlEmbed: false,
  enableYouTubeEmbed: true,
  enableVideoEmbed: true,
  enableInlineFootnotes: true,
  enableImageGrid: true,
  enableBrokenWikilinks: false,
}

const calloutMapping = {
  note: "note",
  abstract: "abstract",
  summary: "abstract",
  tldr: "abstract",
  info: "info",
  todo: "todo",
  tip: "tip",
  hint: "tip",
  important: "tip",
  success: "success",
  check: "success",
  done: "success",
  question: "question",
  help: "question",
  faq: "question",
  warning: "warning",
  attention: "warning",
  caution: "warning",
  failure: "failure",
  missing: "failure",
  fail: "failure",
  danger: "danger",
  error: "danger",
  bug: "bug",
  example: "example",
  quote: "quote",
  cite: "quote",
} as const

const arrowMapping: Record<string, string> = {
  "->": "&rarr;",
  "-->": "&rArr;",
  "=>": "&rArr;",
  "==>": "&rArr;",
  "<-": "&larr;",
  "<--": "&lArr;",
  "<=": "&lArr;",
  "<==": "&lArr;",
}

function canonicalizeCallout(calloutName: string): keyof typeof calloutMapping {
  const normalizedCallout = calloutName.toLowerCase() as keyof typeof calloutMapping
  // if callout is not recognized, make it a custom one
  return calloutMapping[normalizedCallout] ?? calloutName
}

export const externalLinkRegex = /^https?:\/\//i

export const arrowRegex = new RegExp(/(-{1,2}>|={1,2}>|<-{1,2}|<={1,2})/g)

// !?                 -> optional embedding
// \[\[               -> open brace
// ([^\[\]\|\#]+)     -> one or more non-special characters ([,],|, or #) (name)
// (#[^\[\]\|\#]+)?   -> # then one or more non-special characters (heading link)
// (\\?\|[^\[\]\#]+)? -> optional escape \ then | then zero or more non-special characters (alias)
export const wikilinkRegex = new RegExp(
  /!?\[\[([^\[\]\|\#\\]+)?(#+[^\[\]\|\#\\]+)?(\\?\|[^\[\]\#]*)?\]\]/g,
)

export const inlineFootnoteRegex = /\^\[((?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*)\]/g

// ^\|([^\n])+\|\n(\|) -> matches the header row
// ( ?:?-{3,}:? ?\|)+  -> matches the header row separator
// (\|([^\n])+\|\n)+   -> matches the body rows
export const tableRegex = new RegExp(/^\|([^\n])+\|\n(\|)( ?:?-{3,}:? ?\|)+\n(\|([^\n])+\|\n?)+/gm)

// matches any wikilink, only used for escaping wikilinks inside tables
export const tableWikilinkRegex = new RegExp(/(!?\[\[[^\]]*?\]\]|\[\^[^\]]*?\])/g)

const highlightRegex = new RegExp(/==([^=]+)==/g)
const commentRegex = new RegExp(/%%[\s\S]*?%%/g)
// from https://github.com/escwxyz/remark-obsidian-callout/blob/main/src/index.ts
const calloutRegex = new RegExp(/^\[\!([\w-]+)\|?(.+?)?\]([+-]?)/)
const calloutLineRegex = new RegExp(/^> *\[\!\w+\|?.*?\][+-]?.*$/gm)
// (?<=^| )             -> a lookbehind assertion, tag should start be separated by a space or be the start of the line
// #(...)               -> capturing group, tag itself must start with #
// (?:[-_\p{L}\d\p{Z}])+       -> non-capturing group, non-empty string of (Unicode-aware) alpha-numeric characters and symbols, hyphens and/or underscores
// (?:\/[-_\p{L}\d\p{Z}]+)*)   -> non-capturing group, matches an arbitrary number of tag strings separated by "/"
const tagRegex = new RegExp(
  /(?<=^| )#((?:[-_\p{L}\p{Emoji}\p{M}\d])+(?:\/[-_\p{L}\p{Emoji}\p{M}\d]+)*)/gu,
)
const blockReferenceRegex = new RegExp(/\^([-_A-Za-z0-9]+)$/g)
const ytLinkRegex = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/
const ytPlaylistLinkRegex = /[?&]list=([^#?&]*)/
const videoExtensionRegex = new RegExp(/\.(mp4|webm|ogg|avi|mov|flv|wmv|mkv|mpg|mpeg|3gp|m4v)$/)
const wikilinkImageEmbedRegex = new RegExp(
  /^(?<alt>(?!^\d*x?\d*$).*?)?(\|?\s*?(?<width>\d+)(x(?<height>\d+))?)?$/,
)

export const checkMermaidCode = ({ tagName, properties }: Element) =>
  tagName === "code" &&
  Boolean(properties.className) &&
  (properties.className as string[]).includes("mermaid")

export const wikiTextTransform = (src: string) => {
  // replace all wikilinks inside a table first
  src = src.replace(tableRegex, (value) => {
    // escape all aliases and headers in wikilinks inside a table
    return value.replace(tableWikilinkRegex, (_value, raw) => {
      // const [raw]: (string | undefined)[] = capture
      let escaped = raw ?? ""
      escaped = escaped.replace("#", "\\#")
      // escape pipe characters if they are not already escaped
      escaped = escaped.replace(/((^|[^\\])(\\\\)*)\|/g, "$1\\|")

      return escaped
    })
  })

  // replace all other wikilinks
  src = src.replace(wikilinkRegex, (value, ...capture) => {
    const [rawFp, rawHeader, rawAlias]: (string | undefined)[] = capture

    const [fp, anchor] = splitAnchor(`${rawFp ?? ""}${rawHeader ?? ""}`)
    const blockRef = Boolean(rawHeader?.startsWith("#^")) ? "^" : ""
    const displayAnchor = anchor ? `#${blockRef}${anchor.trim().replace(/^#+/, "")}` : ""
    const displayAlias = rawAlias ?? rawHeader?.replace("#", "|") ?? ""
    const embedDisplay = value.startsWith("!") ? "!" : ""

    if (rawFp?.match(externalLinkRegex)) {
      return `${embedDisplay}[${displayAlias.replace(/^\|/, "")}](${rawFp})`
    }

    return `${embedDisplay}[[${fp}${displayAnchor}${displayAlias}]]`
  })
  return src
}

export const ObsidianFlavoredMarkdown: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  const allowDangerousHtml = true

  const mdastToHtml = (ast: PhrasingContent | Paragraph) => {
    const hast = toHast(ast, { allowDangerousHtml })!
    return toHtml(hast, { allowDangerousHtml })
  }

  return {
    name: "ObsidianFlavoredMarkdown",
    textTransform(_, src: any) {
      // do comments at text level
      if (opts.comments) {
        src = src.replace(commentRegex, "")
      }

      // pre-transform blockquotes
      if (opts.callouts) {
        src = src.replace(calloutLineRegex, (value: string) => {
          // force newline after title of callout
          return value + "\n> "
        })
      }

      // pre-transform wikilinks (fix anchors to things that may contain illegal syntax e.g. codeblocks, latex)
      if (opts.wikilinks) {
        src = wikiTextTransform(src)
      }

      if (opts.enableInlineFootnotes) {
        // Replaces ^[inline] footnotes with regular footnotes [^1]:
        const footnotes: Record<string, string> = {}
        let counter = 0

        // Replace inline footnotes with references and collect definitions
        const result = src.replace(inlineFootnoteRegex, (_match: string, content: string) => {
          counter++
          const id = `generated-inline-footnote-${counter}`
          footnotes[id] = content.trim()
          return `[^${id}]`
        })

        // Append footnote definitions if any are found
        if (Object.keys(footnotes).length > 0) {
          return (
            result +
            "\n\n" +
            Object.entries(footnotes)
              .map(([id, content]) => `[^${id}]: ${content}`)
              .join("\n") +
            "\n"
          )
        }
      }

      return src
    },
    markdownPlugins(ctx) {
      const plugins: PluggableList = []

      // regex replacements
      plugins.push(() => {
        return (tree: Root, file) => {
          const replacements: [RegExp, string | ReplaceFunction][] = []
          const base = pathToRoot(file.data.slug!)

          if (opts.wikilinks) {
            replacements.push([
              wikilinkRegex,
              (value: string, ...capture: string[]) => {
                let [rawFp, rawHeader, rawAlias] = capture
                const fp = rawFp?.trim() ?? ""
                const anchor = rawHeader?.trim() ?? ""
                const alias: string | undefined = rawAlias?.slice(1).trim()

                // embed cases
                if (value.startsWith("!")) {
                  const ext: string = path.extname(fp).toLowerCase()
                  const url = slugifyFilePath(fp as FilePath)
                  switch (ext) {
                    // images
                    case ".png":
                    case ".jpg":
                    case ".jpeg":
                    case ".gif":
                    case ".bmp":
                    case ".svg":
                    case ".webp": {
                      const match = wikilinkImageEmbedRegex.exec(alias ?? "")
                      const alt = match?.groups?.alt ?? ""
                      const width = match?.groups?.width ?? "auto"
                      const height = match?.groups?.height ?? "auto"
                      return {
                        type: "image",
                        url,
                        data: {
                          hProperties: {
                            width,
                            height,
                            alt,
                          },
                        },
                      }
                    }
                    // videos
                    case ".mp4":
                    case ".webm":
                    case ".ogv":
                    case ".mov":
                    case ".mkv": {
                      return {
                        type: "html",
                        value: `<video src="${url}" controls loop></video>`,
                      }
                    }
                    // audio
                    case ".mp3":
                    case ".wav":
                    case ".m4a":
                    case ".ogg":
                    case ".3gp":
                    case ".flac": {
                      return {
                        type: "html",
                        value: `<audio src="${url}" controls></audio>`,
                      }
                    }
                    // documents
                    case ".pdf": {
                      return {
                        type: "html",
                        value: `<iframe src="${url}" class="pdf"></iframe>`,
                      }
                    }
                    // default: transclude block
                    default: {
                      const block = anchor
                      return {
                        type: "html",
                        data: { hProperties: { transclude: true } },
                        value: toHtml(
                          h(
                            "blockquote.transclude",
                            { "data-url": url, "data-block": block, "data-embed-alias": alias },
                            h("a.transclude-inner", { href: url + anchor }, [
                              { type: "text", value: `Transclude of ${url} ${block}` },
                            ]),
                          ),
                          { allowDangerousHtml },
                        ),
                      }
                    }
                  }

                  // otherwise, fall through to regular link
                }

                // internal link
                const url = fp + anchor

                // treat as broken link if slug not in ctx.allSlugs
                if (opts.enableBrokenWikilinks) {
                  const slug = slugifyFilePath(fp as FilePath)
                  const exists = ctx.allSlugs && ctx.allSlugs.includes(slug)
                  if (!exists) {
                    return {
                      type: "link",
                      url,
                      data: {
                        hProperties: {
                          className: ["broken"],
                        },
                      },
                      children: [
                        {
                          type: "text",
                          value: alias ?? fp,
                        },
                      ],
                    }
                  }
                }

                return {
                  type: "link",
                  url,
                  children: [
                    {
                      type: "text",
                      value: alias ?? fp,
                    },
                  ],
                }
              },
            ])
          }

          if (opts.highlight) {
            replacements.push([
              highlightRegex,
              (_value: string, ...capture: string[]) => {
                const [inner] = capture
                return { type: "html", value: `<mark>${inner}</mark>` }
              },
            ])
          }

          if (opts.parseArrows) {
            replacements.push([
              arrowRegex,
              (value: string, ..._capture: string[]) => {
                const maybeArrow = arrowMapping[value]
                if (maybeArrow === undefined) return SKIP
                return {
                  type: "html",
                  value: `<span>${maybeArrow}</span>`,
                }
              },
            ])
          }

          if (opts.parseTags) {
            replacements.push([
              tagRegex,
              (_value: string, tag: string) => {
                // Check if the tag only includes numbers and slashes
                if (/^[\/\d]+$/.test(tag)) {
                  return false
                }

                tag = slugTag(tag)
                if (file.data.frontmatter) {
                  const noteTags = file.data.frontmatter.tags ?? []
                  file.data.frontmatter.tags = [...new Set([...noteTags, tag])]
                }

                return {
                  type: "link",
                  url: base + `/tags/${tag}`,
                  data: {
                    hProperties: {
                      className: ["tag-link"],
                    },
                  },
                  children: [
                    {
                      type: "text",
                      value: tag,
                    },
                  ],
                }
              },
            ])
          }

          if (opts.enableInHtmlEmbed) {
            visit(tree, "html", (node) => {
              for (const [regex, replace] of replacements) {
                if (typeof replace === "string") {
                  node.value = node.value.replace(regex, replace)
                } else {
                  node.value = node.value.replace(regex, (substring: string, ...args) => {
                    const replaceValue = replace(substring, ...args)
                    if (typeof replaceValue === "string") {
                      return replaceValue
                    } else if (Array.isArray(replaceValue)) {
                      return replaceValue.map(mdastToHtml).join("")
                    } else if (typeof replaceValue === "object" && replaceValue !== null) {
                      return mdastToHtml(replaceValue)
                    } else {
                      return substring
                    }
                  })
                }
              }
            })
          }
          mdastFindReplace(tree, replacements)
        }
      })

      if (opts.enableVideoEmbed) {
        plugins.push(() => {
          return (tree: Root, _file) => {
            visit(tree, "image", (node, index, parent) => {
              if (parent && index != undefined && videoExtensionRegex.test(node.url)) {
                const newNode: Html = {
                  type: "html",
                  value: `<video controls src="${node.url}"></video>`,
                }

                parent.children.splice(index, 1, newNode)
                return SKIP
              }
            })
          }
        })
      }

      if (opts.callouts) {
        plugins.push(() => {
          return (tree: Root, _file) => {
            visit(tree, "blockquote", (node) => {
              if (node.children.length === 0) {
                return
              }

              // find first line and callout content
              const [firstChild, ...calloutContent] = node.children
              if (firstChild.type !== "paragraph" || firstChild.children[0]?.type !== "text") {
                return
              }

              const text = firstChild.children[0].value
              const restOfTitle = firstChild.children.slice(1)
              const [firstLine, ...remainingLines] = text.split("\n")
              const remainingText = remainingLines.join("\n")

              const match = firstLine.match(calloutRegex)
              if (match && match.input) {
                const [calloutDirective, typeString, calloutMetaData, collapseChar] = match
                const calloutType = canonicalizeCallout(typeString.toLowerCase())
                const collapse = collapseChar === "+" || collapseChar === "-"
                const defaultState = collapseChar === "-" ? "collapsed" : "expanded"
                const titleContent = match.input.slice(calloutDirective.length).trim()
                const useDefaultTitle = titleContent === "" && restOfTitle.length === 0
                const titleNode: Paragraph = {
                  type: "paragraph",
                  children: [
                    {
                      type: "text",
                      value: useDefaultTitle
                        ? capitalize(typeString).replace(/-/g, " ")
                        : titleContent + " ",
                    },
                    ...restOfTitle,
                  ],
                }
                const titleChildren = [
                  h(".callout-icon"),
                  h(".callout-title-inner", toHast(titleNode, { allowDangerousHtml })),
                ]
                if (collapse) titleChildren.push(h(".fold-callout-icon"))

                const titleHtml: Html = {
                  type: "html",
                  value: toHtml(h(".callout-title", titleChildren), { allowDangerousHtml }),
                }

                const blockquoteContent: (BlockContent | DefinitionContent)[] = [titleHtml]
                if (remainingText.length > 0) {
                  blockquoteContent.push({
                    type: "paragraph",
                    children: [
                      {
                        type: "text",
                        value: remainingText,
                      },
                    ],
                  })
                }

                // replace first line of blockquote with title and rest of the paragraph text
                node.children.splice(0, 1, ...blockquoteContent)

                const classNames = ["callout", calloutType]
                if (collapse) {
                  classNames.push("is-collapsible")
                }
                if (defaultState === "collapsed") {
                  classNames.push("is-collapsed")
                }

                // add properties to base blockquote
                node.data = {
                  hProperties: {
                    ...(node.data?.hProperties ?? {}),
                    className: classNames.join(" "),
                    "data-callout": calloutType,
                    "data-callout-fold": collapse,
                    "data-callout-metadata": calloutMetaData,
                  },
                }

                // Add callout-content class to callout body if it has one.
                if (calloutContent.length > 0) {
                  const contentData: BlockContent | DefinitionContent = {
                    data: {
                      hProperties: {
                        className: "callout-content",
                      },
                      hName: "div",
                    },
                    type: "blockquote",
                    children: [...calloutContent],
                  }
                  node.children = [node.children[0], contentData]
                }
              }
            })
          }
        })
      }

      if (opts.mermaid) {
        plugins.push(() => {
          return (tree) => {
            visit(tree, "code", (node: Code) => {
              if (node.lang === "mermaid") {
                node.data = {
                  hProperties: {
                    className: ["mermaid"],
                    "data-clipboard": toString(node),
                  },
                }
              }
            })
          }
        })
      }

      if (opts.enableImageGrid) {
        plugins.push(() => {
          return (tree: Root) => {
            visit(tree, "paragraph", (node: Paragraph, index: number | undefined, parent) => {
              if (index === undefined || parent === undefined) return

              const isOnlyImages = node.children.every((child) => {
                if (child.type === "image") return true
                if (child.type === "text") return (child.value as string).trim() === ""
                return false
              })

              const imageNodes = node.children.filter((c) => c.type === "image")
              if (isOnlyImages && imageNodes.length >= 2) {
                const htmlContent = node.children
                  .filter((c) => c.type === "image")
                  .map((img) => mdastToHtml(img))
                  .join("\n")

                const gridNode: Html = {
                  type: "html",
                  value: `<div class="image-grid">\n${htmlContent}\n</div>`,
                }

                parent.children.splice(index, 1, gridNode)
              }
            })
          }
        })
      }

      return plugins
    },
    htmlPlugins() {
      const plugins: PluggableList = [rehypeRaw]

      if (opts.parseBlockReferences) {
        plugins.push(() => {
          const inlineTagTypes = new Set(["p", "li"])
          const blockTagTypes = new Set(["blockquote"])
          return (tree: HtmlRoot, file) => {
            file.data.blocks = {}

            visit(tree, "element", (node, index, parent) => {
              if (blockTagTypes.has(node.tagName)) {
                const nextChild = parent?.children.at(index! + 2) as Element
                if (nextChild && nextChild.tagName === "p") {
                  const text = nextChild.children.at(0) as Literal
                  if (text && text.value && text.type === "text") {
                    const matches = text.value.match(blockReferenceRegex)
                    if (matches && matches.length >= 1) {
                      parent!.children.splice(index! + 2, 1)
                      const block = matches[0].slice(1)

                      if (!Object.keys(file.data.blocks!).includes(block)) {
                        node.properties = {
                          ...node.properties,
                          id: block,
                        }
                        file.data.blocks![block] = node
                      }
                    }
                  }
                }
              } else if (inlineTagTypes.has(node.tagName)) {
                const last = node.children.at(-1) as Literal
                if (last && last.value && typeof last.value === "string") {
                  const matches = last.value.match(blockReferenceRegex)
                  if (matches && matches.length >= 1) {
                    last.value = last.value.slice(0, -matches[0].length)
                    const block = matches[0].slice(1)

                    if (last.value === "") {
                      // this is an inline block ref but the actual block
                      // is the previous element above it
                      let idx = (index ?? 1) - 1
                      while (idx >= 0) {
                        const element = parent?.children.at(idx)
                        if (!element) break
                        if (element.type !== "element") {
                          idx -= 1
                        } else {
                          if (!Object.keys(file.data.blocks!).includes(block)) {
                            element.properties = {
                              ...element.properties,
                              id: block,
                            }
                            file.data.blocks![block] = element
                          }
                          return
                        }
                      }
                    } else {
                      // normal paragraph transclude
                      if (!Object.keys(file.data.blocks!).includes(block)) {
                        node.properties = {
                          ...node.properties,
                          id: block,
                        }
                        file.data.blocks![block] = node
                      }
                    }
                  }
                }
              }
            })

            file.data.htmlAst = tree
          }
        })
      }

      if (opts.highlight) {
        plugins.push(() => {
          return (tree) => {
            visit(tree, { tagName: "p" }, (node) => {
              const stack: number[] = []
              const highlights: [number, number][] = []
              const children = [...node.children]

              for (let i = 0; i < children.length; i++) {
                const child = children[i]
                if (child.type === "text" && child.value.includes("==")) {
                  // Split text node if it contains == marker
                  const parts: string[] = child.value.split("==")

                  if (parts.length > 1) {
                    // Replace original node with split parts
                    const newNodes: (typeof child)[] = []

                    parts.forEach((part, idx) => {
                      if (part) {
                        newNodes.push({ type: "text", value: part })
                      }
                      // Add marker position except for last part
                      if (idx < parts.length - 1) {
                        if (stack.length === 0) {
                          stack.push(i + newNodes.length)
                        } else {
                          const start = stack.pop()!
                          highlights.push([start, i + newNodes.length])
                        }
                      }
                    })

                    children.splice(i, 1, ...newNodes)
                    i += newNodes.length - 1
                  }
                }
              }

              // Apply highlights in reverse to maintain indices
              for (const [start, end] of highlights.reverse()) {
                const highlightSpan: Element = {
                  type: "element",
                  tagName: "mark",
                  properties: {},
                  children: children.slice(start, end + 1),
                }
                children.splice(start, end - start + 1, highlightSpan)
              }

              node.children = children
            })
          }
        })
      }

      if (opts.enableYouTubeEmbed) {
        const checkEmbed = ({ tagName, properties }: Element) =>
          tagName === "img" && Boolean(properties.src) && typeof properties.src === "string"

        plugins.push(() => {
          return (tree) => {
            visit(tree, (node: Element) => {
              if (checkEmbed(node)) {
                const src = (node as Element).properties.src as string
                const match = src.match(ytLinkRegex)
                const videoId = match && match[2].length == 11 ? match[2] : null
                const playlistId = src.match(ytPlaylistLinkRegex)?.[1]

                const baseProperties = {
                  class: "external-embed youtube",
                  allow: "fullscreen",
                  frameborder: 0,
                  width: "600px",
                }

                switch (true) {
                  case Boolean(videoId && playlistId):
                    // Video with playlist
                    node.tagName = "iframe"
                    node.properties = {
                      ...baseProperties,
                      src: `https://www.youtube.com/embed/${videoId}?list=${playlistId}`,
                    }
                    break
                  case Boolean(videoId):
                    // Single video
                    node.tagName = "iframe"
                    node.properties = {
                      ...baseProperties,
                      src: `https://www.youtube.com/embed/${videoId}`,
                    }
                    break
                  case Boolean(playlistId):
                    // Playlist only
                    node.tagName = "iframe"
                    node.properties = {
                      ...baseProperties,
                      src: `https://www.youtube.com/embed/videoseries?list=${playlistId}`,
                    }
                    break
                }
              }
            })
          }
        })
      }

      if (opts.mermaid) {
        plugins.push(() => {
          return (tree) => {
            visit(
              tree,
              (node) => checkMermaidCode(node as Element),
              (node: Element, _, parent: HtmlRoot) => {
                parent.children = [
                  h(
                    "span.expand-button",
                    {
                      type: "button",
                      ariaLabel: "Expand mermaid diagram",
                      tabindex: -1,
                      "data-view-component": true,
                    },
                    [
                      s("svg", { ...svgOptions, viewbox: "0 -8 24 24", tabindex: -1 }, [
                        s("use", { href: "#expand-e-w" }),
                      ]),
                    ],
                  ),
                  h(
                    "span.clipboard-button",
                    {
                      type: "button",
                      ariaLabel: "copy source",
                    },
                    [
                      s("svg", { ...svgOptions, viewbox: "0 -8 24 24", class: "copy-icon" }, [
                        s("use", { href: "#github-copy" }),
                      ]),
                      s("svg", { ...svgOptions, viewbox: "0 -8 24 24", class: "check-icon" }, [
                        s("use", { href: "#github-check" }),
                      ]),
                    ],
                  ),
                  node,
                  h("#mermaid-container", { role: "dialog" }, [
                    h("#mermaid-space", [h(".mermaid-content")]),
                  ]),
                ]
              },
            )
          }
        })
      }

      plugins.push(() => {
        return (tree, file) => {
          const onlyImage = ({ children }: Element) =>
            children.every((child) => (child as Element).tagName === "img" || whitespace(child))
          const withAlt = ({ tagName, properties }: Element) =>
            tagName === "img" && Boolean(properties.alt) && Boolean(properties.src)
          const withCaption = ({ tagName, children }: Element) => {
            return (
              tagName === "figure" &&
              children.some((child) => (child as Element).tagName === "figcaption")
            )
          }

          // support better image captions
          visit(tree, { tagName: "p" }, (node, idx, parent) => {
            if (!onlyImage(node)) return
            remove(node, "text")
            parent?.children.splice(idx!, 1, ...node.children)
            return idx
          })

          file.data.images = {}
          let counter = 0

          visit(
            tree,
            (node) => withAlt(node as Element),
            (node, idx, parent) => {
              if (withCaption(parent as Element) || (parent as Element)!.tagName === "a") {
                return
              }

              counter++
              parent?.children.splice(
                idx!,
                1,
                h("figure", { "data-img-w-caption": true }, [
                  h("img", { ...(node as Element).properties }),
                  h("figcaption", [
                    h("span", { class: "figure-prefix", style: "margin-right: 0.2em;" }, `figure`),
                    h("span", { class: "figure-number" }, `${counter}`),
                    h("span", { class: "figure-caption" }, `: ${(node as Element).properties.alt}`),
                  ]),
                ]),
              )
            },
          )
        }
      })

      return plugins
    },
    externalResources() {
      const js: JSResource[] = []
      const css: CSSResource[] = []

      if (opts.callouts) {
        js.push({
          script: calloutScript,
          loadTime: "afterDOMReady",
          contentType: "inline",
        })
      }

      return { js, css }
    },
  }
}

declare module "vfile" {
  interface DataMap {
    images: Record<string, { count: number; el: Element }>
    blocks: Record<string, Element>
    htmlAst: HtmlRoot
  }
}
