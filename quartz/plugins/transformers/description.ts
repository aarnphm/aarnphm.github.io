import { Root as HTMLRoot } from "hast"
import { toString } from "hast-util-to-string"
import { QuartzTransformerPlugin } from "../../types/plugin"
import { escapeHTML } from "../../util/escape"
import readingTime, { ReadTimeResults } from "reading-time"
import { i18n } from "../../i18n"
import {
  stripWikilinkFormatting,
  extractWikilinks,
  resolveWikilinkTarget,
} from "../../util/wikilinks"
import { simplifySlug, type FullSlug, type SimpleSlug } from "../../util/path"
import { processWikilinksToHtml, renderLatexInString } from "../../util/description"

export interface Options {
  descriptionLength: number
  maxDescriptionLength: number
  replaceExternalLinks: boolean
}

const defaultOptions: Options = {
  descriptionLength: 150,
  maxDescriptionLength: 300,
  replaceExternalLinks: true,
}

const urlRegex = new RegExp(
  /(https?:\/\/)?(?<domain>([\da-z\.-]+)\.([a-z\.]{2,6})(:\d+)?)(?<path>[\/\w\.-]*)(\?[\/\w\.=&;-]*)?/,
  "g",
)

export const Description: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "Description",
    htmlPlugins({ cfg }) {
      return [
        () => {
          return async (tree: HTMLRoot, file) => {
            const currentSlug = file.data.slug as FullSlug
            const descriptionLinks: Set<SimpleSlug> = new Set()

            let frontMatterDescription = file.data.frontmatter?.description

            // Extract and track wikilinks from frontmatter description
            if (typeof frontMatterDescription === "string") {
              const wikilinks = extractWikilinks(frontMatterDescription)
              for (const link of wikilinks) {
                const resolved = resolveWikilinkTarget(link, currentSlug)
                if (resolved) {
                  descriptionLinks.add(simplifySlug(resolved.slug))
                }
              }
            }

            let text = escapeHTML(toString(tree))

            if (opts.replaceExternalLinks) {
              frontMatterDescription = frontMatterDescription?.replace(
                urlRegex,
                "$<domain>" + "$<path>",
              )
              text = text.replace(urlRegex, "$<domain>" + "$<path>")
            }

            const processDescription = (desc: string): string => {
              const sentences = desc.replace(/\s+/g, " ").split(/\.\s/)
              let finalDesc = ""
              let sentenceIdx = 0

              // Add full sentences until we exceed the guideline length
              while (sentenceIdx < sentences.length) {
                const sentence = sentences[sentenceIdx]
                if (!sentence) break

                const currentSentence = sentence.endsWith(".") ? sentence : sentence + "."
                const nextLength = finalDesc.length + currentSentence.length + (finalDesc ? 1 : 0)

                // Add the sentence if we're under the guideline length
                // or if this is the first sentence (always include at least one)
                if (nextLength <= opts.descriptionLength || sentenceIdx === 0) {
                  finalDesc += (finalDesc ? " " : "") + currentSentence
                  sentenceIdx++
                } else {
                  break
                }
              }
              return finalDesc.length > opts.maxDescriptionLength
                ? finalDesc.slice(0, opts.maxDescriptionLength) + "..."
                : finalDesc
            }

            // Process frontmatter description with wikilinks converted to HTML and LaTeX rendered
            let processedFrontMatterDesc = frontMatterDescription
              ? renderLatexInString(processWikilinksToHtml(frontMatterDescription, currentSlug))
              : undefined

            // For length calculation and truncation, use plain text
            const plainTextForProcessing = frontMatterDescription
              ? stripWikilinkFormatting(frontMatterDescription)
              : text

            const processedPlainDesc = processDescription(plainTextForProcessing)

            // If we had a frontmatter description, truncate the HTML version to match processed length
            if (
              processedFrontMatterDesc &&
              processedPlainDesc.length < plainTextForProcessing.length
            ) {
              // Description was truncated, apply same truncation to HTML version
              processedFrontMatterDesc = processDescription(processedFrontMatterDesc)
            }

            file.data.description =
              processedFrontMatterDesc ||
              processedPlainDesc ||
              i18n(cfg.configuration.locale).propertyDefaults.description

            // Process abstract with wikilinks support
            let abstractText = file.data.frontmatter?.abstract
            if (abstractText) {
              // Extract and track wikilinks from abstract
              const abstractWikilinks = extractWikilinks(abstractText)
              for (const link of abstractWikilinks) {
                const resolved = resolveWikilinkTarget(link, currentSlug)
                if (resolved) {
                  descriptionLinks.add(simplifySlug(resolved.slug))
                }
              }
              // Convert wikilinks to HTML and render LaTeX in abstract
              abstractText = renderLatexInString(processWikilinksToHtml(abstractText, currentSlug))
            }

            file.data.abstract = abstractText ?? processDescription(text)
            file.data.text = text
            file.data.readingTime = readingTime(file.data.text!)

            // Merge description links with existing links
            if (descriptionLinks.size > 0) {
              const existingLinks = file.data.links || []
              file.data.links = [...new Set([...existingLinks, ...descriptionLinks])]
            }
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    description: string
    abstract: string
    text: string
    readingTime: ReadTimeResults
  }
}
