import { Root as HTMLRoot } from "hast"
import { toString } from "hast-util-to-string"
import { QuartzTransformerPlugin } from "../types"
import { escapeHTML, unescapeHTML } from "../../util/escape"
import readingTime, { ReadTimeResults } from "reading-time"
import { i18n } from "../../i18n"
import { stripWikilinkFormatting } from "../../util/wikilinks"

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
            let frontMatterDescription = file.data.frontmatter?.description
            if (typeof frontMatterDescription === "string") {
              frontMatterDescription = stripWikilinkFormatting(frontMatterDescription)
            }
            let text = escapeHTML(toString(tree))

            if (opts.replaceExternalLinks) {
              frontMatterDescription = frontMatterDescription?.replace(
                urlRegex,
                "$<domain>" + "$<path>",
              )
              text = text.replace(urlRegex, "$<domain>" + "$<path>")
            }

            // truncate to max length if necessary
            file.data.description = file.data.text = text
            file.data.readingTime = readingTime(file.data.text!)

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

            const description = processDescription(frontMatterDescription ?? text)
            file.data.description = unescapeHTML(
              frontMatterDescription ||
                description.trim() ||
                i18n(cfg.configuration.locale).propertyDefaults.description,
            )
            file.data.description = description
            file.data.abstract = file.data.frontmatter?.abstract ?? processDescription(text)
            file.data.text = text
            file.data.readingTime = readingTime(file.data.text!)
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
