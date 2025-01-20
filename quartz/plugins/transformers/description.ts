import { Root as HTMLRoot } from "hast"
import { toString } from "hast-util-to-string"
import { QuartzTransformerPlugin } from "../types"
import { escapeHTML } from "../../util/escape"
import readingTime, { ReadTimeResults } from "reading-time"

export interface Options {
  descriptionLength: number
  replaceExternalLinks: boolean
}

const defaultOptions: Options = {
  descriptionLength: 150,
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
    htmlPlugins() {
      return [
        () => {
          return (tree: HTMLRoot, file) => {
            let frontMatterDescription = file.data.frontmatter?.description
            let text = escapeHTML(toString(tree))

            if (opts.replaceExternalLinks) {
              frontMatterDescription = frontMatterDescription?.replace(
                urlRegex,
                "$<domain>" + "$<path>",
              )
              text = text.replace(urlRegex, "$<domain>" + "$<path>")
            }

            const processDescription = (desc: string, includeTripleDots: boolean): string => {
              const sentences = desc.replace(/\s+/g, " ").split(/\.\s/)
              const finalDesc: string[] = []
              const len = opts.descriptionLength
              let sentenceIdx = 0
              let currentDescriptionLength = 0

              if (sentences[0] !== undefined && sentences[0].length >= len) {
                const firstSentence = sentences[0].split(" ")
                while (currentDescriptionLength < len) {
                  const sentence = firstSentence[sentenceIdx]
                  if (!sentence) break
                  finalDesc.push(sentence)
                  currentDescriptionLength += sentence.length
                  sentenceIdx++
                }
                if (includeTripleDots) finalDesc.push("...")
              } else {
                while (currentDescriptionLength < len) {
                  const sentence = sentences[sentenceIdx]
                  if (!sentence) break
                  const currentSentence = sentence.endsWith(".") ? sentence : sentence + "."
                  finalDesc.push(currentSentence)
                  currentDescriptionLength += currentSentence.length
                  sentenceIdx++
                }
              }
              return finalDesc.join(" ")
            }

            file.data.description = processDescription(frontMatterDescription ?? text, true)
            file.data.abstract = file.data.frontmatter?.abstract ?? processDescription(text, false)
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
