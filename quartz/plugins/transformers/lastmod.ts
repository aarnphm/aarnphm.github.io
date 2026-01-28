import { Repository } from "@napi-rs/simple-git"
import fs from "node:fs/promises"
import { styleText } from "node:util"
import path from "path"
import { QuartzTransformerPlugin } from "../../types/plugin"

export interface Options {
  priority: ("frontmatter" | "git" | "filesystem")[]
}

const defaultOptions: Options = { priority: ["frontmatter", "git", "filesystem"] }

// YYYY-MM-DD
const iso8601DateOnlyRegex = /^\d{4}-\d{2}-\d{2}$/

function coerceDate(fp: string, d: any): Date {
  // check ISO8601 date-only format
  // we treat this one as local midnight as the normal
  // js date ctor treats YYYY-MM-DD as UTC midnight
  if (typeof d === "string" && iso8601DateOnlyRegex.test(d)) {
    d = `${d}T00:00:00`
  }

  const dt = new Date(d)
  const invalidDate = isNaN(dt.getTime()) || dt.getTime() === 0
  if (invalidDate && d !== undefined) {
    console.log(
      styleText(
        "yellow",
        `\nWarning: found invalid date "${d}" in \`${fp}\`. Supported formats: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date#date_time_string_format`,
      ),
    )
  }

  return invalidDate ? new Date() : dt
}

type MaybeDate = undefined | string | number
export const CreatedModifiedDate: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "CreatedModifiedDate",
    markdownPlugins(ctx) {
      return [
        () => {
          let repo: Repository | undefined = undefined
          return async (_, file) => {
            let created: MaybeDate = undefined
            let modified: MaybeDate = undefined
            let published: MaybeDate = undefined

            const fp = file.data.relativePath!
            const fullFp = path.posix.join(ctx.argv.directory, fp)
            for (const source of opts.priority) {
              switch (source) {
                case "filesystem": {
                  const st = await fs.stat(fullFp)
                  created ||= st.birthtimeMs
                  modified ||= st.mtimeMs
                  break
                }
                case "frontmatter": {
                  if (file.data.frontmatter) {
                    created ||= file.data.frontmatter.created as MaybeDate
                    modified ||= file.data.frontmatter.modified as MaybeDate
                    published ||= file.data.frontmatter.published as MaybeDate
                  }
                  break
                }
                case "git": {
                  if (!repo) {
                    // Get a reference to the main git repo.
                    // It's either the same as the workdir,
                    // or 1+ level higher in case of a submodule/subtree setup
                    repo = Repository.discover(ctx.argv.directory)
                  }
                  try {
                    modified ||= await repo.getFileLatestModifiedDateAsync(fullFp)
                  } catch {
                    console.log(
                      styleText(
                        "yellow",
                        `\nWarning: ${file.data
                          .filePath!} isn't yet tracked by git, last modification date is not available for this file`,
                      ),
                    )
                  }
                  break
                }
              }
            }

            file.data.dates = {
              created: coerceDate(fp, created),
              modified: coerceDate(fp, modified),
              published: coerceDate(fp, published),
            }
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    dates: { created: Date; modified: Date; published: Date }
  }
}
