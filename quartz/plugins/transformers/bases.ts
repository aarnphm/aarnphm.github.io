import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import { parseFilter, parseViews, BaseFile, PropertyConfig } from "../../util/base/types"
import yaml from "js-yaml"

export interface Options {
  enableBases: boolean
}

const defaultOptions: Options = {
  enableBases: true,
}

export const ObsidianBases: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "ObsidianBases",
    markdownPlugins(_ctx) {
      if (!opts.enableBases) return []

      return [
        () => {
          return async (tree: Root, file) => {
            // Detect .base files by extension
            const isBaseFile = file.path?.endsWith(".base")

            if (!isBaseFile) {
              file.data.bases = false
              return
            }

            // Mark as bases file
            file.data.bases = true

            try {
              // Parse YAML and store config for emitter to use
              const parsed = yaml.load(String(file.value)) as any
              if (!parsed || typeof parsed !== "object") {
                throw new Error("Invalid .base file format")
              }

              const normalizeProperties = (
                raw: unknown,
              ): Record<string, PropertyConfig> | undefined => {
                if (!raw || typeof raw !== "object") {
                  return undefined
                }

                const normalized: Record<string, PropertyConfig> = {}

                for (const [key, value] of Object.entries(raw)) {
                  if (!value || typeof value !== "object") {
                    continue
                  }

                  const propConfig = value as PropertyConfig
                  normalized[key] = propConfig

                  const withoutPrefix = key.replace(/^(?:note|file)\./, "")
                  if (withoutPrefix !== key && !(withoutPrefix in normalized)) {
                    normalized[withoutPrefix] = propConfig
                  }

                  const segments = withoutPrefix.split(".")
                  const lastSegment = segments[segments.length - 1]
                  if (lastSegment && !(lastSegment in normalized)) {
                    normalized[lastSegment] = propConfig
                  }
                }

                return Object.keys(normalized).length > 0 ? normalized : undefined
              }

              const properties = normalizeProperties(parsed.properties)

              // Store config in file.data - emitter will do actual filtering
              const config: BaseFile = {
                filters: parseFilter(parsed.filters),
                views: parseViews(parsed.views),
                properties,
                summaries: parsed.summaries,
              }
              file.data.basesConfig = config

              // Create empty tree - emitter will build the actual table
              tree.children = []

              // Add bases tag to frontmatter
              if (!file.data.frontmatter) {
                file.data.frontmatter = {
                  title: file.path?.replace(".base", "").split("/").pop() || "",
                  pageLayout: "default" as const,
                }
              }
              const existingTags = (file.data.frontmatter.tags as string[]) || []
              file.data.frontmatter.tags = [...existingTags, "bases"]
            } catch (error) {
              console.error(`Error processing .base file ${file.path}:`, error)
              // Keep empty tree on error
              tree.children = []
            }
          }
        },
      ]
    },
  }
}
