import { QuartzTransformerPlugin } from "../types"
import { Root } from "mdast"
import {
  parseFilter,
  parseViews,
  parseFormulas,
  BaseFile,
  PropertyConfig,
} from "../../util/base/types"
import yaml from "js-yaml"

export const ObsidianBases: QuartzTransformerPlugin = () => {
  return {
    name: "ObsidianBases",
    markdownPlugins() {
      return [
        () => {
          return async (tree: Root, file) => {
            // Detect .base files by extension
            const isBaseFile = file.path?.endsWith(".base")

            if (!isBaseFile) {
              file.data.bases = false
              return
            }

            file.data.bases = true

            // Parse YAML and store config for emitter to use
            const parsed = yaml.load(String(file.value)) as any

            const normalizeProperties = (
              raw: object,
            ): Record<string, PropertyConfig> | undefined => {
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

            const config: BaseFile = {
              filters: parseFilter(parsed.filters),
              views: parseViews(parsed.views),
              properties,
              summaries: parsed.summaries,
              formulas: parseFormulas(parsed.formulas),
            }
            file.data.basesConfig = config

            tree.children = []

            file.data.frontmatter = {
              title: file.path?.replace(".base", "").split("/").pop() || "",
              pageLayout: "default" as const,
              description: `bases renderer of ${file.data.slug}`,
              tags: ["bases"],
            }
          }
        },
      ]
    },
  }
}

declare module "vfile" {
  interface DataMap {
    bases?: boolean
    basesConfig?: BaseFile
  }
}
