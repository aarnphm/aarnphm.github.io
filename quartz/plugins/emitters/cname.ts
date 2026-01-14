import { FilePath, joinSegments } from "../../util/path"
import { QuartzEmitterPlugin } from "../../types/plugin"
import fs from "node:fs/promises"
import { styleText } from "node:util"

export function extractDomainFromBaseUrl(baseUrl: string) {
  const url = new URL(`https://${baseUrl}`)
  return url.hostname
}

const name = "CNAME"
export const CNAME: QuartzEmitterPlugin = () => ({
  name,
  async emit({ argv, cfg }) {
    if (!cfg.configuration.baseUrl) {
      console.warn(
        styleText("yellow", `[emit:${name}] requires \`baseUrl\` to be set in your configuration`),
      )
      return []
    }
    const path = joinSegments(argv.output, "CNAME")
    const content = extractDomainFromBaseUrl(cfg.configuration.baseUrl)
    if (!content) return []
    await fs.writeFile(path, content)
    return [path] as FilePath[]
  },
  async *partialEmit() {},
})
