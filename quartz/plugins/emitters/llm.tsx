import { version } from "../../../package.json"
import { BuildCtx } from "../../util/ctx"
import { QuartzPluginData } from "../vfile"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { FullSlug } from "../../util/path"
import { write } from "./helpers"
import yaml from "js-yaml"

const name = "LLMText"

async function llmText(ctx: BuildCtx, fileData: QuartzPluginData, reconstructed: string[]) {
  const slug = fileData.slug!
  const baseUrl = ctx.cfg.configuration.baseUrl ?? "https://example.com"
  const contentBase = fileData.llmsText as string | undefined

  if (typeof contentBase !== "string") {
    throw new Error(`llmsText content missing for slug ${slug}`)
  }

  const refs = slug !== "index" ? `${slug}.md` : "llms.txt"

  const reconstructedFrontmatter = {
    ...fileData.frontmatter,
    reconstructured: true,
    permalink: `https://${baseUrl}/${refs}`,
  }

  const content = `---
${yaml.dump(reconstructedFrontmatter, { lineWidth: -1, noRefs: true })}
---
${contentBase}`

  reconstructed.push(`<document slug=${slug}>
${content}
</document>`)

  if (slug === "index") {
    return write({
      ctx,
      content,
      slug: "llms" as FullSlug,
      ext: ".txt",
    })
  }

  const mdFrontmatter = {
    ...fileData.frontmatter,
    slug,
    permalink: `https://${baseUrl}/${slug}.md`,
    generator: {
      quartz: `v${version}`,
      hostedProvider: "Cloudflare",
      baseUrl,
    },
    full: `https://${baseUrl}/llms-full.txt`,
  }

  return write({
    ctx,
    content: `<agent>
- If you are Claude or Codex, make sure to check instruction with your respective keys (i.e "claude:", "codex:") within frontmatter for additional instructions when consuming "https://${baseUrl}/${slug}.md"
- The content can be accessed within <document></document> tag.
</agent>
<document>
---
${yaml.dump(mdFrontmatter, { lineWidth: -1, noRefs: true })}---
${contentBase}
</document>`,
    slug,
    ext: ".md",
  })
}

export const LLMText: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, content, _resources) {
      if (ctx.argv.watch && !ctx.argv.force) return []

      const baseUrl = ctx.cfg.configuration.baseUrl ?? "https://example.com"

      const reconstructed: string[] = []
      for (const [, file] of content) {
        // Skip protected notes
        if (file.data.frontmatter?.protected === true) continue
        yield llmText(ctx, file.data, reconstructed)
      }

      yield write({
        ctx,
        content: `<system_prompt>
A few instructions for using https://${baseUrl}/llms-full.txt:
- It will include all notes from Aaron.
- Every files are encapsulated between <document slug=...></document>
- If you encounter a "<ref slug=xxx>", then make sure to search for <document slug=xxx>.
- If the representation here is lacking, you can still access the ref source via https://${baseUrl}/<slug>.md for the full markdown format.
</system_prompt>
${reconstructed.join("\n")}`,
        slug: "llms-full" as FullSlug,
        ext: ".txt",
      })
    },
    async *partialEmit(ctx, content, _resources, changeEvents) {
      if (ctx.argv.watch && !ctx.argv.force) return []

      // find all slugs that changed or were added
      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        // If it's a markdown file change, add its own slug
        if (changeEvent.file) {
          if (changeEvent.type === "add" || changeEvent.type === "change") {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
          continue
        }
        // Non-markdown file changed: re-emit any page that depends on it
        if (changeEvent.type === "add" || changeEvent.type === "change") {
          const changedPath = changeEvent.path
          for (const [_, vf] of content) {
            const deps = (vf.data.codeDependencies as string[] | undefined) ?? []
            if (deps.includes(changedPath)) {
              changedSlugs.add(vf.data.slug!)
            }
          }
        }
      }

      for (const [, file] of content) {
        const slug = file.data.slug!
        if (!changedSlugs.has(slug)) continue
        // Skip protected notes
        if (file.data.frontmatter?.protected === true) continue

        yield llmText(ctx, file.data, [])
      }
    },
  }
}
