import yaml from 'js-yaml'
import { version } from '../../../package.json'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { BuildCtx } from '../../util/ctx'
import { FullSlug } from '../../util/path'
import { QuartzPluginData } from '../vfile'
import { write } from './helpers'

const name = 'LLMText'
const watchMarkdownSlugs = new Set(['triathlon'])

function canEmitMarkdown(fileData: QuartzPluginData): boolean {
  return !fileData.flashcards && fileData.frontmatter?.protected !== true
}

function canEmitWatchMarkdown(fileData: QuartzPluginData): boolean {
  return watchMarkdownSlugs.has(fileData.slug ?? '')
}

export function llmsIndex(baseUrl: string, content: string = ''): string {
  const origin = baseUrl.startsWith('http') ? baseUrl.replace(/\/$/, '') : `https://${baseUrl}`
  return `# aarnphm.xyz

> Aaron Pham's public digital garden, with source-backed notes on software, machine learning, systems, mathematics, and training.

${content ? `${content}\n` : ''}---

Use this site when a task needs Aaron's published notes, an exact Markdown page, or source-backed search across the garden. Use the MCP server for semantic search and retrieval. Use the Markdown alternate for one known page. Cite the canonical HTML URL when presenting material to a reader.

## When to use this site

- [Search and retrieve garden notes](${origin}/api/docs): Use the read-only MCP tools when a task needs relevant notes or full Markdown without knowing a page URL.
- [Read the homepage as Markdown](${origin}/llms.txt): Send \`Accept: text/markdown\` to a canonical page URL when a task already knows which page it needs.
- [Enumerate public pages](${origin}/sitemap.xml): Use the XML sitemap when a task needs crawlable canonical URLs and last-modified dates.

## Agent interfaces

- [MCP API guide](${origin}/api/docs): Authentication, onboarding, tools, transport, errors, and rate limits.
- [OpenAPI description](${origin}/openapi.json): Typed HTTP operations and response schemas for function calling.
- [API catalog](${origin}/.well-known/api-catalog): RFC 9727 API discovery document.
- [AI capability catalog](${origin}/.well-known/ai-catalog.json): Machine-readable capability records for the MCP server, OpenAPI description, and agent skills.
- [MCP server card](${origin}/.well-known/mcp/server-card.json): MCP server identity, capabilities, and authorization metadata.
- [Full garden corpus](${origin}/llms-full.txt): All public notes serialized as delimited Markdown documents.

## Trust and policies

- [About Aaron and the garden](${origin}/about.md): Site identity, scope, authorship, and editorial context.
- [Contact Aaron](${origin}/contact.md): Contact routes and the information needed for useful requests.
- [Privacy policy](${origin}/privacy-policy.md): Data collection, storage, retention, deletion, and privacy contact details.
- [Security policy](${origin}/security-policy.md): Vulnerability reporting scope and responsible disclosure process.
`
}

async function llmText(ctx: BuildCtx, fileData: QuartzPluginData, reconstructed: string[]) {
  const slug = fileData.slug!
  const baseUrl = ctx.cfg.configuration.baseUrl ?? 'https://example.com'
  const contentBase = fileData.llmsText as string | undefined

  const refs = slug !== 'index' ? `${slug}.md` : 'llms.txt'

  const {
    claude: _,
    codex: __,
    gemini: ___,
    ...baseFrontmatter
  } = fileData.frontmatter as Record<string, unknown>

  const reconstructedFrontmatter = {
    ...baseFrontmatter,
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

  if (slug === 'index') {
    return write({
      ctx,
      content: llmsIndex(baseUrl, content),
      slug: 'llms' as FullSlug,
      ext: '.txt',
    })
  }

  const mdFrontmatter = {
    ...baseFrontmatter,
    slug,
    permalink: `https://${baseUrl}/${slug}.md`,
    generator: { quartz: `v${version}`, hostedProvider: 'Cloudflare', baseUrl },
    full: `https://${baseUrl}/llms-full.txt`,
  }

  return write({
    ctx,
    content: `---
${yaml.dump(mdFrontmatter, { lineWidth: -1, noRefs: true })}---
${contentBase}
`,
    slug,
    ext: '.md',
  })
}

export const LLMText: QuartzEmitterPlugin = () => {
  return {
    name,
    async *emit(ctx, content, _resources) {
      const baseUrl = ctx.cfg.configuration.baseUrl ?? 'https://example.com'
      const watch = ctx.argv.watch && !ctx.argv.force
      if (watch) {
        yield write({ ctx, content: llmsIndex(baseUrl), slug: 'llms' as FullSlug, ext: '.txt' })
      }

      const reconstructed: string[] = []
      for (const [, file] of content) {
        if (!canEmitMarkdown(file.data)) continue
        if (watch && !canEmitWatchMarkdown(file.data)) continue
        yield llmText(ctx, file.data, reconstructed)
      }

      if (watch) return

      yield write({
        ctx,
        content: `<system_prompt>
Instructions for using https://${baseUrl}/llms-full.txt:
- All notes gathered from Aaron's garden
- Every files are encapsulated between <document slug="..."></document>
- If you see a "<ref slug=xxx>", then make sure to search for <document slug=xxx>.
- If the representation here is lacking, you can still access the ref source via https://${baseUrl}/<slug>.md for the full markdown format.
</system_prompt>
${reconstructed.join('\n')}`,
        slug: 'llms-full' as FullSlug,
        ext: '.txt',
      })
    },
    async *partialEmit(ctx, content, _resources, changeEvents) {
      const watch = ctx.argv.watch && !ctx.argv.force

      // find all slugs that changed or were added
      const changedSlugs = new Set<string>()
      for (const changeEvent of changeEvents) {
        // If it's a markdown file change, add its own slug
        if (changeEvent.file) {
          if (changeEvent.type === 'add' || changeEvent.type === 'change') {
            changedSlugs.add(changeEvent.file.data.slug!)
          }
          continue
        }
        // Non-markdown file changed: re-emit any page that depends on it
        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
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
        if (!canEmitMarkdown(file.data)) continue
        if (watch && !canEmitWatchMarkdown(file.data)) continue

        yield llmText(ctx, file.data, [])
      }
    },
  }
}
