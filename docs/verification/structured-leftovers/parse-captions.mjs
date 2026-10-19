// Parses the JSX fences of the edited notes with remark-mdx, as the codeblock transformer does,
// and prints each figure's caption as the component will receive it.
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import remarkMdx from 'remark-mdx'
import remarkParse from 'remark-parse'
import { unified } from 'unified'

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')
const NOTES = ['content/thoughts/structured outputs.md', 'content/posts/structured decoding.md']
const NAMES = new Set(['TokenBoundaryShift', 'GrammarBitmaskBatch', 'LogitsProcessorTrace'])
const processor = unified().use(remarkParse).use(remarkMdx)

export const captions = () => {
  const out = []
  for (const note of NOTES) {
    const text = readFileSync(path.join(repo, note), 'utf8')
    for (const m of text.matchAll(/```jsx imports=\{[^}]*\}\n([\s\S]*?)```/g)) {
      const visit = n => {
        if (NAMES.has(n.name)) {
          const props = Object.fromEntries(
            n.attributes.map(a => [a.name, typeof a.value === 'string' ? a.value : a.value?.value]),
          )
          out.push({ note, component: n.name, props })
        }
        for (const c of n.children ?? []) visit(c)
      }
      visit(processor.parse(m[1]))
    }
  }
  return out
}

if (import.meta.url === `file://${process.argv[1]}`)
  for (const c of captions()) console.log(c.note, c.component, JSON.stringify(c.props))
