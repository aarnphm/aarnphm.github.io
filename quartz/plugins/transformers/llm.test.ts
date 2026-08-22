import type { Root } from 'mdast'
import assert from 'node:assert/strict'
import test from 'node:test'
import { VFile } from 'vfile'
import type { BuildCtx } from '../../util/ctx'
import type { FullSlug } from '../../util/path'
import { LLM } from './llm'

type MarkdownTransformer = (tree: Root, file: VFile) => void
type MarkdownPluginFactory = () => MarkdownTransformer

function watchTransformer(): MarkdownTransformer {
  const plugins = LLM().markdownPlugins?.({ argv: { watch: true, force: false } } as BuildCtx)
  const plugin = plugins?.[0]
  assert.equal(typeof plugin, 'function')
  return (plugin as MarkdownPluginFactory)()
}

const tree: Root = {
  type: 'root',
  children: [
    { type: 'heading', depth: 2, children: [{ type: 'text', value: 'agent navigation' }] },
    {
      type: 'paragraph',
      children: [
        { type: 'text', value: 'Use ' },
        { type: 'inlineCode', value: '/triathlon.md' },
        { type: 'text', value: ' as the route-family index.' },
      ],
    },
  ],
}

test('watch processing serializes the triathlon route index', () => {
  const file = new VFile()
  file.data.slug = 'triathlon' as FullSlug

  watchTransformer()(tree, file)

  const output = file.data.llmsText
  if (typeof output !== 'string') throw new Error('expected triathlon Markdown output')
  assert.match(output, /## agent navigation/)
  assert.match(output, /`\/triathlon\.md` as the route-family index/)
})

test('watch processing leaves unrelated notes out of the LLM corpus', () => {
  const file = new VFile()
  file.data.slug = 'thoughts/example' as FullSlug

  watchTransformer()(tree, file)

  assert.equal(file.data.llmsText, undefined)
})
