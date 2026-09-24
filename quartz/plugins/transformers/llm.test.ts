import type { Root } from 'mdast'
import assert from 'node:assert/strict'
import test from 'node:test'
import { VFile } from 'vfile'
import type { BuildCtx } from '../../util/ctx'
import { isFullSlug, type FullSlug } from '../../util/path'
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

test('watch processing serializes Arena channels and their links', () => {
  const file = new VFile()
  const slug = 'are.na'
  assert.ok(isFullSlug(slug))
  file.data.slug = slug
  const arenaTree: Root = {
    type: 'root',
    children: [
      { type: 'heading', depth: 2, children: [{ type: 'text', value: 'engineering' }] },
      {
        type: 'list',
        ordered: false,
        children: [
          {
            type: 'listItem',
            children: [
              {
                type: 'paragraph',
                children: [
                  {
                    type: 'link',
                    url: 'https://example.com',
                    children: [{ type: 'text', value: 'Example' }],
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  }

  watchTransformer()(arenaTree, file)

  const output = file.data.llmsText
  assert.ok(typeof output === 'string')
  assert.match(output, /## engineering/)
  assert.match(output, /- \[Example\]\(https:\/\/example.com\)/)
})
