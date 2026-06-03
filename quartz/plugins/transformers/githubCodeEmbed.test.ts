import type { Element, Root } from 'hast'
import assert from 'node:assert'
import test from 'node:test'
import { GithubCodeEmbed } from './githubCodeEmbed'

type HtmlTransformer = (tree: Root) => void

function runGithubCodeEmbed(tree: Root) {
  const plugins = GithubCodeEmbed().htmlPlugins?.({} as never)
  assert.ok(plugins)
  const createTransformer = plugins[0] as () => HtmlTransformer
  createTransformer()(tree)
}

test('wraps highlighted GitHub code titles in source links', () => {
  const title: Element = {
    type: 'element',
    tagName: 'figcaption',
    properties: { dataRehypePrettyCodeTitle: '' },
    children: [{ type: 'text', value: 'aarnphm/garden · path.ts:10' }],
  }
  const tree: Root = {
    type: 'root',
    children: [
      {
        type: 'element',
        tagName: 'div',
        properties: {
          className: ['github-code-embed'],
          dataGithubHref: 'https://github.com/aarnphm/garden/blob/main/quartz/util/path.ts#L10',
        },
        children: [
          {
            type: 'element',
            tagName: 'figure',
            properties: { dataRehypePrettyCodeFigure: '' },
            children: [
              title,
              {
                type: 'element',
                tagName: 'pre',
                properties: {},
                children: [{ type: 'element', tagName: 'code', properties: {}, children: [] }],
              },
            ],
          },
        ],
      },
    ],
  }

  runGithubCodeEmbed(tree)

  const [link] = title.children
  assert.strictEqual(link.type, 'element')
  assert.strictEqual(link.tagName, 'a')
  assert.strictEqual(
    link.properties.href,
    'https://github.com/aarnphm/garden/blob/main/quartz/util/path.ts#L10',
  )
  assert.deepStrictEqual(link.properties.className, ['github-code-source'])
  assert.strictEqual(link.children[0].type, 'text')
  assert.strictEqual(link.children[0].value, 'aarnphm/garden · path.ts:10')
})
