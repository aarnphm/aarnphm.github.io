import type { Element, Root } from 'hast'
import assert from 'node:assert/strict'
import test from 'node:test'
import { VFile } from 'vfile'
import type { BuildCtx } from '../../util/ctx'
import type { FullSlug } from '../../util/path'
import { Stream } from './stream'

type HtmlTransformer = (tree: Root, file: VFile) => void
type HtmlPluginFactory = () => HtmlTransformer

function text(value: string) {
  return { type: 'text' as const, value }
}

function el(tagName: string, properties: Element['properties'], children: Element['children']) {
  return { type: 'element' as const, tagName, properties, children }
}

function streamHtmlTransformer(): HtmlTransformer {
  const plugins = Stream().htmlPlugins?.({} as BuildCtx) ?? []
  const plugin = plugins[0]
  assert.equal(typeof plugin, 'function')
  return (plugin as HtmlPluginFactory)()
}

test('stream transformer keeps source heading ids on entry titles', () => {
  const tree: Root = {
    type: 'root',
    children: [el('h2', { id: 'intervals' }, [text('intervals')]), el('p', {}, [text('body')])],
  }
  const file = new VFile()
  file.data.slug = 'stream' as FullSlug

  streamHtmlTransformer()(tree, file)

  assert.equal(file.data.streamData?.entries[0]?.title, 'intervals')
  assert.equal(file.data.streamData?.entries[0]?.titleId, 'intervals')
})
