import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import { processHeaders } from './collapsible-headers'

for (const headingToggle of [false, true, undefined]) {
  test(`heading-toggle ${headingToggle} preserves sections and links`, () => {
    const nodes = [
      h('h2#chapter', ['Chapter', h('a', { href: '#chapter' }, '#')]),
      h('p', 'Chapter content'),
      h('h3#nested', 'Nested'),
      h('p', 'Nested content'),
      h('hr'),
      h('h2#last', 'Last'),
      h('p', 'Last content'),
    ]
    const html = toHtml({ type: 'root', children: processHeaders(nodes, headingToggle) })
    assert.equal((html.match(/<section/g) ?? []).length, 3)
    assert.match(html, /href="#chapter"/)
    for (const content of ['Chapter content', 'Nested content', 'Last content']) {
      assert.ok(html.includes(content))
    }
    if (headingToggle === false) {
      assert.doesNotMatch(html, /data-collapse-toggle|aria-expanded|role="button"|<button/)
      assert.equal((html.match(/collapse-shell is-open/g) ?? []).length, 3)
    } else {
      assert.equal((html.match(/data-collapse-toggle/g) ?? []).length, 3)
    }
  })
}

test('collapsible sections retain the canonical heading id once', () => {
  const [section] = processHeaders([
    h('h2#local-marker', ['Local marker', h('a', { href: '#local-marker' }, '#')]),
    h('p', 'Body'),
  ])
  assert.equal(section.type, 'element')
  if (section.type !== 'element') return

  const ids: string[] = []
  const collectIds = (node: typeof section): void => {
    if (typeof node.properties.id === 'string') ids.push(node.properties.id)
    for (const child of node.children) {
      if (child.type === 'element') collectIds(child)
    }
  }
  collectIds(section)
  const html = toHtml({ type: 'root', children: [section] })

  assert.equal(ids.filter(id => id === 'local-marker').length, 1)
  assert.match(html, /<h2 id="local-marker"/)
  assert.match(html, /href="#local-marker"/)
  assert.equal(section.properties.id, 'collapsible-header-local-marker')
  assert.equal(section.properties.dataHeadingAnchor, 'local-marker')
})
