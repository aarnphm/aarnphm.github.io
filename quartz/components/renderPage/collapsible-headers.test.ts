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
