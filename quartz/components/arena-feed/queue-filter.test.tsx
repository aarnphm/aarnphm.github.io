import type { Element } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toText } from 'hast-util-to-text'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import { visit } from 'unist-util-visit'
import type { FeedFilter } from './model'
import { QueueFilter } from './queue-filter'

const cases: { value: FeedFilter; label: string }[] = [
  { value: 'unread', label: 'unread' },
  { value: 'read', label: 'read' },
  { value: 'all', label: 'all saved links' },
]

for (const { value, label } of cases) {
  test(`the queue filter exposes ${label} as its selected option`, () => {
    const tree = fromHtml(renderToString(<QueueFilter value={value} onChange={() => {}} />), {
      fragment: true,
    })
    const elements: Element[] = []
    visit(tree, 'element', element => {
      elements.push(element)
    })
    const trigger = elements.find(element => element.properties.ariaHasPopup === 'listbox')
    const menu = elements.find(element => element.properties.role === 'listbox')
    assert.ok(trigger)
    assert.ok(menu)
    assert.equal(trigger.properties.ariaExpanded, 'false')
    assert.deepEqual(trigger.properties.ariaControls, [menu.properties.id])
    assert.equal(toText(trigger), label)
    assert.equal(menu.properties.hidden, true)
    const options = elements.filter(element => element.properties.role === 'option')
    assert.equal(options.length, 3)
    const selected = options.filter(option => option.properties.ariaSelected === 'true')
    assert.equal(selected.length, 1)
    assert.ok(toText(selected[0]).endsWith(label))
  })
}
