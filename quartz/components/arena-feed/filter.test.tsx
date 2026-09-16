import type { Element } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toText } from 'hast-util-to-text'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import { visit } from 'unist-util-visit'
import { ReaderFilter } from './filter'

const filters = [
  {
    name: 'queue',
    label: 'show',
    options: [
      { value: 'unread', label: 'unread' },
      { value: 'read', label: 'read' },
      { value: 'all', label: 'all saved links' },
    ],
  },
  {
    name: 'notes',
    label: 'show notes',
    options: [
      { value: 'all', label: 'all notes' },
      { value: 'draft', label: 'drafts' },
      { value: 'ready', label: 'ready to backfill' },
      { value: 'backfilled', label: 'backfilled' },
    ],
  },
]

for (const filter of filters) {
  for (const { value, label } of filter.options) {
    test(`the ${filter.name} filter exposes ${label} as its selected option`, () => {
      const tree = fromHtml(
        renderToString(
          <ReaderFilter
            label={filter.label}
            options={filter.options}
            value={value}
            onChange={() => {}}
          />,
        ),
        { fragment: true },
      )
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
      const menuLabel = elements.find(
        element =>
          Array.isArray(menu.properties.ariaLabelledBy) &&
          typeof element.properties.id === 'string' &&
          menu.properties.ariaLabelledBy.includes(element.properties.id),
      )
      assert.ok(menuLabel)
      assert.equal(toText(menuLabel), filter.label)
      assert.equal(
        elements.some(element => element.tagName === 'select'),
        false,
      )
      const options = elements.filter(element => element.properties.role === 'option')
      assert.equal(options.length, filter.options.length)
      const selected = options.filter(option => option.properties.ariaSelected === 'true')
      assert.equal(selected.length, 1)
      assert.ok(toText(selected[0]).endsWith(label))
    })
  }
}
