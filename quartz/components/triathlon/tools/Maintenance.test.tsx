import type { Element, Root, RootContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import type { TriathlonMaintenance } from '../../../util/triathlon-maintenance'
import { Maintenance } from './Maintenance'

const maintenance: TriathlonMaintenance = {
  chains: [
    { id: '3', distance: null, lubricant: 'UFO Wax Drip-On', since: '2026-08-10', waxed: true },
  ],
  wheels: [
    {
      position: 'front',
      part: 'tire',
      type: 'Pirelli P Zero Race SL-R 700x28c',
      distance: null,
      start: '2026-08-12',
      end: null,
      reason: null,
    },
    {
      position: 'rear',
      part: 'tire',
      type: 'Pirelli P Zero Race TLR SL-R 700x28c',
      distance: '619.84 mile',
      start: '2026-07-16',
      end: '2026-08-10',
      reason: 'punctures and big ruptures',
    },
  ],
}

const elements = (root: Root, predicate: (element: Element) => boolean): Element[] => {
  const found: Element[] = []
  const visit = (nodes: RootContent[]): void => {
    for (const node of nodes) {
      if (node.type !== 'element') continue
      if (predicate(node)) found.push(node)
      visit(node.children)
    }
  }
  visit(root.children)
  return found
}

const hasClass = (element: Element, className: string): boolean => {
  const value = element.properties?.className
  return Array.isArray(value) && value.map(String).includes(className)
}

test('renders chain, current tire, and retired tire maintenance records', () => {
  const html = renderToString(<Maintenance maintenance={maintenance} />)
  const root = fromHtml(html, { fragment: true })
  assert.equal(elements(root, element => hasClass(element, 'tri-maintenance-entry')).length, 3)
  assert.match(html, /UFO Wax Drip-On/)
  assert.match(html, /Pirelli P Zero Race SL-R 700x28c/)
  assert.match(html, /2026-08-12 →/)
  assert.match(html, /punctures and big ruptures/)
})

test('renders no maintenance section without records', () => {
  assert.equal(renderToString(<Maintenance maintenance={null} />), '')
})
