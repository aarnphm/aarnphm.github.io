import type { Element, Root, RootContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import type { TriathlonMaintenance } from '../../../util/triathlon-maintenance'
import { Maintenance } from './Maintenance'

const maintenance: TriathlonMaintenance = {
  services: [
    { bike: 'soloist', date: '2026-08-20', distance: '1721.5 mile', place: 'Racer Sportif' },
  ],
  components: [
    {
      component: 'OSPW',
      type: 'CeramicSpeed OSPW RS 5 Spoke',
      distance: null,
      ranges: [{ start: '2026-08-10', end: null }],
      reason: null,
    },
    {
      component: 'OSPW',
      type: 'Ultegra R8100 Pulley Wheel',
      distance: null,
      ranges: [{ start: '2026-05-16', end: '2026-08-10' }],
      reason: 'upgraded to CeramicSpeed OSPW',
    },
    {
      component: 'bottom bracket',
      type: 'FSA T47 BBright',
      distance: '1721.5 mile',
      ranges: [{ start: '2026-05-16', end: '2026-08-20' }],
      reason: 'upgraded to CeramicSpeed',
    },
  ],
  chains: [
    { id: '3', distance: null, lubricant: 'UFO Wax Drip-On', since: '2026-08-10', waxed: true },
  ],
  wheels: [
    {
      position: 'front',
      part: 'tire',
      type: 'Pirelli P Zero Race SL-R 700x28c',
      distance: null,
      ranges: [
        { start: '2026-07-16', end: '2026-08-10' },
        { start: '2026-08-18', end: null },
      ],
      reason: 'punctures, repaired',
      repaired: true,
    },
    {
      position: 'rear',
      part: 'tire',
      type: 'Pirelli P Zero Race TLR SL-R 700x28c',
      distance: '619.84 mile',
      ranges: [{ start: '2026-07-16', end: '2026-08-10' }],
      reason: 'punctures and big ruptures',
      repaired: false,
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

test('renders service, component, chain, and wheel maintenance records', () => {
  const html = renderToString(<Maintenance maintenance={maintenance} />)
  const root = fromHtml(html, { fragment: true })
  assert.equal(elements(root, element => hasClass(element, 'tri-maintenance-entry')).length, 7)
  assert.match(html, /Racer Sportif/)
  assert.match(html, /CeramicSpeed OSPW RS 5 Spoke/)
  assert.match(html, /FSA T47 BBright/)
  assert.match(html, /data-i18n="components">components/)
  assert.match(html, /data-i18n="bottom bracket">bottom bracket/)
  assert.match(html, /UFO Wax Drip-On/)
  assert.match(html, /Pirelli P Zero Race SL-R 700x28c/)
  assert.match(html, /2026-07-16 → 2026-08-10/)
  assert.match(html, /2026-08-18 →/)
  assert.match(html, /data-i18n="repaired">repaired/)
  assert.match(html, /data-i18n="yes">yes/)
  assert.match(html, /data-i18n="no">no/)
  assert.match(html, /punctures and big ruptures/)
})

test('renders no maintenance section without records', () => {
  assert.equal(renderToString(<Maintenance maintenance={null} />), '')
})
