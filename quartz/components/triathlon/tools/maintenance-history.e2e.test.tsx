import type { Element, RootContent } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { toText } from 'hast-util-to-text'
import { load } from 'js-yaml'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import { parseTriathlonMaintenance } from '../../../util/triathlon-maintenance'
import { isRecord } from '../../../util/type-guards'
import { Maintenance } from './Maintenance'

const elements = (nodes: RootContent[]): Element[] =>
  nodes.flatMap(node => (node.type === 'element' ? [node, ...elements(node.children)] : []))

// Exercise authored YAML through parsing and HTML. Failure modes: null lubricant hides a
// chain, IDs collide across component groups, cycles disappear, or unresolved IDs link away.
test('authored chain history reaches the maintenance surface with working links and wax tags', () => {
  const source = readFileSync(
    new URL('./fixtures/maintenance-history.yaml', import.meta.url),
    'utf8',
  )
  const frontmatter: unknown = load(source)
  assert.ok(isRecord(frontmatter))
  const maintenance = parseTriathlonMaintenance(frontmatter.maintenance)
  assert.ok(maintenance)
  const html = renderToString(<Maintenance maintenance={maintenance} />)
  const nodes = elements(fromHtml(html, { fragment: true }).children)
  const chains = nodes.filter(node => node.properties.dataChainId !== undefined)
  assert.equal(chains.length, 5)
  const backup = chains.find(node => String(node.properties.dataChainId) === '4')
  const active = chains.find(node => String(node.properties.dataChainId) === '5')
  assert.ok(backup)
  assert.ok(active)
  assert.match(toText(backup), /backup/)
  assert.match(toText(active), /2026-11-12/)
  assert.match(toText(active), /Cyclowax Performance Tablet/)
  assert.match(toText(active), /waxed with cyclowax/)
  assert.match(toText(active), /Performance Wax Kit/)
  assert.ok(elements(backup.children).some(node => node.properties.dataMi === '1774.35'))
  assert.match(toText(backup), /2026-08-26 → 2026-11-11/)
  assert.ok(elements(active.children).some(node => node.properties.dataMi === '22.39'))
  const links = chains
    .flatMap(node => elements(node.children))
    .filter(node => node.tagName === 'a' && String(node.properties.href).startsWith('#'))
  assert.equal(links.length, 10)
  for (const link of links) {
    const target = nodes.filter(
      node => node.properties.id === String(link.properties.href).slice(1),
    )
    assert.equal(target.length, 1)
    assert.ok(toText(target[0]).includes(toText(link)))
  }
  const directory = 'docs/verification/triathlon-maintenance-2026-11-12/artifacts'
  mkdirSync(directory, { recursive: true })
  writeFileSync(`${directory}/fixture.html`, html)
  writeFileSync(`${directory}/fixture.json`, JSON.stringify(maintenance, null, 2) + '\n')
  const actual: unknown = load(readFileSync('content/triathlon.md', 'utf8').split('---')[1])
  assert.ok(isRecord(actual))
  const current = parseTriathlonMaintenance(actual.maintenance)
  assert.ok(current)
  writeFileSync(
    `${directory}/maintenance.html`,
    renderToString(<Maintenance maintenance={current} />),
  )
  writeFileSync(`${directory}/maintenance.json`, JSON.stringify(current, null, 2) + '\n')
})

test('unresolved and duplicate component IDs stay readable without broken links', () => {
  const maintenance = parseTriathlonMaintenance({
    chain: {
      spare: {
        since: '2026-11-12',
        distance: null,
        lubricant: null,
        waxed: true,
        chainring: 'missing',
        ospw: 'ambiguous',
        wax: 'Test wax',
        cycles: [{ end: '2026-11-12', reasoning: 'first wax' }],
      },
    },
    OSPW: ['A', 'B'].map(type => [
      { id: 'ambiguous', type, distance: null, range: [{ start: '2026-11-12', end: null }] },
    ]),
  })
  assert.ok(maintenance)
  const nodes = elements(
    fromHtml(renderToString(<Maintenance maintenance={maintenance} />), { fragment: true })
      .children,
  )
  assert.equal(nodes.filter(node => node.tagName === 'a').length, 0)
  const chain = nodes.find(node => node.properties.dataChainId === 'spare')
  assert.ok(chain)
  assert.match(toText(chain), /missing/)
  assert.match(toText(chain), /ambiguous/)
  assert.match(toText(chain), /first wax/)
})
