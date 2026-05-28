import type { Element, Root } from 'hast'
import { h } from 'hastscript'
import assert from 'node:assert/strict'
import test from 'node:test'
import { normalizeCitationBibliography, resolveCitationLocale } from './citations'

test('uses bundled CSL locale when available', () => {
  assert.equal(resolveCitationLocale('fr-FR'), 'fr-FR')
})

test('falls back to canonical CSL locale URL for unbundled locales', () => {
  assert.equal(
    resolveCitationLocale('zz-ZZ'),
    'https://raw.githubusercontent.com/citation-style-language/locales/master/locales-zz-ZZ.xml',
  )
})

test('normalizes root-level CSL bibliography into Quartz references section', () => {
  const root: Root = {
    type: 'root',
    children: [
      h('p', [{ type: 'text', value: 'body' }]),
      h(
        'div.references.csl-bib-body',
        { id: 'refs', dir: 'auto' },
        h('div.csl-entry', { id: 'bib-paper', dir: 'auto' }, [
          { type: 'text', value: 'Paper. https://arxiv.org/abs/1234.5678' },
        ]),
      ),
    ],
  }

  normalizeCitationBibliography(root)

  const bibliography = root.children[1]
  assert.ok(bibliography?.type === 'element')
  assert.equal(bibliography.tagName, 'section')
  assert.equal(bibliography.properties.dataReferences, '')
  assert.deepEqual(bibliography.properties.className, ['bibliography'])

  const entries = bibliography.children.find(
    (child): child is Element => child.type === 'element' && child.tagName === 'ul',
  )
  assert.ok(entries)
  assert.equal(entries.children.length, 1)

  const entry = entries.children[0]
  assert.ok(entry?.type === 'element')
  assert.equal(entry.tagName, 'li')
  assert.equal(entry.properties.id, 'bib-paper')
})
