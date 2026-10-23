import assert from 'node:assert/strict'
import test from 'node:test'
import {
  descriptionToPlainText,
  parseDescriptionManifest,
  processWikilinksToHtml,
  renderLatexInString,
  resolveManifestDescription,
} from './description'
import { renderDescription } from './og'
import { isFilePath, isFullSlug } from './path'

test('renderLatexInString renders inline latex for HTML descriptions', () => {
  const rendered = renderLatexInString('cost drops from $\\mathcal{O}(L^2)$')

  assert.equal(rendered.includes('class="katex"'), true)
  assert.equal(rendered.includes('\\mathcal{O}(L^2)'), true)
})

test('descriptionToPlainText keeps wikilink aliases as text', () => {
  assert.equal(descriptionToPlainText('see [[thoughts/Attention|Attention]]'), 'see Attention')
})

test('processWikilinksToHtml renders description wikilinks as links', () => {
  const currentSlug = 'thoughts/sliding-window'
  if (!isFullSlug(currentSlug)) assert.fail('invalid test slug')

  assert.equal(
    processWikilinksToHtml(
      'see [[thoughts/Attention|attention]] and [[#Local result]]',
      currentSlug,
    ),
    'see <a href="/thoughts/Attention" class="internal">attention</a> and <a href="/thoughts/sliding-window#local-result" class="internal">Local result</a>',
  )
})

test('renderDescription keeps OG image descriptions readable', () => {
  assert.equal(
    renderDescription(
      'cost drops from $\\mathcal{O}(L^2)$ near [[#Local result]]',
      'thoughts/sliding-window',
    ),
    'cost drops from O(L^2) near Local result',
  )
})

test('parseDescriptionManifest resolves wikilink keys for base, canvas, and view slugs without clobbering parent base', () => {
  const manifest = parseDescriptionManifest([
    { '[[antilibrary.base]]': 'personal reading catalogue across shelf and gallery views' },
    { '[[antilibrary.base#philosophy]]': 'philosophy shelf view' },
    { '[[places to go.base]]': 'geospatial map of cafes and bookstores' },
    { '[[singularity.canvas]]': 'systems map of [[thoughts/vllm|vllm]] inference' },
    { '[[fr/parcours.canvas]]': 'carte du parcours français' },
  ])

  const antilibrarySlug = 'antilibrary'
  const philosophyViewSlug = 'antilibrary/philosophy'
  const placesSlug = 'places-to-go'
  const placesPath = 'places to go.base'
  const singularitySlug = 'singularity'
  const parcoursSlug = 'fr/parcours'
  assert.ok(
    isFullSlug(antilibrarySlug) &&
      isFullSlug(philosophyViewSlug) &&
      isFullSlug(placesSlug) &&
      isFilePath(placesPath) &&
      isFullSlug(singularitySlug) &&
      isFullSlug(parcoursSlug),
  )

  assert.equal(
    resolveManifestDescription(manifest, antilibrarySlug),
    'personal reading catalogue across shelf and gallery views',
  )
  assert.equal(resolveManifestDescription(manifest, philosophyViewSlug), 'philosophy shelf view')
  assert.equal(
    resolveManifestDescription(manifest, placesSlug, placesPath),
    'geospatial map of cafes and bookstores',
  )
  assert.equal(
    resolveManifestDescription(manifest, singularitySlug),
    'systems map of [[thoughts/vllm|vllm]] inference',
  )
  assert.equal(resolveManifestDescription(manifest, parcoursSlug), 'carte du parcours français')
})
