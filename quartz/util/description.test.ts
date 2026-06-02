import assert from 'node:assert/strict'
import test from 'node:test'
import { descriptionToPlainText, processWikilinksToHtml, renderLatexInString } from './description'
import { renderDescription } from './og'
import { isFullSlug } from './path'

test('renderLatexInString renders inline latex for HTML descriptions', () => {
  const rendered = renderLatexInString('cost drops from $\\mathcal{O}(L^2)$')

  assert.equal(rendered.includes('class="katex"'), true)
  assert.equal(rendered.includes('\\mathcal{O}(L^2)'), true)
})

test('descriptionToPlainText keeps latex readable for OG descriptions', () => {
  const description =
    'each token attends only inside a fixed radius, local pattern dropping cost from $\\mathcal{O}(L^2)$ to $\\mathcal{O}(Lw)$.'

  assert.equal(
    descriptionToPlainText(description),
    'each token attends only inside a fixed radius, local pattern dropping cost from O(L^2) to O(Lw).',
  )
})

test('descriptionToPlainText keeps wikilink aliases as text', () => {
  assert.equal(descriptionToPlainText('see [[thoughts/Attention|main stage]]'), 'see main stage')
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
