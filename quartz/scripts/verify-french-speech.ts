import type { Element } from 'hast'
import { toHtml } from 'hast-util-to-html'
import assert from 'node:assert/strict'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { registerHooks } from 'node:module'
import path from 'node:path'
import { visit } from 'unist-util-visit'
import { VFile } from 'vfile'
import type { BuildCtx } from '../util/ctx'
import { stripSpeechMarkup } from '../extensions/micromark-extension-speech/source'
import { createMdProcessor, createHtmlProcessor, mdastToHastRoot } from '../processors/parse'
import { parseFlashcards } from '../util/flashcards'
import { isFilePath, isFullSlug } from '../util/path'

function argument(name: string, fallback: string): string {
  const index = process.argv.indexOf(name)
  if (index === -1) return fallback
  const value = process.argv[index + 1]
  if (!value || value.startsWith('--')) throw new Error(`${name} needs a path`)
  return value
}

const output = argument('--output', 'quartz/.quartz-cache/speech-evidence')
const baselinePath = argument('--baseline', 'docs/verification/french-speech/deck-identities.json')
mkdirSync(output, { recursive: true })
const artifact = (name: string) => path.join(output, name)
const command = `pnpm exec tsx quartz/scripts/verify-french-speech.ts --baseline ${baselinePath} --output ${output}`

const assets = registerHooks({
  resolve(specifier, context, next) {
    return next(specifier.endsWith('.inline') ? `${specifier}.ts` : specifier, context)
  },
  load(url, context, next) {
    if (/\.(?:scss|css|inline\.ts)$/.test(url)) {
      return {
        format: 'module',
        source: `export default ${JSON.stringify(readFileSync(new URL(url), 'utf8'))}`,
        shortCircuit: true,
      }
    }
    return next(url, context)
  },
})
const { Flashcards } = await import('../plugins/transformers/flashcards')
const { ObsidianFlavoredMarkdown } = await import('../plugins/transformers/ofm')
const { GitHubFlavoredMarkdown } = await import('../plugins/transformers/gfm')
const { Latex } = await import('../plugins/transformers/latex')
const { Sidenotes } = await import('../plugins/transformers/sidenotes')
const { Speech } = await import('../plugins/transformers/speech')
const { LLM } = await import('../plugins/transformers/llm')
const colors = {
  light: '#fff',
  lightgray: '#eee',
  gray: '#999',
  darkgray: '#555',
  dark: '#111',
  secondary: '#123',
  tertiary: '#456',
  highlight: '#def',
  textHighlight: '#fed',
}
const ctx: BuildCtx = {
  buildId: 'speech-parser-e2e',
  argv: {
    directory: 'content',
    output: 'public',
    verbose: false,
    serve: false,
    watch: false,
    port: 0,
    wsPort: 0,
    force: false,
  },
  cfg: {
    configuration: {
      pageTitle: 'speech fixture',
      enableSPA: false,
      enablePopovers: false,
      analytics: null,
      ignorePatterns: [],
      defaultDateType: 'created',
      locale: 'en-US',
      baseUrl: 'example.com',
      theme: {
        typography: { header: 'sans-serif', body: 'sans-serif', code: 'monospace' },
        cdnCaching: false,
        colors: { lightMode: colors, darkMode: colors },
        fontOrigin: 'local',
      },
    },
    plugins: {
      transformers: [
        Flashcards(),
        ObsidianFlavoredMarkdown(),
        GitHubFlavoredMarkdown(),
        Latex(),
        Sidenotes(),
        Speech({ modelBaseUrl: '/static/speech/model' }),
        LLM(),
      ],
      filters: [],
      emitters: [],
    },
  },
  allFiles: [],
  allSlugs: [],
  incremental: false,
}
const md = createMdProcessor(ctx)
const html = createHtmlProcessor(ctx)
const report: unknown[] = []

async function render(source: string, relativePath: string) {
  const slug = relativePath.replace(/\.md$/, '')
  const filePath = `content/${relativePath}`
  if (!isFullSlug(slug) || !isFilePath(filePath) || !isFilePath(relativePath))
    throw new Error('invalid fixture path')
  const file = new VFile({ path: filePath, value: source })
  file.data.slug = slug
  file.data.relativePath = relativePath
  file.data.filePath = filePath
  file.data.rawMarkdownSource = source
  const tree = await md.run(md.parse(source), file)
  const hast = await html.run(mdastToHastRoot(tree), file)
  const elements: Element[] = []
  visit(hast, 'element', node => {
    elements.push(node)
  })
  const output = toHtml(hast)
  return { elements, output, file }
}

const phraseSource = [
  '# Phrases',
  '',
  '{{Bonjour.}} {{`tu`}} {{**tout**}}',
  '',
  '{{`a }} b`}}',
  '',
  String.raw`\{{échappé}} $x={{y}}$` + ' `{{code}}`',
  '',
  '```text',
  '{{fenced}}',
  '```',
  '',
  '{{sidenotes[note]: Une remarque.}}',
  '',
  '{{sidenotes}} {{ }} {{inachevé',
  '{{Salut.}}',
  '',
  '[{{lien}}](https://example.com)',
  '',
  '<button>{{clic}}</button>',
].join('\n')
const phrase = await render(phraseSource, 'fr/speech-fixture.md')
const buttons = phrase.elements.filter(
  node =>
    node.tagName === 'button' &&
    Array.isArray(node.properties.className) &&
    node.properties.className.includes('speech-play'),
)
assert.equal(buttons.length, 5)
assert.ok(buttons.every(node => node.properties.disabled === true))
for (const button of buttons) {
  visit(button, 'element', node => assert.notEqual(node.tagName, 'svg'))
}
const playable = phrase.elements.filter(node => typeof node.properties.dataSpeechText === 'string')
assert.deepEqual(
  playable.map(node => node.properties.dataSpeechText),
  ['Bonjour.', 'tu', 'tout', 'a }} b', 'Salut.'],
)
assert.ok(playable.every(node => node.properties.dataSpeechModelBaseUrl === '/static/speech/model'))
assert.ok(phrase.output.includes('<code>tu</code>'))
assert.ok(phrase.output.includes('<strong>tout</strong>'))
assert.ok(phrase.output.includes('sidenote'))
const llmsText = phrase.file.data.llmsText
if (typeof llmsText !== 'string') throw new Error('The LLM transformer did not emit Markdown')
assert.ok(llmsText.includes('Bonjour.'))
assert.ok(!llmsText.includes('speech-play'))
writeFileSync(artifact('phrases.input.md'), phraseSource)
writeFileSync(artifact('phrases.output.html'), phrase.output)
writeFileSync(artifact('phrases.llm.md'), llmsText)
report.push({
  fixture: 'phrases',
  playable: playable.map(node => node.properties.dataSpeechText),
  buttons: buttons.length,
})

const surfacesSource =
  '> {{Bonjour.}}\n\n- {{Bonsoir.}}\n\n| French | English |\n| --- | --- |\n| {{Salut.}} | hello |\n\n{{[source](https://example.com)}}'
const surfaces = await render(surfacesSource, 'fr/speech-surfaces.md')
assert.deepEqual(
  surfaces.elements
    .filter(node => typeof node.properties.dataSpeechText === 'string')
    .map(node => node.properties.dataSpeechText),
  ['Bonjour.', 'Bonsoir.', 'Salut.'],
)
writeFileSync(artifact('surfaces.input.md'), surfacesSource)
writeFileSync(artifact('surfaces.output.html'), surfaces.output)
report.push({ fixture: 'blockquote-list-table', playable: 3 })

const plainQa =
  'Q: Say `bonjour`.\nA: `Je viens du Canada.`\n---\nQ: What is **tu**?\nA: informal you'
const speechQa =
  'Q: Say {{`bonjour`}}.\nA: {{`Je viens du Canada.`}}\n---\nQ: What is {{**tu**}}?\nA: informal you'
assert.deepEqual(
  parseFlashcards(speechQa).cards.map(card => card.id),
  parseFlashcards(plainQa).cards.map(card => card.id),
)
const qa = await render(speechQa, 'fr/speech-qa.fc')
assert.deepEqual(
  qa.elements
    .filter(node => typeof node.properties.dataSpeechText === 'string')
    .map(node => node.properties.dataSpeechText),
  ['bonjour', 'Je viens du Canada.', 'tu'],
)
writeFileSync(artifact('qa.input.fc'), speechQa)
writeFileSync(artifact('qa.output.html'), qa.output)
const plainCloze = 'C: `Je [viens] du [Canada].`\nN: Bonjour.'
const speechCloze = 'C: {{`Je [viens] du [Canada].`}}\nN: {{Bonjour.}}'
const identity = (source: string) =>
  parseFlashcards(source).cards.map(({ id, groupId }) => ({ id, groupId }))
assert.deepEqual(identity(speechCloze), identity(plainCloze))
const cloze = await render(speechCloze, 'fr/speech-fixture.fc')
const frontFaces = cloze.elements.filter(node => node.properties.dataFace === 'front')
assert.equal(frontFaces.length, 2)
for (const front of frontFaces) {
  const descendants: Element[] = []
  visit(front, 'element', node => {
    descendants.push(node)
  })
  assert.ok(!descendants.some(node => typeof node.properties.dataSpeechText === 'string'))
  assert.ok(
    !descendants.some(
      node =>
        Array.isArray(node.properties.className) &&
        node.properties.className.includes('speech-play'),
    ),
  )
}
const clozePlayable = cloze.elements.filter(
  node => typeof node.properties.dataSpeechText === 'string',
)
assert.deepEqual(
  clozePlayable.map(node => node.properties.dataSpeechText),
  ['Je viens du Canada.', 'Bonjour.', 'Je viens du Canada.', 'Bonjour.'],
)
writeFileSync(artifact('cloze.input.fc'), speechCloze)
writeFileSync(artifact('cloze.output.html'), cloze.output)
report.push({
  fixture: 'qa-and-cloze',
  qaIds: identity(speechQa),
  clozeIds: identity(speechCloze),
  clozePlayable: clozePlayable.map(node => node.properties.dataSpeechText),
  hiddenFronts: frontFaces.length,
})

const arbitraryClozeSource = '{{Avant <span class="cloze-blank">réponse cachée</span> après}}'
const arbitraryCloze = await render(arbitraryClozeSource, 'fr/speech-hidden.md')
assert.ok(!arbitraryCloze.elements.some(node => typeof node.properties.dataSpeechText === 'string'))
assert.ok(
  !arbitraryCloze.elements.some(
    node =>
      Array.isArray(node.properties.className) && node.properties.className.includes('speech-play'),
  ),
)
writeFileSync(artifact('hidden.input.md'), arbitraryClozeSource)
writeFileSync(artifact('hidden.output.html'), arbitraryCloze.output)
report.push({ fixture: 'arbitrary-hidden-cloze', playable: 0 })

const untouched = String.raw`\{{littéral}} $x={{y}}$ {{sidenotes[n]: texte}}` + ' `{{code}}`'
assert.equal(stripSpeechMarkup(untouched), untouched)
assert.equal(stripSpeechMarkup('{{`bonjour`}} {{**salut**}}'), '`bonjour` **salut**')

interface BaselineCard {
  id: string
  groupId?: string
  kind: string
}
interface BaselineDeck {
  path: string
  source?: string
  cards: BaselineCard[]
  errors?: unknown[]
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
function isBaselineCard(value: unknown): value is BaselineCard {
  return (
    isRecord(value) &&
    typeof value.id === 'string' &&
    typeof value.kind === 'string' &&
    (value.groupId === undefined || typeof value.groupId === 'string')
  )
}
function isBaselineDeck(value: unknown): value is BaselineDeck {
  return (
    isRecord(value) &&
    typeof value.path === 'string' &&
    (value.source === undefined || typeof value.source === 'string') &&
    Array.isArray(value.cards) &&
    value.cards.every(isBaselineCard) &&
    (value.errors === undefined || Array.isArray(value.errors))
  )
}
const baseline: unknown = JSON.parse(readFileSync(baselinePath, 'utf8'))
if (!isRecord(baseline) || !Array.isArray(baseline.decks) || !baseline.decks.every(isBaselineDeck))
  throw new Error(`Invalid deck baseline: ${baselinePath}`)
let corpusCards = 0
let currentCards = 0
let annotatedDecks = 0
let frenchCards = 0
let frenchDecks = 0
let clozeCards = 0
const clozeGroups = new Set<string>()
const deckResults: {
  path: string
  cards: number
  annotated: boolean
  idsPreserved: boolean
  sourcePreserved?: boolean
}[] = []
for (const deck of baseline.decks) {
  const shape = ({ id, groupId, kind }: { id: string; groupId?: string; kind: string }) => ({
    id,
    kind,
    ...(groupId ? { groupId } : {}),
  })
  if (deck.source !== undefined) {
    const original = parseFlashcards(deck.source)
    assert.deepEqual(original.cards.map(shape), deck.cards.map(shape), `original IDs: ${deck.path}`)
    assert.deepEqual(original.errors, deck.errors ?? [], `original parse errors: ${deck.path}`)
  }
  const currentSource = readFileSync(deck.path, 'utf8')
  const current = parseFlashcards(currentSource)
  assert.deepEqual(current.cards.map(shape), deck.cards.map(shape), `current IDs: ${deck.path}`)
  assert.deepEqual(current.errors, deck.errors ?? [], `current parse errors: ${deck.path}`)
  const stripped = stripSpeechMarkup(currentSource)
  if (deck.source !== undefined) {
    assert.equal(stripped, deck.source, `annotation-only source: ${deck.path}`)
  }
  corpusCards += deck.cards.length
  currentCards += current.cards.length
  if (currentSource !== stripped) annotatedDecks++
  if (deck.path.startsWith('content/fr/')) {
    frenchDecks++
    frenchCards += current.cards.length
  }
  for (const card of current.cards) {
    if (card.kind === 'cloze') clozeCards++
    if (card.groupId) clozeGroups.add(card.groupId)
  }
  deckResults.push({
    path: deck.path,
    cards: current.cards.length,
    annotated: currentSource !== stripped,
    idsPreserved: true,
    ...(deck.source !== undefined ? { sourcePreserved: true } : {}),
  })
}
report.push({
  fixture: 'corpus-identity',
  decks: baseline.decks.length,
  corpusCards,
  currentCards,
  annotatedDecks,
  frenchDecks,
  frenchCards,
  clozeCards,
  clozeGroups: clozeGroups.size,
})
writeFileSync(
  artifact('parser-report.json'),
  JSON.stringify({ command, baselinePath, passed: true, report, deckResults }, null, 2),
)
assets.deregister()
console.log(
  JSON.stringify({ passed: true, report, artifact: artifact('parser-report.json') }, null, 2),
)
