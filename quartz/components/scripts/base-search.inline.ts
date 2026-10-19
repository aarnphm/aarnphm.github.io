import { isRecord } from '../../util/type-guards'
import { SemanticClient } from './semantic.inline'

type SearchMode = 'lexical' | 'semantic'
type SemanticState = 'idle' | 'loading' | 'ready' | 'unavailable'

const SEMANTIC_K = 24
// Scores far under the best match are noise for a short query; the ranking keeps its head only.
const SEMANTIC_TAIL = 0.12
const DEBOUNCE_MS = 180
const MODE_KEY = 'base-search-mode'
const SEMANTIC_IDLE_MS = 30_000

type SharedSemantic = { manifest: string; client: SemanticClient; ids: Promise<string[]> }
let shared: SharedSemantic | null = null
let releaseTimer: ReturnType<typeof setTimeout> | undefined
const manifestIdsCache = new Map<string, Promise<string[]>>()

function releaseSemantic() {
  if (!shared || releaseTimer !== undefined) return
  const current = shared
  releaseTimer = setTimeout(() => {
    releaseTimer = undefined
    if (shared !== current) return
    shared = null
    current.client.dispose()
  }, SEMANTIC_IDLE_MS)
}

function manifestIds(manifest: string): Promise<string[]> {
  let cached = manifestIdsCache.get(manifest)
  if (!cached) {
    cached = (async () => {
      const res = await fetch(manifest)
      if (!res.ok) throw new Error(`failed to load ${manifest}: ${res.status}`)
      const data: unknown = await res.json()
      if (
        !isRecord(data) ||
        !Array.isArray(data.ids) ||
        !data.ids.every(id => typeof id === 'string')
      )
        throw new Error('The semantic index has no valid note inventory.')
      return data.ids
    })().catch(err => {
      manifestIdsCache.delete(manifest)
      throw err
    })
    manifestIdsCache.set(manifest, cached)
  }
  return cached
}

function sharedSemantic(manifest: string): SharedSemantic {
  clearTimeout(releaseTimer)
  releaseTimer = undefined
  if (shared?.manifest === manifest) return shared
  const previous = shared
  const client = previous?.client ?? new SemanticClient({ ...semanticCfg, manifestUrl: manifest })
  const swap = previous ? previous.ids.then(() => client.load(manifest)) : client.ensureReady()
  const ids = swap.then(() => manifestIds(manifest))
  const next: SharedSemantic = { manifest, client, ids }
  ids.catch(() => {
    client.dispose()
    if (shared === next) shared = null
  })
  shared = next
  return next
}

function loadStoredMode(): SearchMode {
  try {
    return localStorage.getItem(MODE_KEY) === 'semantic' ? 'semantic' : 'lexical'
  } catch {
    return 'lexical'
  }
}

function storeMode(mode: SearchMode) {
  try {
    localStorage.setItem(MODE_KEY, mode)
  } catch {}
}

function setupBaseSearch(): (() => void) | undefined {
  const searchInput = document.querySelector<HTMLInputElement>('#base-search-input')
  const bar = document.querySelector<HTMLElement>('#base-search-bar')
  const root = document.querySelector<HTMLElement>('.main-col > .base-view')
  if (!searchInput || !bar || !root) return
  let active = true
  let semanticNotice = ''

  const items: HTMLElement[] = []
  for (const row of root.querySelectorAll<HTMLTableRowElement>('table.base-table tbody tr')) {
    if (!row.classList.contains('base-group-header')) items.push(row)
  }
  for (const li of root.querySelectorAll<HTMLLIElement>('.base-list > li')) items.push(li)
  for (const card of root.querySelectorAll<HTMLElement>('.base-card')) items.push(card)

  const listGroups = root.querySelectorAll<HTMLElement>('.base-list-group, .base-calendar-group')
  const cardGroups = root.querySelectorAll<HTMLElement>('.base-card-group')
  const tableGroupHeaders = root.querySelectorAll<HTMLTableRowElement>('tr.base-group-header')
  const kanbanColumns = root.querySelectorAll<HTMLElement>('.base-kanban-column')

  const slugOf = (item: HTMLElement) =>
    item.dataset.baseSlug ?? item.querySelector<HTMLElement>('a[data-slug]')?.dataset.slug
  const isShown = (item: HTMLElement) => item.style.display !== 'none'
  // Only sibling lists without header rows can be reordered; grouped tables keep their rows in place.
  const reorderable = (item: HTMLElement) => !(item instanceof HTMLTableRowElement)
  const originalIndex = new Map<HTMLElement, number>()
  items.forEach((item, index) => originalIndex.set(item, index))
  const scoreLabels = new Map<HTMLElement, { label: HTMLElement; title: HTMLElement }>()
  const lexicalText = new WeakMap<HTMLElement, string>()

  const clearScore = (item: HTMLElement) => {
    delete item.dataset.semanticScore
    const score = scoreLabels.get(item)
    if (!score) return
    score.label.remove()
    score.title.replaceWith(...score.title.childNodes)
    scoreLabels.delete(item)
  }

  const showScore = (item: HTMLElement, score: number) => {
    const link =
      item.querySelector<HTMLElement>('.base-card-title-link') ??
      item.querySelector<HTMLElement>('a[data-slug]')
    if (!link) return
    let elements = scoreLabels.get(item)
    if (!elements) {
      const title = document.createElement('div')
      title.className = 'base-semantic-title'
      const label = document.createElement('span')
      label.className = 'base-semantic-score'
      link.before(title)
      title.append(link, label)
      elements = { label, title }
      scoreLabels.set(item, elements)
    }
    const percentage = Math.round(Math.max(-1, Math.min(1, score)) * 100)
    elements.label.textContent = `${percentage}% similarity`
    elements.label.title = `Cosine similarity to your query: ${score.toFixed(3)}`
  }
  // Grouped views (by year, by category) keep their groups; semantic mode lifts the group that
  // holds the best hit, otherwise the top result can sit several headers down the page.
  const groups: HTMLElement[] = [...listGroups, ...cardGroups]
  groups.forEach((group, index) => originalIndex.set(group, index))

  const collapseGroups = () => {
    for (const group of listGroups) {
      group.hidden = !Array.from(group.querySelectorAll<HTMLElement>('.base-list > li')).some(
        isShown,
      )
    }
    for (const group of cardGroups) {
      group.hidden = !Array.from(group.querySelectorAll<HTMLElement>('.base-card')).some(isShown)
    }
    for (const header of tableGroupHeaders) {
      let visible = false
      for (
        let row = header.nextElementSibling;
        row instanceof HTMLTableRowElement && !row.classList.contains('base-group-header');
        row = row.nextElementSibling
      ) {
        if (isShown(row)) {
          visible = true
          break
        }
      }
      header.style.display = visible ? '' : 'none'
    }
    for (const column of kanbanColumns) {
      const count = Array.from(column.querySelectorAll<HTMLElement>('.base-card')).filter(
        isShown,
      ).length
      const label = column.querySelector<HTMLElement>('.base-kanban-count')
      if (label) {
        label.textContent = String(count)
        label.setAttribute('aria-label', `${count} results`)
      }
      const empty = column.querySelector<HTMLElement>('.base-kanban-column-empty')
      if (empty) empty.hidden = count > 0
    }
    if (root.dataset.baseViewType !== 'map') {
      const uniqueSlugs = (nodes: HTMLElement[]) =>
        new Set(nodes.map(item => slugOf(item) ?? item)).size
      document.dispatchEvent(
        new CustomEvent('base-search-results', {
          detail: {
            visible: uniqueSlugs(items.filter(isShown)),
            total: uniqueSlugs(items),
            notice: semanticNotice,
          },
        }),
      )
    }
  }

  const bySibling = (nodes: Iterable<HTMLElement>) => {
    const byParent = new Map<HTMLElement, HTMLElement[]>()
    for (const node of nodes) {
      if (!node.parentElement) continue
      const siblings = byParent.get(node.parentElement) ?? []
      siblings.push(node)
      byParent.set(node.parentElement, siblings)
    }
    return byParent
  }

  const restoreOrder = () => {
    for (const [parent, siblings] of bySibling([...items.filter(reorderable), ...groups])) {
      siblings.sort((a, b) => originalIndex.get(a)! - originalIndex.get(b)!)
      parent.append(...siblings)
    }
  }

  const showAll = () => {
    for (const item of items) {
      item.style.display = ''
      clearScore(item)
    }
    restoreOrder()
    collapseGroups()
  }

  const filterLexical = () => {
    const query = searchInput.value.toLowerCase().trim()
    if (query.length === 0) {
      showAll()
      return
    }
    for (const item of items) {
      clearScore(item)
      let text = lexicalText.get(item)
      if (text === undefined) {
        text = item.textContent?.toLowerCase() || ''
        lexicalText.set(item, text)
      }
      item.style.display = text.includes(query) ? '' : 'none'
    }
    restoreOrder()
    collapseGroups()
  }

  const applySemantic = (scores: Map<string, number>) => {
    const ranked: HTMLElement[] = []
    const groupBest = new Map<HTMLElement, number>()
    for (const item of items) {
      const slug = slugOf(item)
      const score = slug ? scores.get(slug) : undefined
      item.style.display = score === undefined ? 'none' : ''
      if (score === undefined) {
        clearScore(item)
        continue
      }
      item.dataset.semanticScore = score.toFixed(3)
      showScore(item, score)
      if (reorderable(item)) ranked.push(item)
      const group = item.closest<HTMLElement>('.base-card-group, .base-list-group')
      if (group) groupBest.set(group, Math.max(groupBest.get(group) ?? -Infinity, score))
    }
    const scoreOf = (item: HTMLElement) => scores.get(slugOf(item)!)!
    for (const [parent, siblings] of bySibling(ranked)) {
      siblings.sort((a, b) => scoreOf(b) - scoreOf(a))
      parent.prepend(...siblings)
    }
    for (const [parent, siblings] of bySibling(groupBest.keys())) {
      siblings.sort((a, b) => groupBest.get(b)! - groupBest.get(a)!)
      parent.prepend(...siblings)
    }
    collapseGroups()
  }

  // Semantic mode exists only when the build found a vector index for this base.
  const manifest = bar.dataset.semanticManifest
  const modeButtons = Array.from(bar.querySelectorAll<HTMLButtonElement>('.mode-option'))
  const status = bar.querySelector<HTMLElement>('.semantic-status')
  let mode: SearchMode = 'lexical'
  let semanticState: SemanticState = 'idle'
  let searchSeq = 0
  let debounce: ReturnType<typeof setTimeout> | undefined
  let coverage: Promise<boolean> | undefined

  const setSemanticState = (state: SemanticState) => {
    if (!active) return
    semanticState = state
    if (status) {
      status.dataset.state = state
      status.textContent =
        state === 'loading'
          ? 'loading the semantic model'
          : state === 'unavailable'
            ? semanticNotice || 'semantic search unavailable'
            : ''
    }
    const semanticButton = modeButtons.find(button => button.dataset.mode === 'semantic')
    if (semanticButton) {
      semanticButton.disabled = state === 'unavailable'
      semanticButton.title = semanticNotice
      // The icon spins while the encoder loads; the bar has no room for a text label.
      semanticButton.classList.toggle('is-loading', state === 'loading')
      semanticButton.setAttribute('aria-busy', state === 'loading' ? 'true' : 'false')
    }
  }

  const updateModeUI = () => {
    for (const button of modeButtons) {
      const active = button.dataset.mode === mode
      button.classList.toggle('active', active)
      button.setAttribute('aria-pressed', active ? 'true' : 'false')
    }
    bar.dataset.mode = mode
  }

  const ensureSemantic = async (seq: number): Promise<string[] | null> => {
    if (!manifest || semanticState === 'unavailable') return null
    if (semanticState !== 'ready') setSemanticState('loading')
    try {
      if (bar.closest<HTMLElement>('[data-base-query]')?.dataset.baseQueryModified === 'true') {
        coverage ??= manifestIds(manifest).then(ids => {
          const indexed = new Set(ids)
          return items.every(item => {
            const slug = slugOf(item)
            return slug !== undefined && indexed.has(slug)
          })
        })
        const covered = await coverage
        if (!active || seq !== searchSeq) return null
        if (!covered) {
          semanticNotice =
            'Using full-text search because this query includes notes outside the semantic index.'
          setSemanticState('unavailable')
          mode = 'lexical'
          updateModeUI()
          filterLexical()
          return null
        }
      }
      const semantic = sharedSemantic(manifest)
      const ids = await semantic.ids
      if (!active || seq !== searchSeq) return null
      setSemanticState('ready')
      return ids
    } catch (err) {
      if (!active || seq !== searchSeq) return null
      console.warn('[BaseSearch] semantic index unavailable:', err)
      setSemanticState('unavailable')
      return null
    }
  }

  const runSemantic = async () => {
    const query = searchInput.value.trim()
    const seq = ++searchSeq
    if (query.length === 0) {
      releaseSemantic()
      setSemanticState('idle')
      showAll()
      return
    }
    const ids = await ensureSemantic(seq)
    if (seq !== searchSeq) return
    if (!ids) {
      mode = 'lexical'
      updateModeUI()
      filterLexical()
      return
    }
    try {
      const { semantic } = await sharedSemantic(manifest!).client.search(query, SEMANTIC_K)
      if (seq !== searchSeq) return
      const scores = new Map<string, number>()
      const floor = (semantic[0]?.score ?? 0) - SEMANTIC_TAIL
      for (const hit of semantic) {
        const slug = ids[hit.id]
        if (slug && Number.isFinite(hit.score) && hit.score >= floor && !scores.has(slug)) {
          scores.set(slug, hit.score)
        }
      }
      applySemantic(scores)
      releaseSemantic()
    } catch (err) {
      if (seq !== searchSeq) return
      console.warn('[BaseSearch] semantic query failed:', err)
      releaseSemantic()
      filterLexical()
    }
  }

  const search = () => {
    if (mode === 'semantic') {
      void runSemantic()
    } else {
      searchSeq++
      releaseSemantic()
      setSemanticState('idle')
      filterLexical()
    }
  }

  const setMode = (next: SearchMode) => {
    if (next === mode) return
    mode = next
    storeMode(next)
    updateModeUI()
    search()
  }

  const handleInput = () => {
    clearTimeout(debounce)
    searchSeq++
    for (const item of items) clearScore(item)
    if (mode === 'semantic' && searchInput.value.trim().length > 0) {
      debounce = setTimeout(search, DEBOUNCE_MS)
    } else {
      search()
    }
  }

  const handleKeydown = (e: KeyboardEvent) => {
    const activeElement = document.activeElement
    const isTypingInInput =
      activeElement instanceof HTMLInputElement ||
      activeElement instanceof HTMLTextAreaElement ||
      activeElement?.getAttribute('contenteditable') === 'true'

    const searchContainers = document.querySelectorAll<HTMLElement>('.search-container')
    const globalSearchOpen = Array.from(searchContainers).some(el =>
      el.classList.contains('active'),
    )

    if (e.key === 'k' && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
      if (globalSearchOpen || isTypingInInput) {
        return
      }
      e.preventDefault()
      searchInput.focus()
      searchInput.select()
    } else if (e.key === 'Escape') {
      if (document.activeElement === searchInput) {
        searchInput.blur()
      }
    }
  }

  const handleModeClick = (event: MouseEvent) => {
    const button = (event.target as Element | null)?.closest<HTMLButtonElement>('.mode-option')
    if (!button || button.disabled) return
    setMode(button.dataset.mode === 'semantic' ? 'semantic' : 'lexical')
    searchInput.focus()
  }

  searchInput.addEventListener('input', handleInput)
  document.addEventListener('keydown', handleKeydown)
  bar.addEventListener('click', handleModeClick)

  setSemanticState('idle')
  if (manifest) {
    if (shared?.manifest === manifest) setSemanticState('ready')
    if (loadStoredMode() === 'semantic') {
      mode = 'semantic'
    }
  }
  updateModeUI()
  search()

  return () => {
    active = false
    clearTimeout(debounce)
    searchSeq++
    releaseSemantic()
    for (const item of items) clearScore(item)
    searchInput.removeEventListener('input', handleInput)
    document.removeEventListener('keydown', handleKeydown)
    bar.removeEventListener('click', handleModeClick)
  }
}

document.addEventListener('nav', () => {
  let cleanup = setupBaseSearch()
  const refresh = () => {
    cleanup?.()
    cleanup = setupBaseSearch()
  }
  document.addEventListener('base-content-changed', refresh)
  window.addCleanup(() => {
    document.removeEventListener('base-content-changed', refresh)
    cleanup?.()
  })
})
