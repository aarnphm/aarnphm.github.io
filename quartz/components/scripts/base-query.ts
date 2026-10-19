import type { QuartzPluginData } from '../../plugins/vfile'
import type { BaseQueryDocument } from '../../util/base/browser-data'
import type { BaseFile, BaseFileFilter, BaseSortConfig } from '../../util/base/types'
import type { Select, SelectGroup, SelectOption } from '../controls/select'
import type { BaseQueryCatalog, BaseQueryEditor } from './base-query-editor'
import { getFullSlug } from '../../util/path'
import { buildSelect } from '../controls/select'
import { cleanupHydratedRoot, rootNavSignal } from './root-lifecycle'

type QueryMode = 'filter' | 'sort'
type QueryRuntime = typeof import('../../util/base/browser-query')
type EditorModule = typeof import('./base-query-editor')
type DataModule = typeof import('../../util/base/browser-data')
type PurifierModule = typeof import('dompurify')
type Dependencies = [QueryRuntime, EditorModule, DataModule, PurifierModule]

let dependenciesPromise: Promise<Dependencies> | undefined

function loadDependencies(): Promise<Dependencies> {
  if (!dependenciesPromise) {
    dependenciesPromise = Promise.all([
      import('../../util/base/browser-query'),
      import('./base-query-editor'),
      import('../../util/base/browser-data'),
      import('dompurify'),
    ]).catch(cause => {
      dependenciesPromise = undefined
      throw cause
    })
  }
  return dependenciesPromise
}

const directionGroups: SelectGroup<BaseSortConfig['direction']>[] = [
  {
    options: [
      { value: 'ASC', label: 'ascending' },
      { value: 'DESC', label: 'descending' },
    ],
  },
]

interface SortRowState {
  row: HTMLTableRowElement
  rank: HTMLTableCellElement
  propertySelect: Select<string>
  directionSelect: Select<BaseSortConfig['direction']>
  removeButton: HTMLButtonElement
  cleanup: () => void
  sort: BaseSortConfig
}

function filterSource(filter: BaseFileFilter | undefined): string {
  if (!filter) return ''
  if (typeof filter === 'string') return filter
  if ('and' in filter)
    return filter.and.length === 0
      ? 'true'
      : filter.and
          .map(filterSource)
          .filter(Boolean)
          .map(value => `(${value})`)
          .join(' && ')
  if ('or' in filter)
    return filter.or.length === 0
      ? 'false'
      : filter.or
          .map(filterSource)
          .filter(Boolean)
          .map(value => `(${value})`)
          .join(' || ')
  return filter.not.length === 0
    ? 'true'
    : filter.not
        .map(filterSource)
        .filter(Boolean)
        .map(value => `!(${value})`)
        .join(' && ')
}

function queryMode(value: string | undefined): QueryMode | undefined {
  return value === 'filter' || value === 'sort' ? value : undefined
}

export async function mountBaseQuery(
  root: HTMLElement,
  initialMode: QueryMode = 'filter',
): Promise<void> {
  if (root.dataset.baseQueryInitialized !== undefined) return
  const panelElement = root.querySelector<HTMLElement>('#base-query-panel')
  const surfaceElement = root.querySelector<HTMLElement>('[data-base-query-surface]')
  const headingElement = root.querySelector<HTMLElement>('[data-base-query-heading]')
  const statusElement = root.querySelector<HTMLElement>('[data-base-query-status]')
  const error = root.querySelector<HTMLElement>('[data-base-query-error]')
  const applyButton = root.querySelector<HTMLButtonElement>('[data-base-query-apply]')
  const resetButton = root.querySelector<HTMLButtonElement>('[data-base-query-reset]')
  const downloadButton = root.querySelector<HTMLButtonElement>('[data-base-query-download]')
  const closeButton = root.querySelector<HTMLButtonElement>('[data-base-query-close]')
  if (
    !panelElement ||
    !surfaceElement ||
    !headingElement ||
    !statusElement ||
    !error ||
    !applyButton ||
    !resetButton ||
    !downloadButton ||
    !closeButton
  )
    return
  const panel = panelElement
  const surface = surfaceElement
  const heading = headingElement
  const status = statusElement

  root.dataset.baseQueryInitialized = ''
  const signal = rootNavSignal(root)
  const toggles = [...root.querySelectorAll<HTMLButtonElement>('[data-base-query-toggle]')]
  const filterSurface = document.createElement('div')
  const sortSurface = document.createElement('div')
  filterSurface.hidden = true
  sortSurface.hidden = true
  surface.replaceChildren(filterSurface, sortSurface)

  let runtime: QueryRuntime | undefined
  let editorModule: EditorModule | undefined
  let queryDocument: BaseQueryDocument | undefined
  let files: QuartzPluginData[] = []
  let config: BaseFile | undefined
  let source = ''
  let selectedView = 0
  let catalog: BaseQueryCatalog | undefined
  let cachedCatalogForOptions: BaseQueryCatalog | undefined
  let catalogPropertySet = new Set<string>()
  let catalogPropertyOptions: SelectOption<string>[] = []
  let catalogPropertyGroup: SelectGroup<string> = { options: [] }
  let editor: BaseQueryEditor | undefined
  let editorSource = ''
  let scopeElement: HTMLDetailsElement | undefined
  let scopeCode: HTMLElement | undefined
  let sortBody: HTMLTableSectionElement | undefined
  let addSortButton: HTMLButtonElement | undefined
  let sortRows: SortRowState[] = []
  let mode: QueryMode = 'filter'
  let initialized = false
  let initialization: Promise<void> | undefined
  let sanitize: ((html: string) => string) | undefined
  let draftDirty = false

  const setError = (messages: string[]): void => {
    error.textContent = messages.join('\n')
    error.hidden = messages.length === 0
  }
  const messageOf = (cause: unknown): string =>
    cause instanceof Error ? cause.message : 'The base editor could not complete this action.'
  const changed = (): void => {
    draftDirty = true
    status.textContent = 'Unapplied changes.'
  }
  const ready = (enabled: boolean): void => {
    applyButton.disabled = !enabled
    resetButton.disabled = !enabled
    downloadButton.disabled = !enabled
  }
  const currentView = () => config?.views[selectedView]

  function getPropertyGroups(propertyName: string): SelectGroup<string>[] {
    if (catalog && catalog !== cachedCatalogForOptions) {
      cachedCatalogForOptions = catalog
      catalogPropertySet = new Set(catalog.properties)
      catalogPropertyOptions = catalog.properties.map(value => ({ value, label: value }))
      catalogPropertyGroup = { options: catalogPropertyOptions }
    }
    if (catalogPropertySet.has(propertyName)) {
      return [catalogPropertyGroup]
    }
    return [{ options: [{ value: propertyName, label: propertyName }, ...catalogPropertyOptions] }]
  }

  async function initialize(): Promise<void> {
    if (initialized) return
    if (initialization) return initialization
    status.textContent = 'Loading base editor…'
    panel.setAttribute('aria-busy', 'true')
    initialization = (async () => {
      const [query, editors, data, purifier] = await loadDependencies()
      const raw: unknown = JSON.parse(root.dataset.baseQuery ?? 'null')
      const documentData = data.parseBaseQueryDocument(raw)
      if (!documentData) throw new Error('The source for this base is unavailable.')
      const parsed = query.parseBaseQuery(documentData.source)
      if (!parsed.ok) throw new Error(parsed.diagnostics.join('\n'))
      const publishedFiles = await query.loadBaseQueryData()
      if (signal.aborted) return
      runtime = query
      editorModule = editors
      queryDocument = documentData
      config = parsed.config
      source = documentData.source
      files = publishedFiles
      selectedView = Math.max(
        0,
        config.views.findIndex(view => view.name === root.dataset.baseQueryView),
      )
      catalog = editors.createBaseQueryCatalog(files, config)
      sanitize = html => purifier.default.sanitize(html)
      initialized = true
      status.textContent = `${files.length.toLocaleString()} published notes available.`
      ready(true)
    })()
      .catch(cause => {
        initialization = undefined
        throw cause
      })
      .finally(() => panel.removeAttribute('aria-busy'))
    return initialization
  }

  function syncSource(): boolean {
    if (!runtime || !config) return false
    if (editor && editor.value() === editorSource) {
      setError([])
      return true
    }
    if (mode === 'filter' && editor) {
      const view = currentView()
      if (!view) return false
      const expression = editor.value().trim()
      const nextView = { ...view }
      if (expression) nextView.filters = expression
      else delete nextView.filters
      const nextConfig = {
        ...config,
        views: config.views.map((entry, index) => (index === selectedView ? nextView : entry)),
      }
      const draft = runtime.baseQuerySource(nextConfig)
      const parsed = runtime.parseBaseQuery(draft)
      if (!parsed.ok) {
        setError(parsed.diagnostics)
        return false
      }
      config = parsed.config
      source = draft
      editorSource = editor.value()
    }
    setError([])
    return true
  }

  function close(returnFocus = true): void {
    panel.hidden = true
    for (const toggle of toggles) toggle.setAttribute('aria-expanded', 'false')
    if (returnFocus) toggles.find(toggle => toggle.dataset.baseQueryToggle === mode)?.focus()
  }

  function updateSort(sorts: BaseSortConfig[]): void {
    if (!config || !runtime) return
    const view = currentView()
    if (!view) return
    view.sort = sorts
    source = runtime.baseQuerySource(config)
    changed()
  }

  function reindexSortRows(startIndex = 0): void {
    for (let index = startIndex; index < sortRows.length; index++) {
      const item = sortRows[index]
      item.rank.textContent = String(index + 1)

      const propId = `base-query-property-${index}`
      const propLabel = `Sort property ${index + 1}`
      const propMenu = item.propertySelect.element.querySelector<HTMLElement>('.g-select-menu')
      if (propMenu) {
        propMenu.id = propId
        propMenu.setAttribute('aria-label', propLabel)
      }
      item.propertySelect.trigger.setAttribute('aria-controls', propId)
      item.propertySelect.trigger.setAttribute('aria-label', `${propLabel}: ${item.sort.property}`)

      const dirId = `base-query-direction-${index}`
      const dirLabel = `Sort direction ${index + 1}`
      const dirText = item.sort.direction === 'ASC' ? 'ascending' : 'descending'
      const dirMenu = item.directionSelect.element.querySelector<HTMLElement>('.g-select-menu')
      if (dirMenu) {
        dirMenu.id = dirId
        dirMenu.setAttribute('aria-label', dirLabel)
      }
      item.directionSelect.trigger.setAttribute('aria-controls', dirId)
      item.directionSelect.trigger.setAttribute('aria-label', `${dirLabel}: ${dirText}`)

      item.removeButton.setAttribute('aria-label', `Remove sort ${index + 1}`)
      item.removeButton.title = `Remove sort ${index + 1}`
    }
  }

  function createSortRow(sort: BaseSortConfig, index: number): SortRowState {
    const row = document.createElement('tr')
    row.className = 'base-query-sort'
    row.dataset.baseQuerySort = ''
    const rank = document.createElement('td')
    rank.className = 'base-query-sort-rank'
    rank.textContent = String(index + 1)
    rank.setAttribute('aria-hidden', 'true')

    const rowController = new AbortController()
    const propertySelect = buildSelect({
      id: `base-query-property-${index}`,
      label: `Sort property ${index + 1}`,
      groups: getPropertyGroups(sort.property),
      selected: sort.property,
      className: 'base-query-select',
      floating: true,
      onSelect: property => {
        rowState.sort = { ...rowState.sort, property }
        updateSort(sortRows.map(item => item.sort))
      },
    })
    const directionSelect = buildSelect<BaseSortConfig['direction']>({
      id: `base-query-direction-${index}`,
      label: `Sort direction ${index + 1}`,
      groups: directionGroups,
      selected: sort.direction,
      className: 'base-query-direction',
      floating: true,
      onSelect: direction => {
        rowState.sort = { ...rowState.sort, direction }
        updateSort(sortRows.map(item => item.sort))
      },
    })
    const removeButton = document.createElement('button')
    removeButton.type = 'button'
    removeButton.innerHTML =
      '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="m4 4 8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>'
    removeButton.setAttribute('aria-label', `Remove sort ${index + 1}`)
    removeButton.title = `Remove sort ${index + 1}`

    const unmountProperty = propertySelect.mount()
    const unmountDirection = directionSelect.mount()
    const rowState: SortRowState = {
      row,
      rank,
      propertySelect,
      directionSelect,
      removeButton,
      sort: { ...sort },
      cleanup: () => {
        rowController.abort()
        unmountProperty()
        unmountDirection()
      },
    }

    removeButton.addEventListener(
      'click',
      () => {
        const currentIndex = sortRows.indexOf(rowState)
        if (currentIndex === -1) return
        addSortButton?.focus({ preventScroll: true })
        rowState.cleanup()
        rowState.row.remove()
        sortRows.splice(currentIndex, 1)
        reindexSortRows(currentIndex)
        updateSort(sortRows.map(item => item.sort))
      },
      { signal: rowController.signal },
    )

    row.append(rank)
    for (const control of [propertySelect.element, directionSelect.element, removeButton]) {
      const cell = document.createElement('td')
      cell.append(control)
      row.append(cell)
    }
    return rowState
  }

  function clearSortRows(): void {
    for (const item of sortRows) item.cleanup()
    sortRows = []
    sortBody?.replaceChildren()
  }

  function rebuildSortRows(): void {
    if (!sortBody || !catalog) return
    clearSortRows()
    const sorts = currentView()?.sort ?? []
    const fragment = document.createDocumentFragment()
    for (const [index, sort] of sorts.entries()) {
      const rowState = createSortRow(sort, index)
      sortRows.push(rowState)
      fragment.append(rowState.row)
    }
    sortBody.append(fragment)
  }

  function ensureSortSurface(): void {
    if (!catalog) return
    if (sortBody) return
    const list = document.createElement('table')
    list.className = 'base-query-sorts'
    list.setAttribute('aria-label', 'Sort order')
    const columns = document.createElement('colgroup')
    for (const name of ['rank', 'property', 'direction', 'remove']) {
      const column = document.createElement('col')
      column.className = `base-query-column-${name}`
      columns.append(column)
    }
    const body = document.createElement('tbody')
    sortBody = body
    list.append(columns, body)

    const add = document.createElement('button')
    add.type = 'button'
    add.dataset.baseQueryAddSort = ''
    add.textContent = 'add sort'
    addSortButton = add
    add.addEventListener(
      'click',
      () => {
        if (!sortBody) return
        const newSort: BaseSortConfig = { property: 'file.name', direction: 'ASC' }
        const rowState = createSortRow(newSort, sortRows.length)
        sortRows.push(rowState)
        sortBody.append(rowState.row)
        updateSort(sortRows.map(item => item.sort))
        requestAnimationFrame(() => {
          rowState.propertySelect.trigger.focus({ preventScroll: true })
        })
      },
      { signal },
    )
    const footer = document.createElement('tfoot')
    const addRow = document.createElement('tr')
    const spacer = document.createElement('td')
    spacer.setAttribute('aria-hidden', 'true')
    const addCell = document.createElement('td')
    addCell.colSpan = 3
    addCell.append(add)
    addRow.append(spacer, addCell)
    footer.append(addRow)
    list.append(footer)
    sortSurface.append(list)
    rebuildSortRows()
  }

  function ensureFilterSurface(): void {
    if (!config || !editorModule || !catalog) return
    const view = currentView()
    const nextFilterSource = filterSource(view?.filters)
    if (!editor) {
      const editorHost = document.createElement('div')
      editorHost.className = 'base-query-editor base-query-editor-filter'
      filterSurface.prepend(editorHost)
      editor = editorModule.createBaseQueryEditor({
        parent: editorHost,
        source: nextFilterSource,
        catalog,
        onChange: changed,
        onApply: () => {
          void apply()
        },
        onClose: close,
      })
      editorSource = editor.value()
    } else if (editor.value() !== nextFilterSource && editor.value() === editorSource) {
      editor.setValue(nextFilterSource)
      editorSource = editor.value()
    }
    if (config.filters) {
      if (!scopeElement || !scopeCode) {
        scopeElement = document.createElement('details')
        scopeElement.className = 'base-query-scope'
        const summary = document.createElement('summary')
        summary.textContent = 'Base filters'
        scopeCode = document.createElement('code')
        scopeElement.append(summary, scopeCode)
        filterSurface.append(scopeElement)
      }
      scopeCode.textContent = filterSource(config.filters)
      scopeElement.hidden = false
    } else if (scopeElement) {
      scopeElement.hidden = true
    }
  }

  function showActiveSurface(): void {
    if (!config || !editorModule) return
    const view = currentView()
    heading.textContent = `${mode === 'filter' ? 'Filter' : 'Sort'} · ${view?.name ?? 'this view'}`
    filterSurface.hidden = mode !== 'filter'
    sortSurface.hidden = mode !== 'sort'
    if (mode === 'sort') {
      ensureSortSurface()
    } else {
      ensureFilterSurface()
    }
  }

  async function open(nextMode: QueryMode): Promise<void> {
    if (config && mode !== nextMode && !syncSource()) {
      panel.hidden = false
      return
    }
    mode = nextMode
    panel.hidden = false
    for (const toggle of toggles) {
      const selected = toggle.dataset.baseQueryToggle === mode
      toggle.setAttribute('aria-expanded', String(selected))
      toggle.setAttribute('aria-selected', String(selected))
      toggle.tabIndex = selected ? 0 : -1
    }
    panel.setAttribute('aria-labelledby', `base-query-tab-${mode}`)
    if (initialized) {
      showActiveSurface()
      return
    }
    try {
      await initialize()
      if (signal.aborted || panel.hidden || mode !== nextMode) return
      showActiveSurface()
    } catch (cause) {
      if (!signal.aborted) {
        status.textContent = 'The editor could not load. Close and reopen it to retry.'
        setError([messageOf(cause)])
      }
    }
  }

  async function apply(reset = false): Promise<void> {
    if (!runtime || !queryDocument || !sanitize || !config || !editorModule) return
    if (!reset && !syncSource()) return
    const nextSource = reset ? queryDocument.source : source
    const viewName = reset ? (root.dataset.baseQueryView ?? '') : (currentView()?.name ?? '')
    ready(false)
    panel.setAttribute('aria-busy', 'true')
    status.textContent = 'Updating this view…'
    try {
      const result = runtime.runBaseQuery({
        source: nextSource,
        baseFile: queryDocument.baseFile,
        thisFile: queryDocument.thisFile,
        files,
        viewName,
        currentSlug: getFullSlug(window),
      })
      if (!result.ok) {
        setError(result.diagnostics)
        status.textContent = 'The previous results are unchanged.'
        return
      }
      const previous = document.querySelector<HTMLElement>('.main-col > .base-view')
      if (!previous) throw new Error('The current base view is unavailable.')
      const coverThumbs = new Map<string, string>()
      for (const img of previous.querySelectorAll<HTMLImageElement>('img[data-bookcase-cover]')) {
        const key = img.dataset.bookcaseCover
        const src = img.getAttribute('src')
        if (key && src) coverThumbs.set(key, src)
      }
      const fragment = document.createElement('template')
      fragment.innerHTML = sanitize(result.html)
      const next = fragment.content.querySelector<HTMLElement>('.base-view')
      if (!next) throw new Error('The query did not produce a view.')
      if (coverThumbs.size > 0) {
        for (const img of next.querySelectorAll<HTMLImageElement>('img[data-bookcase-cover]')) {
          const key = img.dataset.bookcaseCover
          const thumb = key ? coverThumbs.get(key) : undefined
          if (thumb) img.setAttribute('src', thumb)
        }
      }
      cleanupHydratedRoot(previous)
      const diagnostics = previous.previousElementSibling
      if (diagnostics?.classList.contains('base-diagnostics')) diagnostics.remove()
      previous.replaceWith(fragment.content)
      source = nextSource
      draftDirty = false
      config = result.config
      selectedView = Math.max(
        0,
        config.views.findIndex(view => view.name === result.view.name),
      )
      catalog = editorModule.createBaseQueryCatalog(files, config)
      setError([])
      status.textContent = `${result.resultCount.toLocaleString()} of ${result.totalCount.toLocaleString()} matching notes${reset ? ' · source restored' : ''}.`
      root.dataset.baseQueryModified = String(source !== queryDocument.source)
      document.dispatchEvent(new CustomEvent('contentdecrypted', { detail: { content: next } }))
      document.dispatchEvent(new CustomEvent<void>('base-content-changed'))
      if (reset) {
        if (editor) {
          const restoredFilter = filterSource(currentView()?.filters)
          editor.setValue(restoredFilter)
          editorSource = editor.value()
        }
        if (sortBody) {
          rebuildSortRows()
        }
        showActiveSurface()
      }
    } catch (cause) {
      setError([messageOf(cause)])
      status.textContent = 'The previous results are unchanged.'
    } finally {
      ready(true)
      panel.removeAttribute('aria-busy')
    }
  }

  function download(): void {
    if (!syncSource() || !queryDocument) return
    const blob = new Blob([source], { type: 'application/yaml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${String(queryDocument.baseFile.slug).split('/').at(-1) ?? 'query'}.base`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    status.textContent = 'Downloaded the current .base draft with all views.'
  }

  for (const toggle of toggles) {
    toggle.addEventListener(
      'click',
      () => {
        const next = queryMode(toggle.dataset.baseQueryToggle)
        if (!next) return
        if (!panel.hidden && mode === next) close()
        else void open(next)
      },
      { signal },
    )
  }
  closeButton.addEventListener('click', () => close(), { signal })
  document.addEventListener(
    'pointerdown',
    event => {
      if (!panel.hidden && event.target instanceof Node && !root.contains(event.target))
        close(false)
    },
    { signal },
  )
  root.addEventListener(
    'keydown',
    event => {
      if (
        event.key === 'Escape' &&
        event.target instanceof HTMLButtonElement &&
        toggles.includes(event.target)
      ) {
        event.preventDefault()
        close()
      }
    },
    { signal },
  )
  applyButton.addEventListener(
    'click',
    () => {
      void apply()
    },
    { signal },
  )
  resetButton.addEventListener(
    'click',
    () => {
      void apply(true)
    },
    { signal },
  )
  downloadButton.addEventListener('click', download, { signal })
  panel.addEventListener(
    'keydown',
    event => {
      if (event.key === 'Escape' && !event.defaultPrevented) {
        event.preventDefault()
        close()
      }
    },
    { signal },
  )
  document.addEventListener(
    'base-search-results',
    event => {
      if (!(event instanceof CustomEvent) || !config || draftDirty) return
      const detail: unknown = event.detail
      if (typeof detail !== 'object' || !detail || !('visible' in detail) || !('total' in detail))
        return
      if (typeof detail.visible !== 'number' || typeof detail.total !== 'number') return
      const notice = 'notice' in detail && typeof detail.notice === 'string' ? detail.notice : ''
      status.textContent = `${detail.visible.toLocaleString()} of ${detail.total.toLocaleString()} results in this view.${notice ? ` ${notice}` : ''}`
    },
    { signal },
  )
  signal.addEventListener(
    'abort',
    () => {
      editor?.destroy()
      clearSortRows()
      delete root.dataset.baseQueryInitialized
    },
    { once: true },
  )
  await open(initialMode)
}
