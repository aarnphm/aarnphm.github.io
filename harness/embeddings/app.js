const byId = id => {
  const element = document.getElementById(id)
  if (!element) throw new Error(`Missing harness element: ${id}`)
  return element
}
const put = (id, value) => {
  const element = byId(id)
  const next = String(value)
  if (element.textContent !== next) element.textContent = next
}
const count = value =>
  typeof value === 'number' && Number.isFinite(value)
    ? new Intl.NumberFormat('en-CA').format(value)
    : '–'
const percent = (value, total) => (total ? `${((value / total) * 100).toFixed(1)}%` : '–')
const time = value =>
  value
    ? new Date(value).toLocaleTimeString('en-CA', {
        timeZone: 'America/Toronto',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      })
    : '–'
const duration = seconds => {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '–'
  const rounded = Math.max(0, Math.round(seconds))
  const minutes = Math.round(rounded / 60)
  const hours = Math.round(rounded / 3600)
  if (rounded < 60) return `${rounded}s`
  if (minutes < 60) return `${minutes}m`
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
  return `${Math.floor(hours / 24)}d ${hours % 24}h`
}
const make = (tag, className = '', label = '') => {
  const element = document.createElement(tag)
  element.className = className
  element.textContent = label
  return element
}
const progress = (element, value, total) => {
  element.max = Math.max(1, total)
  element.value = Math.min(value, total)
}

let data = null
let view = 'all'
let socket = null
let retryTimer
let retryDelay = 1000
let connected = false
let fault = ''
let closing = false
const rows = new Map()
const query = byId('query')
const table = byId('documents')

function documentRow(item) {
  let saved = rows.get(item.slug)
  if (!saved) {
    const row = make('tr')
    const titleCell = make('td')
    const details = make('details')
    const summary = make('summary')
    const title = make('span')
    summary.append(title)
    const filename = make('span', 'filename')
    const detail = make('div', 'document-detail')
    const readout = make('p')
    const link = make('a', '', 'open PDF ↗')
    link.href = `http://localhost:8080/${item.slug.split('/').map(encodeURIComponent).join('/')}`
    link.target = '_blank'
    link.rel = 'noopener'
    detail.append(readout, link)
    details.append(summary, detail)
    titleCell.append(details, filename)
    const status = make('td')
    const cached = make('td')
    const indexed = make('td')
    const remaining = make('td')
    cached.dataset.label = 'OCR cached'
    indexed.dataset.label = 'published'
    remaining.dataset.label = 'OCR left'
    const cachedCount = make('span', 'cell-count')
    const indexedCount = make('span', 'cell-count')
    const cachedBar = make('progress')
    const indexedBar = make('progress')
    cachedBar.setAttribute('aria-label', `${item.title}: OCR pages`)
    indexedBar.setAttribute('aria-label', `${item.title}: published pages`)
    cached.append(cachedCount, cachedBar)
    indexed.append(indexedCount, indexedBar)
    row.append(titleCell, status, cached, indexed, remaining)
    saved = {
      row,
      title,
      filename,
      readout,
      link,
      status,
      cachedCount,
      indexedCount,
      cachedBar,
      indexedBar,
      remaining,
    }
    rows.set(item.slug, saved)
  }
  saved.row.dataset.status = item.status
  saved.title.textContent = item.title
  saved.filename.textContent = item.slug.replace(data.prefix, '')
  saved.status.textContent =
    {
      ready: 'awaiting embed',
      active: 'OCR active',
      partial: 'partial',
      paused: 'paused',
      complete: 'complete',
      queued: 'queued',
    }[item.status] ?? item.status
  saved.cachedCount.textContent = `${count(item.ocrPages)} / ${count(item.totalPages)}`
  saved.indexedCount.textContent = `${count(item.indexedPages)} / ${count(item.totalPages)}`
  saved.remaining.textContent = count(item.remainingPages)
  progress(saved.cachedBar, item.ocrPages, item.totalPages)
  progress(saved.indexedBar, item.indexedPages, item.totalPages)
  saved.readout.textContent = `highest cached page: ${count(item.lastPage)} · cache updated: ${time(item.lastProgressAt)}${item.invalidPages ? ` · ${item.invalidPages} invalid cache pages require extraction` : ''}`
  saved.link.hash = `page=${item.lastPage ?? 1}`
  return saved.row
}

function renderDocuments() {
  if (!data) return
  const term = query.value.trim().toLocaleLowerCase()
  let visible = 0
  const currentSlugs = new Set(data.documents.map(item => item.slug))
  for (const [slug, saved] of rows) {
    if (!currentSlugs.has(slug)) {
      saved.row.remove()
      rows.delete(slug)
    }
  }
  for (const item of data.documents) {
    const row = documentRow(item)
    const matchesView =
      view === 'all' ||
      (view === 'remaining' && item.status !== 'complete') ||
      (view === 'active' && ['active', 'paused', 'ready'].includes(item.status)) ||
      (view === 'complete' && item.status === 'complete')
    const visibleRow =
      matchesView && (!term || `${item.title} ${item.slug}`.toLocaleLowerCase().includes(term))
    row.hidden = !visibleRow
    if (!row.parentElement) table.append(row)
    if (visibleRow) visible++
  }
  put('result-count', `${count(visible)} / ${count(data.documents.length)} papers`)
  byId('empty').hidden = visible > 0
}

function renderTables() {
  const history = byId('history')
  const indexes = byId('indexes')
  history.replaceChildren(
    ...[...data.history].reverse().map(batch => {
      const row = make('tr')
      row.append(
        ...[
          time(batch.at),
          `+${count(batch.newVectors)}`,
          count(batch.rows),
          count(batch.pages),
        ].map(value => make('td', '', value)),
      )
      return row
    }),
  )
  indexes.replaceChildren(
    ...data.indexes.map(index => {
      const row = make('tr')
      row.title = `generated: ${time(index.updatedAt)} · ${index.generation ?? 'no generation id'}`
      row.append(
        ...[index.name, count(index.rows), count(index.servedRows), count(index.shards)].map(
          value => make('td', '', value),
        ),
      )
      return row
    }),
  )
  byId('history-empty').hidden = data.history.length > 0
}

function renderCurrent() {
  const runner = data.runner
  const phase = runner.phase
  const label =
    !runner.alive && !['complete', 'error', 'idle'].includes(phase)
      ? 'stopped'
      : runner.stale
        ? 'runner silent'
        : ({
            extracting: 'extracting',
            publishing: 'embedding + publishing',
            complete: 'complete',
            starting: 'starting models',
            error: 'error',
            idle: 'idle',
          }[phase] ?? phase)
  put('runner-phase', label)
  byId('runner-phase').dataset.phase = phase
  for (const step of document.querySelectorAll('.pipeline li')) {
    if (step.dataset.phase === phase) step.setAttribute('aria-current', 'step')
    else step.removeAttribute('aria-current')
  }
  put(
    'current-title',
    data.current?.title ??
      {
        publishing: 'embedding the completed batch',
        complete: 'all papers in this queue are published',
        starting: 'loading the local models',
        error: 'the queue needs attention',
        idle: 'no active queue',
      }[phase] ??
      'preparing the next paper',
  )
  put('current-path', data.current?.slug ?? runner.log ?? '–')
  put(
    'current-page',
    data.current
      ? `pages ${(data.current.pages ?? [data.current.page]).join(', ')} / ${count(data.current.totalPages)}`
      : phase === 'publishing'
        ? `${count(data.totals.awaitingPublicationPages)} cached pages awaiting publication`
        : '–',
  )
  put(
    'batch-count',
    runner.batchCompleted === null
      ? `batch size · ${count(runner.batchPages)} pages`
      : `${count(runner.batchCompleted)} / ${count(runner.batchPlanned)} done · ${count(runner.batchActive)} active · ${count(runner.batchQueued)} queued`,
  )
  const grid = byId('batch-grid')
  const cells = Math.min(runner.pages.length, 128)
  if (grid.children.length !== cells)
    grid.replaceChildren(...Array.from({ length: cells }, () => make('span')))
  Array.from(grid.children).forEach((cell, index) => {
    const page = runner.pages[index]
    cell.className = page.state
    cell.title = `${page.slug} · page ${page.page} · ${page.state}`
  })
  const models = runner.servers
    .map(
      server =>
        `${server.port} ${server.healthy ? 'healthy' : server.alive ? 'unavailable' : 'stopped'}`,
    )
    .join(' · ')
  put('runner-note', `pid ${runner.pid ?? '–'} · ${models || 'model status unavailable'}`)
  const ocr = runner.servers.find(server => server.port === '8001' && server.healthy)
  put(
    'model-metrics',
    ocr
      ? `${count(ocr.inFlight)} / ${count(ocr.maxSequences)} requests active · ${count(ocr.queued)} queued · ${ocr.completedTokensPerSecond === null ? 'collecting token rate' : `${Math.round(ocr.completedTokensPerSecond)} completed tokens/s over ${Math.round(ocr.sampleSeconds)}s`} · ${count(ocr.failedRequests)} failed · prefix cache ${ocr.prefixCache ? 'on' : 'off'} · batching ${ocr.continuousBatching ? 'on' : 'off'}`
      : '',
  )
  put('batch-total', `${count(runner.batches)} completed batches`)
}

function render(next) {
  data = next
  fault = ''
  const totals = data.totals
  put('scope', `PDF queue / ${data.prefix}`)
  put(
    'summary',
    `${count(totals.documents)} papers · ${count(totals.totalPages)} pages · ${count(totals.completeDocuments)} papers complete`,
  )
  put('vectors', count(totals.vectors))
  put('vector-note', 'page-aware passage vectors')
  put('ocr-pages', count(totals.ocrPages))
  put(
    'ocr-note',
    `${percent(totals.ocrPages, totals.totalPages)} of ${count(totals.totalPages)} pages`,
  )
  put('indexed-pages', count(totals.indexedPages))
  put(
    'indexed-note',
    `${percent(totals.indexedPages, totals.totalPages)} of ${count(totals.totalPages)} pages`,
  )
  put('remaining-docs', count(totals.remainingDocuments))
  put(
    'document-note',
    `${count(totals.completeDocuments)} / ${count(totals.documents)} fully published`,
  )
  for (const [id, value] of [
    ['ocr', totals.ocrPages],
    ['index', totals.indexedPages],
    ['served', totals.servedPages],
  ]) {
    progress(byId(`${id}-progress`), value, totals.totalPages)
    put(
      `${id}-ratio`,
      `${count(value)} / ${count(totals.totalPages)} · ${percent(value, totals.totalPages)}`,
    )
  }
  const unpublished = totals.awaitingPublicationPages
  const siteLag = Math.max(0, totals.indexedPages - totals.servedPages)
  put(
    'publication-note',
    `${count(unpublished)} cached pages await the next publication. ${siteLag ? `${count(siteLag)} generated pages await the local site output.` : 'The local site output has the current page coverage.'}`,
  )
  put(
    'pace',
    data.rate.pagesPerHour === null ? '–' : `${count(Math.round(data.rate.pagesPerHour))} pages/h`,
  )
  put('remaining-pages', count(data.rate.remainingUniquePages))
  put('awaiting-pages', count(unpublished))
  put(
    'eta',
    data.rate.remainingOcrSeconds === null
      ? 'collecting timings'
      : `~${duration(data.rate.remainingOcrSeconds)}`,
  )
  put(
    'estimate-note',
    data.rate.remainingUniquePages === 0
      ? 'OCR is cached for every requested page. embedding and publication may remain.'
      : data.rate.samples >= data.rate.minimumSamples
        ? `mean ${duration(data.rate.secondsPerPage)} per page from ${count(data.rate.samples)} pages across ${count(data.rate.sampleDocuments)} ${data.rate.sampleDocuments === 1 ? 'PDF' : 'PDFs'} in this run. duplicate files count once. other documents may differ; embedding, publication and pauses are excluded.`
        : `collecting timings: ${count(data.rate.samples)} / ${count(data.rate.minimumSamples)} completed pages from this run. earlier runs and migrated pages are excluded.`,
  )
  put('observed', `observed ${time(data.observedAt)} · America/Toronto`)
  renderCurrent()
  renderDocuments()
  renderTables()
  health()
}

function health() {
  const since = data ? Math.max(0, (Date.now() - Date.parse(data.observedAt)) / 1000) : null
  const fresh = since !== null && since < 12
  const state = !connected
    ? 'disconnected · reconnecting'
    : fault
      ? 'observer error'
      : !data
        ? 'reading queue'
        : !fresh
          ? 'updates delayed'
          : 'live'
  put('connection-state', state)
  byId('signal').dataset.state = connected && fresh && !fault ? 'live' : 'waiting'
  put(
    'freshness',
    since === null
      ? 'waiting for the first snapshot'
      : `snapshot ${duration(since)} ago · runner ${data.runner.updatedAt ? duration((Date.now() - Date.parse(data.runner.updatedAt)) / 1000) + ' ago' : 'unknown'}`,
  )
  const messages = []
  if (fault) messages.push(`observer: ${fault}. reconnect to retry.`)
  if (!connected && data)
    messages.push(
      'connection lost. counts below are the last received snapshot; reconnecting automatically.',
    )
  if (connected && data && !fresh)
    messages.push(
      'the connection is open but observations have stopped arriving. counts below are the last received snapshot.',
    )
  if (data?.runner.error) messages.push(`runner: ${data.runner.error}`)
  if (data?.runner.stale)
    messages.push(
      data.runner.alive
        ? 'the runner heartbeat is stale. the process exists; page progress may be stalled.'
        : 'the runner process is stopped. cached progress is retained.',
    )
  if (data?.totals.invalidPages)
    messages.push(
      `${count(data.totals.invalidPages)} OCR cache pages failed validation and are excluded from progress.`,
    )
  if (data?.issues.length) messages.push(...data.issues)
  put('notice', messages.join(' '))
  byId('notice').hidden = messages.length === 0
}

function connect() {
  clearTimeout(retryTimer)
  const previous = socket
  const next = new WebSocket(
    `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/events`,
  )
  socket = next
  previous?.close()
  connected = false
  health()
  next.addEventListener('open', () => {
    if (socket !== next) return
    connected = true
    retryDelay = 1000
    health()
  })
  next.addEventListener('message', event => {
    if (socket !== next) return
    try {
      const frame = JSON.parse(event.data)
      if (frame.type === 'snapshot') render(frame.data)
      else if (frame.type === 'error') {
        fault = frame.message
        health()
      }
    } catch {
      fault = 'could not read the latest status'
      health()
    }
  })
  next.addEventListener('close', () => {
    if (socket !== next || closing) return
    connected = false
    health()
    retryTimer = setTimeout(connect, retryDelay)
    retryDelay = Math.min(10_000, retryDelay * 2)
  })
  next.addEventListener('error', () => next.close())
}

query.addEventListener('input', renderDocuments)
for (const button of document.querySelectorAll('[data-filter]')) {
  button.addEventListener('click', () => {
    view = button.dataset.filter
    for (const other of document.querySelectorAll('[data-filter]'))
      other.setAttribute('aria-pressed', String(other === button))
    renderDocuments()
  })
}
byId('reconnect').addEventListener('click', connect)
const applyTheme = theme => {
  document.documentElement.dataset.theme = theme
  const next = theme === 'dark' ? 'light' : 'dark'
  put('theme', next)
  byId('theme').setAttribute('aria-label', `switch to ${next} theme`)
}
applyTheme(
  localStorage.getItem('embedding-harness-theme') ??
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
)
byId('theme').addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'
  localStorage.setItem('embedding-harness-theme', theme)
  applyTheme(theme)
})
const ageTimer = setInterval(health, 1000)
window.addEventListener(
  'pagehide',
  () => {
    closing = true
    clearTimeout(retryTimer)
    clearInterval(ageTimer)
    socket?.close()
  },
  { once: true },
)
if (location.protocol === 'file:') {
  fault = 'start pnpm harness:embeddings and open http://127.0.0.1:8091'
  health()
} else connect()
