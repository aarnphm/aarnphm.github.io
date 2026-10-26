export function searchResultsTable(columns: readonly string[], label = 'search results') {
  const table = document.createElement('table')
  table.className = 'search-results-table'
  table.setAttribute('aria-label', label)
  const head = table.createTHead().insertRow()
  for (const name of columns) {
    const heading = document.createElement('th')
    heading.scope = 'col'
    heading.className = `search-column-${name === 'file' ? 'note' : name === 'rank' ? 'score' : name}`
    heading.textContent = name
    head.append(heading)
  }
  return { table, body: table.createTBody() }
}

export function searchStatusTable(query: string, scope = 'notes', message?: string) {
  const { table, body } = searchResultsTable([scope === 'notes' ? 'note' : 'file', 'match'])
  const row = body.insertRow()
  row.className = 'search-empty-row'
  const cell = row.insertCell()
  cell.colSpan = 2
  const status = document.createElement('p')
  status.setAttribute('role', 'status')
  status.textContent =
    message || (query.trim() ? `no results for “${query.trim()}”` : 'search notes and files')
  const hint = document.createElement('p')
  hint.className = 'search-empty-hint'
  hint.textContent = query.trim()
    ? 'try fewer words or change the filter.'
    : 'type a title, phrase, or filter:code to begin.'
  cell.append(status, hint)
  return table
}
