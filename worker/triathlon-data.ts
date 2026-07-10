type FeedRecord = Record<string, unknown>

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

function recordLabel(record: FeedRecord): string {
  const kind = typeof record.kind === 'string' ? record.kind : 'record'
  const when = record.date ?? record.weekStart ?? record.today
  const parts: string[] = []
  if (typeof when === 'string') parts.push(when)
  if (typeof record.sport === 'string') parts.push(record.sport)
  if (typeof record.name === 'string') parts.push(record.name)
  return `<span class="k">${escapeHtml(kind)}</span>${parts.length > 0 ? ` ${escapeHtml(parts.join(' · '))}` : ''}`
}

function renderRecord(line: string, index: number): string {
  let record: FeedRecord | null = null
  try {
    const parsed: unknown = JSON.parse(line)
    if (typeof parsed === 'object' && parsed !== null) record = parsed as FeedRecord
  } catch {}
  const open = index === 0 ? ' open' : ''
  const label = record ? recordLabel(record) : '<span class="k">unparsed</span>'
  const body = record ? JSON.stringify(record, null, 2) : line
  return `<details${open}><summary>${label}</summary><pre>${escapeHtml(body)}</pre></details>`
}

const PAGE_STYLE = `
:root { --paper: #fffcf0; --ink: #100f0f; --muted: #6f6e69; --line: #e6e4d9; }
@media (prefers-color-scheme: dark) {
  :root { --paper: #100f0f; --ink: #cecdc3; --muted: #878580; --line: #282726; }
}
* { box-sizing: border-box; }
body {
  margin: 2rem auto 4rem;
  max-width: 78ch;
  padding: 0 1rem;
  background: var(--paper);
  color: var(--ink);
  font: 13px/1.55 'Berkeley Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}
header { margin-bottom: 1.25rem; }
h1 { margin: 0; font-size: 1rem; font-weight: 600; }
header p { margin: 0.2rem 0 0; color: var(--muted); }
a { color: inherit; }
details { border-top: 1px solid var(--line); }
details:last-of-type { border-bottom: 1px solid var(--line); }
summary { padding: 0.35rem 0; color: var(--muted); cursor: pointer; }
summary .k { color: var(--ink); font-weight: 600; }
pre { margin: 0 0 0.75rem; padding: 0.25rem 0 0.5rem 1ch; overflow-x: auto; }
`

export function triathlonDataHtml(text: string): string {
  const lines = text.split('\n').filter(line => line.trim().length > 0)
  const counts = new Map<string, number>()
  for (const line of lines) {
    let kind = 'unparsed'
    try {
      const parsed: unknown = JSON.parse(line)
      if (typeof parsed === 'object' && parsed !== null) {
        const k = (parsed as FeedRecord).kind
        kind = typeof k === 'string' ? k : 'record'
      }
    } catch {}
    counts.set(kind, (counts.get(kind) ?? 0) + 1)
  }
  const summary = [...counts.entries()].map(([kind, n]) => `${n} ${kind}`).join(' · ')
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>triathlon/data</title>
<style>${PAGE_STYLE}</style>
</head>
<body>
<header>
<h1>triathlon/data</h1>
<p>${escapeHtml(summary)} · <a href="/triathlon/data.jsonl">raw jsonl</a></p>
</header>
${lines.map(renderRecord).join('\n')}
</body>
</html>`
}
