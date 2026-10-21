type Table = Record<string, [string, string]>

const parseTable = (raw: string | undefined): Table => {
  try {
    const value: unknown = JSON.parse(raw ?? '{}')
    return value && typeof value === 'object' ? (value as Table) : {}
  } catch {
    return {}
  }
}

/** `*word*` marks emphasis, the split the server render uses. */
const paintWhy = (el: HTMLElement, why: string) => {
  el.replaceChildren(
    ...why.split('*').map((part, i) => {
      if (i % 2 === 0) return document.createTextNode(part)
      const em = document.createElement('em')
      em.textContent = part
      return em
    }),
  )
}

function setup(root: HTMLElement): () => void {
  const table = parseTable(root.dataset.table)
  const groups = Array.from(root.querySelectorAll<HTMLElement>('.fr-rw-group'))
  const phrase = root.querySelector<HTMLElement>('.speech-phrase')
  const play = root.querySelector<HTMLButtonElement>('.speech-play')
  const why = root.querySelector<HTMLElement>('.fr-rw-why')
  const ac = new AbortController()

  const paint = () => {
    const key = groups
      .map(g => g.querySelector<HTMLElement>('[aria-pressed="true"]')?.dataset.option ?? '')
      .join('.')
    const pair = table[key]
    if (!pair || !phrase || !play || !why) return
    const [text, note] = pair
    phrase.dataset.speechText = text
    play.textContent = text
    play.setAttribute('aria-label', `Écouter : ${text}`)
    paintWhy(why, note)
  }

  root.addEventListener(
    'click',
    event => {
      const btn = (event.target as HTMLElement).closest<HTMLButtonElement>('.fr-rw-btn')
      const group = btn?.closest<HTMLElement>('.fr-rw-group')
      if (!btn || !group) return
      for (const b of group.querySelectorAll<HTMLButtonElement>('.fr-rw-btn'))
        b.setAttribute('aria-pressed', String(b === btn))
      paint()
    },
    { signal: ac.signal },
  )
  return () => ac.abort()
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-fr-rewriter]')) {
    window.addCleanup?.(setup(root))
  }
})
