import { registerMdxContent } from './mdx-content'

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

/** Prefer the exact locale, then any fr voice; `undefined` lets the browser choose by `lang`. */
const pickVoice = (lang: string): SpeechSynthesisVoice | undefined => {
  const voices = speechSynthesis.getVoices().filter(v => /^fr\b/i.test(v.lang.replace('_', '-')))
  return (
    voices.find(v => v.lang.replace('_', '-').toLowerCase() === lang.toLowerCase()) ?? voices[0]
  )
}

function setup(root: HTMLElement): () => void {
  const table = parseTable(root.dataset.table)
  const groups = Array.from(root.querySelectorAll<HTMLElement>('.fr-rw-group'))
  const out = root.querySelector<HTMLElement>('.fr-rw-out')
  const why = root.querySelector<HTMLElement>('.fr-rw-why')
  const speak = root.querySelector<HTMLButtonElement>('.fr-rw-speak')
  const ac = new AbortController()
  const { signal } = ac

  const selected = (group: HTMLElement) =>
    group.querySelector<HTMLElement>('[aria-selected="true"]')?.dataset.option ?? ''

  const paint = () => {
    const pair = table[groups.map(selected).join('.')]
    if (!pair || !out || !why) return
    out.textContent = pair[0]
    paintWhy(why, pair[1])
  }

  const select = (tab: HTMLElement, focus: boolean) => {
    const group = tab.closest<HTMLElement>('.fr-rw-group')
    if (!group) return
    for (const t of group.querySelectorAll<HTMLElement>('[role="tab"]')) {
      t.setAttribute('aria-selected', String(t === tab))
      t.tabIndex = t === tab ? 0 : -1
    }
    if (focus) tab.focus()
    paint()
  }

  root.addEventListener(
    'click',
    event => {
      const tab = (event.target as HTMLElement).closest<HTMLElement>('[role="tab"]')
      if (tab) select(tab, false)
    },
    { signal },
  )

  root.addEventListener(
    'keydown',
    event => {
      const tab = (event.target as HTMLElement).closest<HTMLElement>('[role="tab"]')
      const group = tab?.closest<HTMLElement>('.fr-rw-group')
      if (!tab || !group) return
      const tabs = Array.from(group.querySelectorAll<HTMLElement>('[role="tab"]'))
      const at = tabs.indexOf(tab)
      const next =
        event.key === 'ArrowRight' || event.key === 'ArrowDown'
          ? (at + 1) % tabs.length
          : event.key === 'ArrowLeft' || event.key === 'ArrowUp'
            ? (at - 1 + tabs.length) % tabs.length
            : event.key === 'Home'
              ? 0
              : event.key === 'End'
                ? tabs.length - 1
                : -1
      if (next < 0) return
      event.preventDefault()
      select(tabs[next], true)
    },
    { signal },
  )

  if (speak && 'speechSynthesis' in window) {
    speak.hidden = false
    speak.addEventListener(
      'click',
      () => {
        const text = out?.textContent?.trim()
        if (!text) return
        const variety = groups.find(g => g.dataset.axis === 'variete')
        const lang = variety && selected(variety) === 'qc' ? 'fr-CA' : 'fr-FR'
        const utterance = new SpeechSynthesisUtterance(text)
        const voice = pickVoice(lang)
        utterance.voice = voice ?? null
        utterance.lang = voice?.lang ?? lang
        speechSynthesis.cancel()
        speechSynthesis.speak(utterance)
      },
      { signal },
    )
  }
  return () => {
    ac.abort()
    if ('speechSynthesis' in window) speechSynthesis.cancel()
  }
}

registerMdxContent('[data-fr-rewriter]', root => {
  window.addCleanup?.(setup(root))
})
