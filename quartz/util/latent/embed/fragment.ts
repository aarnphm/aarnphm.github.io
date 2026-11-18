import { readSharedView, type SharedView } from '../share'

/**
 * `![[embeddings#focus=3716#view=<base64url>]]`: the fragments of a reference to a `layout:
 * latent` page, as `key=value` segments (the triathlon convention). The selector keeps the raw
 * text so it stays serializable; the renderer decodes it once (util/latent/embed/render.ts).
 */
export interface LatentFragment {
  focus?: number
  view?: string
  /** why the fragments do not name a view; the reference still renders a notice */
  errors: string[]
  /** keys that are not `focus` or `view` */
  unsupported: string[]
}

export function parseLatentFragment(fragments: readonly string[]): LatentFragment {
  const out: LatentFragment = { errors: [], unsupported: [] }
  for (const raw of fragments) {
    const segment = raw.trim()
    if (!segment) continue
    const at = segment.indexOf('=')
    const key = at < 0 ? segment : segment.slice(0, at)
    const value = at < 0 ? '' : segment.slice(at + 1).trim()
    if (key === 'focus') {
      if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)))
        out.errors.push(`focus must be a row number: ${value || '(empty)'}`)
      else out.focus = Number(value)
    } else if (key === 'view') {
      if (!value) out.errors.push('view is empty')
      else out.view = value
    } else out.unsupported.push(key)
  }
  if (out.view && !decodeLatentView(out.view)) out.errors.push('view does not decode')
  return out
}

/** The share payload as /embeddings reads it (util/latent/share.ts bounds every field). */
export function decodeLatentView(payload: string): SharedView | null {
  return readSharedView(`#view=${payload}`)
}

/** The fragment of `![[<slug>#focus=N#view=…]]` for a view (the share button's "copy embed"). */
export function latentEmbedSnippet(slug: string, view: SharedView, payload: string): string {
  return `![[${slug}#focus=${view.focus}#view=${payload}]]`
}
