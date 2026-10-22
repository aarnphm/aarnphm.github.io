import type { ComponentChildren } from 'preact'

export type Option = { id: string; label: string }
export type Axis = { id: string; label: string; options: Option[] }
/** `"<option>.<option>"` in axis order, to `[sentence, why]`. */
export type Table = Record<string, [string, string]>
export type RewriterProps = {
  caption?: string
  label?: string
  initial?: string
  /** MDX hands JSON-valued props over as strings. */
  pairs?: string | Record<string, unknown>
}

export const parseJson = <T,>(raw: unknown): T | undefined => {
  if (typeof raw !== 'string') return raw as T | undefined
  try {
    return JSON.parse(raw) as T
  } catch {
    return undefined
  }
}

/** `*word*` marks emphasis in a why line; the client renders the same split. */
export const emParts = (text: string): ComponentChildren[] =>
  text.split('*').map((part, i) => (i % 2 ? <em>{part}</em> : part))

type Props = RewriterProps & { cls: string; axes: Axis[]; table: Table }

/** Shared frame of the French rewriters: axis buttons, one speakable sentence, one why line. */
export function FrRewriter({ cls, axes, table, caption, label, initial }: Props) {
  const start = (initial ?? '').split('.')
  const picked = axes.map((axis, i) =>
    axis.options.some(o => o.id === start[i]) ? start[i] : axis.options[0].id,
  )
  const [text, why] = table[picked.join('.')] ?? ['–', '']
  return (
    <figure class={`fr-rewriter ${cls}`} data-fr-rewriter data-table={JSON.stringify(table)}>
      {axes.map((axis, i) => (
        <div class="fr-rw-group" role="tablist" aria-label={axis.label} data-axis={axis.id}>
          {axis.options.map(o => (
            <button
              type="button"
              role="tab"
              class="fr-rw-btn"
              {...{ 'data-option': o.id }}
              aria-selected={o.id === picked[i]}
              tabIndex={o.id === picked[i] ? 0 : -1}
            >
              {o.label}
            </button>
          ))}
        </div>
      ))}
      <div class="fr-rw-row">
        <p class="fr-rw-out" role="status" lang="fr" aria-label={label}>
          {text}
        </p>
        <button type="button" class="fr-rw-speak" aria-label="écouter" hidden>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M2 6h3l4-3v10l-4-3H2zM11.5 5.5a3.5 3.5 0 0 1 0 5M13 3.5a6.5 6.5 0 0 1 0 9" />
          </svg>
        </button>
      </div>
      <p class="fr-rw-why">{emParts(why)}</p>
      {caption && <figcaption class="fr-rw-caption">{caption}</figcaption>}
    </figure>
  )
}
