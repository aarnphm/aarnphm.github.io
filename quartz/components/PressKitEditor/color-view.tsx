import { useEffect, useId, useRef, useState } from 'preact/hooks'
import { cardColors } from './card-colors'
import { backgroundKey } from './color'
import { ColorPicker } from './color-picker'
import { selection, type EditorState } from './layout'
import { edit, queue } from './stage'

function hexColor(value: string) {
  const hex = value.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i)?.[1]
  if (!hex) return
  return `#${hex.length === 3 ? [...hex].map(char => char + char).join('') : hex}`.toLowerCase()
}

const OPTIONS = [
  { value: 'auto', label: 'auto' },
  { value: '#000000', label: 'noir' },
  { value: '#ffffff', label: 'blanc' },
]

/** One colour slot (ink, or a highlighter) for every selected layer, keyed into `colors`. */
function ColorField({
  state,
  keys,
  title,
  name,
}: {
  state: EditorState
  keys: string[]
  title: string
  name: string
}) {
  const errorId = useId()
  const values = new Set(keys.map(key => state.draft.value.colors[key] ?? 'auto'))
  const value = values.size === 1 ? [...values][0] : 'mixed'
  const keyset = keys.join('\n')
  const [hex, setHex] = useState('')
  const [invalid, setInvalid] = useState(false)
  // Colours from before a picker drag; the drag paints live and lands as one undo step.
  const origin = useRef<Map<string, string | undefined> | null>(null)
  useEffect(() => {
    setHex(value.startsWith('#') ? value : '')
    setInvalid(false)
  }, [value, keyset])

  const apply = (color?: string) => {
    setHex(color ?? '')
    setInvalid(false)
    if (keys.every(key => state.draft.value.colors[key] === color)) return
    edit(state, (_, doc) => {
      for (const key of keys) {
        if (color) doc.colors[key] = color
        else delete doc.colors[key]
      }
    })
  }
  const preview = (color: string) => {
    const colors = state.draft.value.colors
    origin.current ??= new Map(keys.map(key => [key, colors[key]]))
    for (const key of keys) colors[key] = color
    setHex(color)
    queue(state)
  }
  const commit = (color: string) => {
    const before = origin.current
    origin.current = null
    const colors = state.draft.value.colors
    for (const [key, previous] of before ?? []) {
      if (previous) colors[key] = previous
      else delete colors[key]
    }
    apply(color)
  }
  const custom = (input: string) => {
    const color = hexColor(input)
    setInvalid(!color)
    if (color) apply(color)
  }
  return (
    <div class="press-kit-editor-color" role="group" aria-label={`${title} de la sélection`}>
      <span>
        {title}
        {value === 'mixed' ? ' · couleurs différentes' : ''}
      </span>
      <div class="press-kit-segments">
        {OPTIONS.map(option => (
          <button
            type="button"
            class="press-kit-segment"
            aria-pressed={value === option.value}
            onClick={() => apply(option.value === 'auto' ? undefined : option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
      <div class="press-kit-editor-color-custom">
        <ColorPicker
          label={`choisir ${name}`}
          value={value.startsWith('#') ? value : undefined}
          onPreview={preview}
          onCommit={commit}
          sample={() => cardColors(state.canvas)}
        />
        <input
          type="text"
          aria-label={`${name} en hexadécimal`}
          aria-invalid={invalid}
          aria-describedby={invalid ? errorId : undefined}
          placeholder={value === 'mixed' ? 'couleurs différentes' : '#RRGGBB'}
          value={hex}
          spellcheck={false}
          maxLength={7}
          onInput={event => {
            setHex(event.currentTarget.value)
            setInvalid(false)
          }}
          onChange={event => custom(event.currentTarget.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              event.preventDefault()
              custom(event.currentTarget.value)
            }
          }}
        />
      </div>
      {invalid && (
        <p id={errorId} role="alert">
          utiliser #RGB ou #RRGGBB
        </p>
      )}
    </div>
  )
}

export function SelectionColor({ state }: { state: EditorState }) {
  const selected = selection(state)
  const colorable = selected.filter(({ layer }) => layer.recolor !== false)
  const ids = colorable.map(({ layer }) => layer.id)
  // Highlighted text gets a second slot so ink and highlighter change apart.
  const highlighted = colorable.filter(({ layer }) => layer.highlight)
  if (!ids.length) return <p>la couleur de cette image est conservée</p>
  return (
    <>
      <ColorField
        state={state}
        keys={ids}
        title={highlighted.length ? 'texte' : 'couleur'}
        name={highlighted.length ? 'la couleur du texte' : 'la couleur de la sélection'}
      />
      {highlighted.length > 0 && (
        <ColorField
          state={state}
          keys={highlighted.map(({ layer }) => backgroundKey(layer.id))}
          title="surlignage"
          name="la couleur du surlignage"
        />
      )}
      {ids.length !== selected.length && <p>les images gardent leurs couleurs</p>}
    </>
  )
}
