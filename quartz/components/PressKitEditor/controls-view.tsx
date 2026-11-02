import { render } from 'preact'
import { useEffect, useRef } from 'preact/hooks'
import { buildSelect, type Select, type SelectOption } from '../controls/select'
import { SelectionColor } from './color-view'
import { history, resizeKey } from './gestures'
import {
  bounds,
  clamp,
  layers,
  placement,
  resizeHandles,
  resizeLimits,
  selection,
  selectLayer,
  type EditorState,
} from './layout'
import { edit, editBounds, positionOverlay, redrawFailed, resizeRailText } from './stage'

// Hosts the shared select inside the editor's Preact tree; it rebuilds when its options change.
function EditorSelect({
  id,
  label,
  options,
  selected,
  onSelect,
}: {
  id: string
  label: string
  options: SelectOption<string>[]
  selected: string
  onSelect: (value: string) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const select = useRef<Select<string>>()
  const latest = useRef(onSelect)
  latest.current = onSelect
  const shape = JSON.stringify(options)
  useEffect(() => {
    const control = buildSelect({
      id,
      label,
      groups: [{ options }],
      selected,
      onSelect: value => latest.current(value),
    })
    host.current?.replaceChildren(control.element)
    const unmount = control.mount()
    select.current = control
    return () => {
      unmount()
      control.element.remove()
      select.current = undefined
    }
  }, [id, label, shape])
  useEffect(() => select.current?.setSelected(selected), [selected])
  return <div ref={host} />
}

export const updateEditorView = (state: EditorState) => {
  if (state.signal.aborted) return
  state.overlay.hidden = !state.details.open || !state.available || !state.scene
  state.rulers.hidden = state.overlay.hidden || !state.showRulers
  state.rulerToggle?.setAttribute('aria-pressed', String(!state.rulers.hidden))
  state.canvas.parentElement?.toggleAttribute('data-editor-rulers', !state.rulers.hidden)
  const allLayers = layers(state)
  const selected = selection(state)
  const selBounds = bounds(selected)
  const limits = selected.length ? resizeLimits(state, selected) : undefined
  const label = selected.length === 1 ? selected[0].layer.label : `${selected.length} éléments`
  const allHidden = selected.every(({ placement: p }) => p.hidden)
  const doc = state.draft.value
  const bg = doc.background
  const number = (
    fieldLabel: string,
    value: number,
    min: number,
    max: number,
    apply: (value: number) => void,
  ) => (
    <label>
      {fieldLabel}
      <input
        type="number"
        value={Math.round(value * 10) / 10}
        min={min}
        max={max}
        step="1"
        onChange={event => {
          const next = event.currentTarget.valueAsNumber
          if (Number.isFinite(next)) apply(clamp(next, min, max))
        }}
      />
    </label>
  )
  const handles = (name: string) =>
    resizeHandles.map(({ axis, label: axisLabel }) => (
      <button
        key={axis}
        type="button"
        class="press-kit-editor-resize"
        data-editor-resize={axis}
        aria-label={`redimensionner ${axisLabel} : ${name}`}
        onKeyDown={event => resizeKey(state, event, axis)}
      />
    ))
  render(
    <>
      <div class="press-kit-segments press-kit-editor-toolbar" role="group" aria-label="édition">
        <button
          type="button"
          class="press-kit-segment"
          aria-label="annuler la modification"
          disabled={!state.draft.past.length}
          onClick={() => history(state, false)}
        >
          annuler
        </button>
        <button
          type="button"
          class="press-kit-segment"
          aria-label="rétablir la modification"
          disabled={!state.draft.future.length}
          onClick={() => history(state, true)}
        >
          rétablir
        </button>
        <button
          type="button"
          class="press-kit-segment"
          onClick={() =>
            edit(state, (layout, document) => {
              layout.placements = {}
              layout.order = []
              document.background = undefined
              document.metricGap = 0
              document.railTextScale = 1
              document.colors = {}
              document.title = undefined
            })
          }
        >
          réinitialiser
        </button>
        <button
          type="button"
          class="press-kit-segment"
          aria-label={bg ? 'changer l’image de fond' : 'ajouter une image de fond'}
          title={bg ? 'changer l’image de fond' : 'ajouter une image de fond'}
          disabled={state.loading}
          onClick={() => state.file.click()}
        >
          fond…
        </button>
        <button
          type="button"
          class="press-kit-segment press-kit-editor-rulers-toggle"
          aria-pressed={!state.rulers.hidden}
          title="afficher les règles en pixels autour de la carte"
          disabled={!state.available}
          onClick={() => {
            state.showRulers = !state.showRulers
            state.syncView()
          }}
        >
          règles
        </button>
      </div>
      {state.showRulers && (
        <div class="press-kit-editor-view">
          <output aria-label="position sur la carte" />
        </div>
      )}
      <label class="press-kit-editor-field">
        titre de l’image
        <input
          type="text"
          value={doc.title ?? ''}
          placeholder={state.defaultTitle()}
          maxLength={240}
          onChange={event => {
            const title = event.currentTarget.value.trim() || undefined
            if (title !== doc.title)
              edit(state, (_, document) => {
                document.title = title
              })
          }}
        />
      </label>
      {state.loading && <p role="status">chargement de l’image…</p>}
      {state.error && <p role="alert">{state.error}</p>}
      {redrawFailed(state) && <p role="alert">impossible de redimensionner le graphique</p>}
      {state.formatKey.endsWith(':chips') &&
        number('écart des cellules (px)', doc.metricGap, 0, 24, value =>
          edit(state, (_, document) => {
            document.metricGap = value
          }),
        )}
      {state.formatKey.endsWith(':rail') &&
        number('taille du texte (%)', doc.railTextScale * 100, 50, 250, value =>
          resizeRailText(state, value / 100),
        )}
      {bg && (
        <fieldset>
          <legend>{bg.name}</legend>
          <div class="press-kit-editor-field">
            <span aria-hidden="true">ajustement</span>
            <EditorSelect
              id="press-kit-editor-fit"
              label="ajustement"
              options={[
                { value: 'cover', label: 'remplir' },
                { value: 'contain', label: 'contenir' },
              ]}
              selected={bg.fit}
              onSelect={fit => {
                if (fit === 'cover' || fit === 'contain')
                  edit(state, (_, document) => {
                    if (document.background) document.background.fit = fit
                  })
              }}
            />
          </div>
          <div class="press-kit-editor-fields">
            {number('zoom (%)', bg.zoom * 100, 100, 400, value =>
              edit(state, (_, document) => {
                if (document.background) document.background.zoom = value / 100
              }),
            )}
            {number('opacité (%)', bg.opacity * 100, 0, 100, value =>
              edit(state, (_, document) => {
                if (document.background) document.background.opacity = value / 100
              }),
            )}
            {number('cadrage x (%)', bg.x * 100, 0, 100, value =>
              edit(state, (_, document) => {
                if (document.background) document.background.x = value / 100
              }),
            )}
            {number('cadrage y (%)', bg.y * 100, 0, 100, value =>
              edit(state, (_, document) => {
                if (document.background) document.background.y = value / 100
              }),
            )}
          </div>
          <button
            type="button"
            class="g-button"
            onClick={() =>
              edit(state, (_, document) => {
                document.background = undefined
              })
            }
          >
            retirer l’image
          </button>
        </fieldset>
      )}
      <div class="press-kit-editor-field">
        <span aria-hidden="true">élément</span>
        <EditorSelect
          id="press-kit-editor-layer"
          label="élément"
          options={[
            {
              value: '',
              label:
                selected.length > 1 ? `${selected.length} éléments sélectionnés` : 'sélectionner…',
            },
            ...allLayers.map(layer => ({
              value: layer.id,
              label: `${layer.label}${placement(state, layer).hidden ? ' (masqué)' : ''}`,
            })),
          ]}
          selected={selected.length === 1 ? selected[0].layer.id : ''}
          onSelect={value => {
            state.selected = new Set(value ? [value] : [])
            state.syncView()
          }}
        />
      </div>
      {selBounds && limits && (
        <fieldset>
          <legend>{label}</legend>
          <SelectionColor state={state} />
          <div class="press-kit-editor-fields">
            {number(
              'x (px)',
              selBounds.x,
              -(state.scene?.width ?? 1200),
              state.scene?.width ?? 1200,
              value => editBounds(state, { ...selBounds, x: value }),
            )}
            {number(
              'y (px)',
              selBounds.y,
              -(state.scene?.height ?? 1920),
              state.scene?.height ?? 1920,
              value => editBounds(state, { ...selBounds, y: value }),
            )}
            {number(
              'largeur (px)',
              selBounds.width,
              selBounds.width * limits.minX,
              selBounds.width * limits.maxX,
              value => editBounds(state, { ...selBounds, width: value }),
            )}
            {number(
              'hauteur (px)',
              selBounds.height,
              selBounds.height * limits.minY,
              selBounds.height * limits.maxY,
              value => editBounds(state, { ...selBounds, height: value }),
            )}
          </div>
          <div class="press-kit-editor-actions">
            <button
              type="button"
              class="g-button"
              aria-pressed={!allHidden}
              onClick={() =>
                edit(state, layout => {
                  for (const { layer, placement: p } of selected)
                    layout.placements[layer.id] = { ...p, hidden: !allHidden }
                })
              }
            >
              {allHidden ? 'afficher' : 'masquer'}
            </button>
            <button
              type="button"
              class="g-button"
              onClick={() =>
                edit(state, layout => {
                  layout.order = [
                    ...allLayers
                      .filter(layer => !state.selected.has(layer.id))
                      .map(layer => layer.id),
                    ...selected.map(({ layer }) => layer.id),
                  ]
                })
              }
            >
              devant
            </button>
            <button
              type="button"
              class="g-button"
              onClick={() =>
                edit(state, layout => {
                  layout.order = [
                    ...selected.map(({ layer }) => layer.id),
                    ...allLayers
                      .filter(layer => !state.selected.has(layer.id))
                      .map(layer => layer.id),
                  ]
                })
              }
            >
              derrière
            </button>
          </div>
        </fieldset>
      )}
    </>,
    state.controls,
  )
  render(
    <>
      {allLayers
        .filter(layer => !placement(state, layer).hidden)
        .map(layer => (
          <div
            key={layer.id}
            class={`press-kit-editor-layer${state.selected.has(layer.id) ? ' is-selected' : ''}`}
            data-editor-layer={layer.id}
          >
            <button
              type="button"
              class="press-kit-editor-select"
              aria-label={`déplacer ${layer.label}`}
              aria-pressed={state.selected.has(layer.id)}
              onClick={event => {
                // Pointer selection happens before dragging; keyboard and AT activate via click.
                if (event.detail) return
                selectLayer(state, layer.id, event.shiftKey || event.metaKey || event.ctrlKey)
                state.syncView()
              }}
            />
            {selected.length === 1 && state.selected.has(layer.id) && handles(layer.label)}
          </div>
        ))}
      {selected.length > 1 && !allHidden && (
        <>
          <div
            class="press-kit-editor-selection"
            data-editor-selection=""
            role="group"
            aria-label={`sélection de ${label}`}
          >
            {handles(`sélection de ${label}`)}
          </div>
          <button
            type="button"
            class="press-kit-editor-group-move"
            aria-label={`déplacer la sélection de ${label}`}
          >
            {label}
          </button>
        </>
      )}
      <div
        class="press-kit-editor-marquee"
        hidden={state.gesture?.kind !== 'marquee'}
        aria-hidden="true"
      />
    </>,
    state.overlay,
  )
  positionOverlay(state)
}
