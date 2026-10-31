import type { CardModel } from '../scripts/press-kit-card'

const Include = ({
  name,
  label,
  shortLabel,
  pressed = false,
  tab,
  pageOnly,
  data,
}: {
  name: keyof CardModel['show']
  label: string
  shortLabel?: string
  pressed?: boolean
  tab?: 'note' | 'quote'
  pageOnly?: boolean
  data?: boolean
}) => (
  <button
    type="button"
    class={data ? 'press-kit-data-toggle' : 'press-kit-segment'}
    name={name}
    data-include={name}
    data-for={tab}
    data-page-only={pageOnly ? '' : undefined}
    data-tractatus-only={name === 'tractatus' ? '' : undefined}
    aria-pressed={pressed}
    aria-label={shortLabel ? label : undefined}
    title={shortLabel ? label : undefined}
  >
    {data ? <span class="press-kit-data-label">{shortLabel ?? label}</span> : label}
  </button>
)

const POWER_WINDOWS = [
  ['30', '30 s'],
  ['300', '5 min'],
] as const

const POWER_SCALES = [
  ['linear', 'linéaire'],
  ['log', 'logarithmique'],
] as const

export const TrainingControls = () => (
  <>
    <fieldset class="press-kit-field" data-activity-only data-for="note">
      <legend>inclure</legend>
      <div class="press-kit-data-grid press-kit-data-grid--compact">
        <Include name="title" label="titre" pressed data />
        <Include name="date" label="date" pressed data />
        <Include name="metrics" label="mesures" pressed data />
        <div class="press-kit-data-fields" data-training-options>
          <Include name="map" label="carte du parcours" shortLabel="carte" pressed data />
          <Include name="distance" label="distance" pressed data />
          <Include name="time" label="durée" pressed data />
          <Include name="pace" label="allure / vitesse" shortLabel="allure / vit." pressed data />
          <Include name="elevation" label="dénivelé" shortLabel="D+" pressed data />
          <Include name="heartRate" label="fréquence cardiaque" shortLabel="FC" pressed data />
          <Include name="power" label="puissance" shortLabel="puiss." pressed data />
        </div>
      </div>
    </fieldset>

    <fieldset class="press-kit-field" data-day-only data-for="note">
      <legend>blocs de la journée</legend>
      <div class="press-kit-data-grid press-kit-block-list">
        <Include name="entry" label="entrée" data />
        <div class="press-kit-entry-display" />
      </div>
      <div class="press-kit-data-grid press-kit-block-list press-kit-day-blocks" />
    </fieldset>

    <fieldset class="press-kit-field" data-training-options>
      <legend>superposition</legend>
      <div class="press-kit-overlay" />
      <div class="press-kit-data-grid press-kit-overlay-background">
        <Include name="transparent" label="fond transparent" data />
        <button
          type="button"
          class="press-kit-data-toggle"
          data-editor-rulers
          aria-pressed="false"
          disabled
        >
          <span class="press-kit-data-label">règles</span>
        </button>
      </div>
    </fieldset>

    <ChartControls />
  </>
)

export const NoteIncludes = () => (
  <fieldset class="press-kit-field" data-note-only>
    <legend>inclure</legend>
    <div class="press-kit-segments press-kit-includes">
      <Include name="description" label="description" tab="note" pageOnly pressed />
      <Include name="excerpt" label="extrait" tab="note" pageOnly />
      <Include name="meta" label="méta" tab="note" pressed />
      <Include name="art" label="illustration" />
      <Include name="highlight" label="surligner" tab="quote" />
      <Include name="section" label="section" tab="quote" pressed />
      <Include name="tractatus" label="tractatus" tab="quote" />
    </div>
  </fieldset>
)

const ChartControls = () => (
  <fieldset class="press-kit-field" data-training-options>
    <legend>graphiques</legend>
    <div class="press-kit-data-grid press-kit-data-grid--compact">
      <Include name="cyclingPower" label="puissance vélo" shortLabel="puiss. vélo" data />
      <Include name="powerCurve" label="courbe de puissance" shortLabel="courbe puiss." data />
      <Include name="cyclingLaps" label="analyse des tours vélo" shortLabel="tours vélo" data />
      <Include name="runPace" label="d d’allure" data />
      <Include name="runHeartRate" label="d de FC" data />
      <Include name="runLaps" label="analyse des tours" shortLabel="tours" data />
    </div>
    <div class="press-kit-chart-setting press-kit-power-window" hidden>
      <span class="press-kit-legend" id="press-kit-power-window-label">
        moyenne de puissance
      </span>
      <div
        class="press-kit-segments"
        role="radiogroup"
        aria-labelledby="press-kit-power-window-label"
      >
        {POWER_WINDOWS.map(([value, label]) => (
          <label class="press-kit-segment">
            <input type="radio" name="press-kit-power-window" value={value} />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </div>
    <div class="press-kit-chart-setting press-kit-power-scale" hidden>
      <span class="press-kit-legend" id="press-kit-power-scale-label">
        échelle de la courbe
      </span>
      <div
        class="press-kit-segments"
        role="radiogroup"
        aria-labelledby="press-kit-power-scale-label"
      >
        {POWER_SCALES.map(([value, label]) => (
          <label class="press-kit-segment">
            <input type="radio" name="press-kit-power-scale" value={value} />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </div>
    <div class="press-kit-chart-setting press-kit-power-unit" hidden>
      <span class="press-kit-legend" id="press-kit-power-unit-label">
        unité de la courbe
      </span>
      <div
        class="press-kit-segments"
        role="radiogroup"
        aria-labelledby="press-kit-power-unit-label"
      >
        <label class="press-kit-segment">
          <input type="radio" name="press-kit-power-unit" value="watts" />
          <span>W</span>
        </label>
        <label class="press-kit-segment">
          <input
            type="radio"
            name="press-kit-power-unit"
            value="relative"
            aria-describedby="press-kit-power-weight"
          />
          <span>W/kg</span>
        </label>
      </div>
      <span class="press-kit-hint" id="press-kit-power-weight" data-power-weight hidden />
    </div>
  </fieldset>
)

export const ExportActions = () => (
  <div class="press-kit-actions">
    <button type="button" class="g-button" data-action="download">
      télécharger
    </button>
    <button type="button" class="g-button" data-action="copy-image">
      copier l'image
    </button>
    <button type="button" class="g-button" data-action="share-image" hidden>
      partager
    </button>
  </div>
)
