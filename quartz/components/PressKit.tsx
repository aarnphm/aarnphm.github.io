import { render } from 'preact-render-to-string'
import type { CardModel } from './scripts/press-kit-card'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
import { simplifySlug } from '../util/path'
import { formatDate, getDate } from './Date'
// @ts-ignore
import script from './scripts/press-kit.inline'
import style from './styles/pressKit.scss'

interface Options {
  author: string
  handle: string
}

const defaultOptions: Options = { author: 'Aaron Pham', handle: 'aarnphm' }

/** Serialised into the dialog; `press-kit.inline.ts` reads it as `PressKitPage`. */
export interface PressKitPage {
  slug: string
  url: string
  site: string
  crumbs: string[]
  date?: string
  iso?: string
  reading?: string
  tags: string[]
  author: string
  handle: string
  colors: { light: Record<string, string>; dark: Record<string, string> }
}

const FORMAT_OPTIONS = [
  ['story', 'story', '9:16'],
  ['portrait', 'portrait', '4:5'],
  ['square', 'carré', '1:1'],
  ['landscape', 'paysage', '1.91:1'],
] as const

const THEME_OPTIONS = [
  ['papier', 'papier'],
  ['encre', 'encre'],
  ['rose', 'rose'],
  ['sauge', 'sauge'],
] as const

// Brand marks from simple-icons (24-unit grid), drawn in currentColor to sit with the ink controls.
const INTENTS = [
  [
    'x',
    'x',
    'M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z',
  ],
  [
    'bluesky',
    'bluesky',
    'M12 10.8c-1.087-2.114-4.046-6.053-6.798-7.995C2.566.944 1.561 1.266.902 1.565.139 1.908 0 3.08 0 3.768c0 .69.378 5.65.624 6.479.815 2.736 3.713 3.66 6.383 3.364.136-.02.275-.039.415-.056-.138.022-.276.04-.415.056-3.912.58-7.387 2.005-2.83 7.078 5.013 5.19 6.87-1.113 7.823-4.308.953 3.195 2.05 9.271 7.733 4.308 4.267-4.308 1.172-6.498-2.74-7.078a8.741 8.741 0 0 1-.415-.056c.14.017.279.036.415.056 2.67.297 5.568-.628 6.383-3.364.246-.828.624-5.79.624-6.478 0-.69-.139-1.861-.902-2.206-.659-.298-1.664-.62-4.3 1.24C16.046 4.748 13.087 8.687 12 10.8Z',
  ],
  [
    'threads',
    'threads',
    'M12.186 24h-.007c-3.581-.024-6.334-1.205-8.184-3.509C2.35 18.44 1.5 15.586 1.472 12.01v-.017c.03-3.579.879-6.43 2.525-8.482C5.845 1.205 8.6.024 12.18 0h.014c2.746.02 5.043.725 6.826 2.098 1.677 1.29 2.858 3.13 3.509 5.467l-2.04.569c-1.104-3.96-3.898-5.984-8.304-6.015-2.91.022-5.11.936-6.54 2.717C4.307 6.504 3.616 8.914 3.589 12c.027 3.086.718 5.496 2.057 7.164 1.43 1.783 3.631 2.698 6.54 2.717 2.623-.02 4.358-.631 5.8-2.045 1.647-1.613 1.618-3.593 1.09-4.798-.31-.71-.873-1.3-1.634-1.75-.192 1.352-.622 2.446-1.284 3.272-.886 1.102-2.14 1.704-3.73 1.79-1.202.065-2.361-.218-3.259-.801-1.063-.689-1.685-1.74-1.752-2.964-.065-1.19.408-2.285 1.33-3.082.88-.76 2.119-1.207 3.583-1.291a13.853 13.853 0 0 1 3.02.142c-.126-.742-.375-1.332-.75-1.757-.513-.586-1.308-.883-2.359-.89h-.029c-.844 0-1.992.232-2.721 1.32L7.734 7.847c.98-1.454 2.568-2.256 4.478-2.256h.044c3.194.02 5.097 1.975 5.287 5.388.108.046.216.094.321.142 1.49.7 2.58 1.761 3.154 3.07.797 1.82.871 4.79-1.548 7.158-1.85 1.81-4.094 2.628-7.277 2.65Zm1.003-11.69c-.242 0-.487.007-.739.021-1.836.103-2.98.946-2.916 2.143.067 1.256 1.452 1.839 2.784 1.767 1.224-.065 2.818-.543 3.086-3.71a10.5 10.5 0 0 0-2.215-.221z',
  ],
  [
    'linkedin',
    'linkedin',
    'M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z',
  ],
  [
    'hn',
    'hacker news',
    'M0 24V0h24v24H0zM6.951 5.896l4.112 7.708v5.064h1.583v-4.972l4.148-7.799h-1.749l-2.457 4.875c-.372.745-.688 1.434-.688 1.434s-.297-.708-.651-1.434L8.831 5.896h-1.88z',
  ],
] as const

const Glyph = ({ d, stroke }: { d: string; stroke?: boolean }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    aria-hidden="true"
    {...(stroke
      ? {
          fill: 'none',
          stroke: 'currentColor',
          'stroke-width': 1.75,
          'stroke-linecap': 'round',
          'stroke-linejoin': 'round',
        }
      : { fill: 'currentColor' })}
  >
    <path d={d} />
  </svg>
)

const MAIL = 'M3 5.5h18v13H3zM3.5 6l8.5 7 8.5-7'
const SHARE = 'M12 3v12M7.5 7.5 12 3l4.5 4.5M5 11v10h14V11'

const CloseIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.5"
    stroke-linecap="round"
    aria-hidden="true"
  >
    <path d="M18 6 6 18M6 6l12 12" />
  </svg>
)

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
    aria-pressed={pressed}
    aria-label={shortLabel ? label : undefined}
    title={shortLabel ? label : undefined}
  >
    {data ? <span class="press-kit-data-label">{shortLabel ?? label}</span> : label}
  </button>
)

const SHARE_URL_TOKEN = '__PRESS_KIT_SHARE_URL__'
const MAILTO_URL_TOKEN = '__PRESS_KIT_MAILTO_URL__'
const PAGE_JSON_TOKEN = '__PRESS_KIT_PAGE_JSON__'
const pressKitTemplateBySite = new Map<string, string>()

const encodeAttr = (value: string): string =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')

function getPressKitInnerTemplate(site: string): string {
  let cached = pressKitTemplateBySite.get(site)
  if (cached !== undefined) return cached
  cached = render(
    <>
      <header class="press-kit-header">
        <span id="press-kit-title">kit de presse</span>
        <div class="press-kit-tabs" role="tablist" aria-label="kit de presse">
          {(
            [
              ['note', 'note', '1'],
              ['quote', 'citation', '2'],
              ['links', 'liens', '3'],
            ] as const
          ).map(([tab, label, key], index) => (
            <button
              type="button"
              role="tab"
              id={`press-kit-tab-${tab}`}
              class="press-kit-tab"
              data-tab={tab}
              aria-selected={index === 0 ? 'true' : 'false'}
              aria-controls="press-kit-panel"
              aria-keyshortcuts={key}
              tabindex={index === 0 ? 0 : -1}
            >
              {label}
            </button>
          ))}
        </div>
        <button type="button" class="press-kit-close g-icon-button" aria-label="fermer">
          <CloseIcon />
        </button>
      </header>

      <div
        class="press-kit-body"
        id="press-kit-panel"
        role="tabpanel"
        aria-labelledby="press-kit-tab-note"
      >
        <figure class="press-kit-stage">
          <div class="press-kit-frame">
            <canvas class="press-kit-canvas" role="img" aria-label="aperçu de la carte" />
            <div class="press-kit-unfurl" hidden>
              <img class="press-kit-unfurl-image" alt="" decoding="async" data-ignore-popup />
              <div class="press-kit-unfurl-text">
                <span class="press-kit-unfurl-site">{site}</span>
                <span class="press-kit-unfurl-title" />
              </div>
            </div>
          </div>
          <figcaption class="press-kit-caption" aria-live="polite" />
        </figure>

        <div class="press-kit-controls">
          <div class="press-kit-group" data-for="image">
            <fieldset class="press-kit-field">
              <legend>format</legend>
              <div class="press-kit-segments">
                {FORMAT_OPTIONS.map(([value, label, ratio]) => (
                  <label class="press-kit-segment">
                    <input type="radio" name="press-kit-format" value={value} />
                    <span>{label}</span>
                    <small>{ratio}</small>
                  </label>
                ))}
              </div>
            </fieldset>

            <fieldset class="press-kit-field">
              <legend>thème</legend>
              <div class="press-kit-segments">
                {THEME_OPTIONS.map(([value, label]) => (
                  <label class="press-kit-segment press-kit-swatch" data-theme={value}>
                    <input type="radio" name="press-kit-theme" value={value} />
                    <i aria-hidden="true" />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <fieldset class="press-kit-field" data-activity-only data-for="note">
              <legend>langue</legend>
              <div class="press-kit-segments press-kit-includes press-kit-language">
                {[
                  ['en', 'English'],
                  ['fr', 'Français'],
                ].map(([value, label]) => (
                  <label class="press-kit-segment">
                    <input type="radio" name="press-kit-language" value={value} />
                    <span>{label}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div class="press-kit-field" data-for="quote">
              <label class="press-kit-legend" for="press-kit-quote">
                texte
              </label>
              <textarea
                id="press-kit-quote"
                class="press-kit-textarea"
                rows={5}
                spellcheck={false}
                aria-describedby="press-kit-quote-hint"
              />
              <p class="press-kit-hint" id="press-kit-quote-hint">
                sélectionnez un passage pour le citer. <code>*italique*</code> <code>**gras**</code>{' '}
                <code>`code`</code> <code>$tex$</code>
              </p>
            </div>

            <div class="press-kit-field" data-for="quote">
              <label class="press-kit-legend" for="press-kit-attribution">
                attribution
              </label>
              <input
                id="press-kit-attribution"
                class="press-kit-input"
                type="text"
                spellcheck={false}
                aria-describedby="press-kit-attribution-hint"
              />
              <p class="press-kit-hint" id="press-kit-attribution-hint">
                laisser vide pour utiliser le titre d’origine.
              </p>
            </div>

            <fieldset class="press-kit-field" data-note-only>
              <legend>inclure</legend>
              <div class="press-kit-segments press-kit-includes">
                <Include name="description" label="description" tab="note" pageOnly pressed />
                <Include name="excerpt" label="extrait" tab="note" pageOnly />
                <Include name="meta" label="méta" tab="note" pressed />
                <Include name="art" label="illustration" />
                <Include name="highlight" label="surligner" tab="quote" />
                <Include name="section" label="section" tab="quote" pressed />
              </div>
            </fieldset>

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
                  <Include
                    name="pace"
                    label="allure / vitesse"
                    shortLabel="allure / vit."
                    pressed
                    data
                  />
                  <Include name="elevation" label="dénivelé" shortLabel="D+" pressed data />
                  <Include
                    name="heartRate"
                    label="fréquence cardiaque"
                    shortLabel="FC"
                    pressed
                    data
                  />
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

            <fieldset class="press-kit-field" data-training-options>
              <legend>graphiques</legend>
              <div class="press-kit-data-grid press-kit-data-grid--compact">
                <Include name="cyclingPower" label="puissance vélo" shortLabel="puiss. vélo" data />
                <Include
                  name="powerCurve"
                  label="courbe de puissance"
                  shortLabel="courbe puiss."
                  data
                />
                <Include
                  name="cyclingLaps"
                  label="analyse des tours vélo"
                  shortLabel="tours vélo"
                  data
                />
                <Include name="runPace" label="d d’allure" data />
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
                  {[
                    ['30', '30 s'],
                    ['300', '5 min'],
                  ].map(([value, label]) => (
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
                  {[
                    ['linear', 'linéaire'],
                    ['log', 'logarithmique'],
                  ].map(([value, label]) => (
                    <label class="press-kit-segment">
                      <input type="radio" name="press-kit-power-scale" value={value} />
                      <span>{label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </fieldset>

            <div class="press-kit-metrics" data-training-options />

            <div class="press-kit-actions">
              <button type="button" class="g-button" data-action="download">
                télécharger
              </button>
              <button type="button" class="g-button" data-action="copy-image">
                copier l'image
              </button>
              <button type="button" class="g-button" data-action="share-image" hidden>
                partager…
              </button>
            </div>
          </div>

          <div class="press-kit-group" data-for="links">
            <div class="press-kit-field">
              <span class="press-kit-legend" id="press-kit-anchor-label">
                ancre
              </span>
              <div class="press-kit-anchor" />
            </div>

            <div class="press-kit-field">
              <label class="press-kit-legend" for="press-kit-url">
                lien
              </label>
              <div class="press-kit-url-row">
                <input id="press-kit-url" class="press-kit-input" type="url" readonly />
                <button type="button" class="g-button" data-copy="url">
                  copier
                </button>
              </div>
            </div>

            <div class="press-kit-field">
              <span class="press-kit-legend">copier comme</span>
              <div class="press-kit-chips">
                <button type="button" class="g-button" data-copy="markdown">
                  markdown
                </button>
                <button type="button" class="g-button" data-copy="wikilink">
                  wikilink
                </button>
                <button type="button" class="g-button" data-copy="bibtex" data-citation-only>
                  bibtex
                </button>
                <button type="button" class="g-button" data-copy="citation" data-citation-only>
                  citation
                </button>
              </div>
            </div>

            <div class="press-kit-field">
              <label class="press-kit-legend" for="press-kit-note">
                note
              </label>
              <textarea
                id="press-kit-note"
                class="press-kit-textarea"
                rows={4}
                aria-describedby="press-kit-note-count"
              />
              <p class="press-kit-hint" id="press-kit-note-count" />
            </div>

            <div class="press-kit-field">
              <span class="press-kit-legend">publier sur</span>
              <div class="press-kit-chips">
                {INTENTS.map(([intent, label, d]) => (
                  <a
                    class="press-kit-chip press-kit-intent"
                    data-intent={intent}
                    href={SHARE_URL_TOKEN}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={label}
                    title={label}
                    data-no-popover
                  >
                    <Glyph d={d} />
                  </a>
                ))}
                <a
                  class="press-kit-chip press-kit-intent"
                  data-intent="email"
                  href={MAILTO_URL_TOKEN}
                  aria-label="courriel"
                  title="courriel"
                  data-no-popover
                >
                  <Glyph d={MAIL} stroke />
                </a>
                <button
                  type="button"
                  class="g-button press-kit-intent"
                  data-action="share-link"
                  aria-label="partager…"
                  title="partager…"
                  hidden
                >
                  <Glyph d={SHARE} stroke />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <p class="press-kit-help" role="status" />
      <script
        type="application/json"
        class="press-kit-data"
        dangerouslySetInnerHTML={{ __html: PAGE_JSON_TOKEN }}
      />
    </>,
  )
  pressKitTemplateBySite.set(site, cached)
  return cached
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const PressKit: QuartzComponent = ({ cfg, fileData, displayClass }: QuartzComponentProps) => {
    if (fileData.frontmatter?.protected === true || !fileData.slug) return null

    const slug = fileData.slug
    const simple = simplifySlug(slug)
    const site = cfg.baseUrl ?? 'aarnphm.xyz'
    const path = simple === '/' ? '' : simple
    const created = getDate(cfg, fileData)
    const minutes = Math.ceil(fileData.readingTime?.minutes ?? 0)
    const tags = Array.isArray(fileData.frontmatter?.tags)
      ? (fileData.frontmatter.tags as unknown[]).map(String).slice(0, 4)
      : []

    const shareUrl = `https://${site}/${path}`
    const mailtoUrl = `mailto:?body=${encodeURIComponent(shareUrl)}`
    const page: PressKitPage = {
      slug: path,
      url: shareUrl,
      site,
      crumbs: [
        site,
        ...path
          .split('/')
          .filter(Boolean)
          .map(part => part.toLowerCase()),
      ],
      date: created ? formatDate(created, cfg.locale) : undefined,
      iso: created?.toISOString(),
      reading: minutes > 0 ? `${minutes} min` : undefined,
      tags,
      author: opts.author,
      handle: opts.handle,
      colors: { light: { ...cfg.theme.colors.lightMode }, dark: { ...cfg.theme.colors.darkMode } },
    }
    const pageJson = JSON.stringify(page).replace(/</g, '\\u003c')
    const innerHtml = getPressKitInnerTemplate(site)
      .replaceAll(SHARE_URL_TOKEN, encodeAttr(shareUrl))
      .replace(MAILTO_URL_TOKEN, encodeAttr(mailtoUrl))
      .replace(PAGE_JSON_TOKEN, () => pageJson)

    return (
      <dialog
        class={classNames(displayClass, 'press-kit')}
        aria-labelledby="press-kit-title"
        data-tab="note"
        tabindex={-1}
        lang="fr"
        dangerouslySetInnerHTML={{ __html: innerHtml }}
      />
    )
  }

  PressKit.css = style
  PressKit.afterDOMLoaded = script

  return PressKit
}) satisfies QuartzComponentConstructor
