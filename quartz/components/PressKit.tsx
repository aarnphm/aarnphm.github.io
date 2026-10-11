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

const INTENTS = [
  ['x', 'x'],
  ['bluesky', 'bluesky'],
  ['threads', 'threads'],
  ['linkedin', 'linkedin'],
  ['hn', 'hacker news'],
  ['email', 'courriel'],
] as const

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

const Check = ({ name, label, checked }: { name: string; label: string; checked?: boolean }) => (
  <label class="press-kit-check">
    <input type="checkbox" name={name} checked={checked} />
    <span>{label}</span>
  </label>
)

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

    const page: PressKitPage = {
      slug: path,
      url: `https://${site}/${path}`,
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

    return (
      <dialog
        class={classNames(displayClass, 'press-kit')}
        aria-labelledby="press-kit-title"
        data-tab="note"
        tabindex={-1}
        lang="fr"
      >
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
          <button type="button" class="press-kit-close" aria-label="fermer">
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
                <img class="press-kit-unfurl-image" alt="" decoding="async" />
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
                  sélectionnez un passage pour le citer. <code>*italique*</code>{' '}
                  <code>**gras**</code> <code>`code`</code> <code>$tex$</code>
                </p>
              </div>

              <fieldset class="press-kit-field">
                <legend>inclure</legend>
                <div class="press-kit-checks">
                  <span data-for="note" data-page-only>
                    <Check name="description" label="description" checked />
                  </span>
                  <span data-for="note" data-page-only>
                    <Check name="excerpt" label="extrait" />
                  </span>
                  <span data-for="note">
                    <Check name="meta" label="méta" checked />
                  </span>
                  <span data-for="quote">
                    <Check name="highlight" label="surligner" />
                  </span>
                  <span data-for="quote">
                    <Check name="section" label="section" checked />
                  </span>
                </div>
              </fieldset>

              <div class="press-kit-actions">
                <button type="button" class="press-kit-action" data-action="download">
                  télécharger
                </button>
                <button type="button" class="press-kit-action" data-action="copy-image">
                  copier l'image
                </button>
                <button type="button" class="press-kit-action" data-action="share-image" hidden>
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
                  <button type="button" class="press-kit-action" data-copy="url">
                    copier
                  </button>
                </div>
              </div>

              <div class="press-kit-field">
                <span class="press-kit-legend">copier comme</span>
                <div class="press-kit-chips">
                  <button type="button" class="press-kit-chip" data-copy="markdown">
                    markdown
                  </button>
                  <button type="button" class="press-kit-chip" data-copy="wikilink">
                    wikilink
                  </button>
                  <button type="button" class="press-kit-chip" data-copy="bibtex">
                    bibtex
                  </button>
                  <button type="button" class="press-kit-chip" data-copy="citation">
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
                  {INTENTS.map(([intent, label]) => (
                    <a
                      class="press-kit-chip"
                      data-intent={intent}
                      href={`https://${site}/${path}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      data-no-popover
                    >
                      {label}
                    </a>
                  ))}
                  <button type="button" class="press-kit-chip" data-action="share-link" hidden>
                    partager…
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
          dangerouslySetInnerHTML={{ __html: JSON.stringify(page).replace(/</g, '\\u003c') }}
        />
      </dialog>
    )
  }

  PressKit.css = style
  PressKit.afterDOMLoaded = script

  return PressKit
}) satisfies QuartzComponentConstructor
