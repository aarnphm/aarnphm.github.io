import type { TrailSeason } from '../../util/visitor-trail-weather'
import {
  isTrailHoliday,
  readTrailHolidayChoice,
  setTrailHolidayChoice,
  trailHolidayChoices,
  type TrailHolidayChoice,
} from '../../util/trail-holiday-choice'
import {
  isTrailSeason,
  readTrailSeason,
  setTrailSeason,
  trailSeasons,
} from '../../util/trail-season'
import { holidayLabels } from '../../util/visitor-holiday'

// The boulder reveals the existing cycles; returning to the forecast or calendar clears an override.
export function mountSeasonToggle(scene: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) return
  const boulder = document.createElement('button')
  boulder.type = 'button'
  boulder.className = 'trail-scene-toggle'
  boulder.setAttribute('aria-label', 'Scene controls')
  boulder.innerHTML = `<svg class="trail-scene-mark" viewBox="0 0 10 11" aria-hidden="true" focusable="false"><path class="trail-scene-mark-edge" d="m5 1 4 4-4 4-4-4Z" transform="translate(0 1)"/><path d="m5 1 4 4-4 4-4-4Z"/></svg>`
  boulder.hidden = true
  const panel = document.createElement('div')
  panel.className = 'trail-scene-controls'
  panel.popover = 'auto'
  panel.setAttribute('role', 'group')
  panel.setAttribute('aria-label', 'Season and holiday')
  boulder.popoverTargetElement = panel
  scene.append(boulder, panel)
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'trail-season-toggle'
  const seasonLabel = document.createElement('span')
  seasonLabel.textContent = 'season'
  const seasonValue = document.createElement('span')
  seasonValue.className = 'trail-scene-value'
  button.append(seasonLabel, seasonValue)
  const holidayButton = document.createElement('button')
  holidayButton.type = 'button'
  holidayButton.className = 'trail-holiday-toggle'
  const holidayLabel = document.createElement('span')
  holidayLabel.textContent = 'scene'
  const holidayValue = document.createElement('span')
  holidayValue.className = 'trail-scene-value'
  holidayButton.append(holidayLabel, holidayValue)
  panel.append(button, holidayButton)
  const nextHoliday = (): TrailHolidayChoice | null => {
    const chosen = readTrailHolidayChoice(document.defaultView)
    const index = chosen ? trailHolidayChoices.indexOf(chosen) : -1
    return trailHolidayChoices[index + 1] ?? null
  }
  const holidayName = (choice: TrailHolidayChoice | null) =>
    choice === null ? 'the calendar' : holidayLabels[choice]
  const labelHoliday = () => {
    const chosen = readTrailHolidayChoice(document.defaultView)
    const shown = isTrailHoliday(scene.dataset.holiday) ? scene.dataset.holiday : null
    holidayValue.textContent = shown ? holidayLabels[shown].toLowerCase() : 'no holiday'
    holidayButton.dataset.forced = String(!!chosen)
    const title = `Holiday: ${shown ? holidayLabels[shown] : 'none'}${chosen ? '' : ' (calendar)'}. Click for ${holidayName(nextHoliday())}`
    holidayButton.title = title
    holidayButton.setAttribute('aria-label', title)
  }
  const forecast = () => (isTrailSeason(scene.dataset.season) ? scene.dataset.season : null)
  const next = (): TrailSeason | null => {
    const forced = readTrailSeason(document.defaultView)
    const from = forced ?? forecast()
    const index = from ? trailSeasons.indexOf(from) : -1
    const season = trailSeasons[(index + 1) % trailSeasons.length]
    return season === (forecast() ?? trailSeasons[0]) ? null : season
  }
  const label = () => {
    const forced = readTrailSeason(document.defaultView)
    const shown = isTrailSeason(scene.dataset.foliageSeason)
      ? scene.dataset.foliageSeason
      : (forced ?? forecast())
    seasonValue.textContent = shown ?? '·'
    button.dataset.forced = String(!!forced)
    const title = `Season: ${shown ?? 'unknown'}${forced ? '' : ' (forecast)'}. Click for ${next() ?? 'the forecast'}`
    button.title = title
    button.setAttribute('aria-label', title)
  }
  const positionPanel = () => {
    if (!panel.matches(':popover-open')) return
    const anchor = boulder.getBoundingClientRect()
    const bounds = panel.getBoundingClientRect()
    panel.style.left = `${Math.max(8, Math.min(view.innerWidth - bounds.width - 8, anchor.left + anchor.width / 2 - bounds.width / 2))}px`
    const above = anchor.top - bounds.height - 4
    panel.style.top = `${Math.max(8, Math.min(view.innerHeight - bounds.height - 8, above >= 8 ? above : anchor.bottom + 4))}px`
  }
  // Prefer a low, central boulder. The hit area covers its painted bounds and stays usable on touch.
  const place = () => {
    const width = scene.clientWidth
    const height = scene.clientHeight
    let best: { x: number; y: number; rock: SVGGElement } | null = null
    let score = -Infinity
    for (const rock of scene.querySelectorAll<SVGGElement>(
      '.trail-understory[data-plant="boulder"]',
    )) {
      const x = Number(rock.dataset.groundX)
      const y = Number(rock.dataset.groundY)
      const r = Number(rock.dataset.solidRadius) || 6
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue
      const edge = Math.min(120, width * 0.1)
      if (x < edge || x > width - edge || y > height - 16 || y < height * 0.5) continue
      const fit = y - Math.abs(x - width / 2) * 0.25 + r * 2
      if (fit <= score) continue
      score = fit
      best = { x, y, rock }
    }
    if (best) {
      const bounds = best.rock.getBBox()
      const targetWidth = Math.max(44, bounds.width)
      const targetHeight = Math.max(44, bounds.height)
      boulder.style.width = `${targetWidth}px`
      boulder.style.height = `${targetHeight}px`
      boulder.style.setProperty('--scene-mark-offset', `${Math.round(bounds.height * 0.15)}px`)
      boulder.style.left = `${best.x + bounds.x + (bounds.width - targetWidth) / 2}px`
      boulder.style.top = `${best.y + bounds.y + (bounds.height - targetHeight) / 2}px`
      boulder.hidden = false
    } else {
      boulder.hidden = true
      panel.hidePopover()
    }
    label()
    labelHoliday()
    positionPanel()
  }
  button.addEventListener('click', () => setTrailSeason(document, next()), { signal })
  holidayButton.addEventListener('click', () => setTrailHolidayChoice(document, nextHoliday()), {
    signal,
  })
  scene.addEventListener('trail:reflow', place, { signal })
  scene.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape' || !panel.matches(':popover-open')) return
      event.preventDefault()
      event.stopPropagation()
      panel.hidePopover()
      boulder.focus({ preventScroll: true })
    },
    { signal },
  )
  panel.addEventListener('toggle', positionPanel, { signal })
  view.addEventListener('scroll', positionPanel, { signal, passive: true })
  const observer = new MutationObserver(records => {
    if (scene.dataset.trailVisible === 'false') panel.hidePopover()
    if (records.some(record => record.attributeName === 'data-trail-paint-ready')) place()
    else {
      label()
      labelHoliday()
      positionPanel()
    }
  })
  observer.observe(scene, {
    attributes: true,
    attributeFilter: [
      'data-trail-paint-ready',
      'data-trail-visible',
      'data-season',
      'data-foliage-season',
      'data-season-override',
      'data-holiday',
      'data-holiday-selection',
    ],
  })
  signal.addEventListener('abort', () => {
    observer.disconnect()
    panel.remove()
    boulder.remove()
  })
  if (scene.dataset.trailPaintReady === 'true') place()
}
