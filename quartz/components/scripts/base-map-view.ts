import { escapeHTML } from '../../util/escape'
import { createPopupContent, markerColor, type BaseMapData, type MarkerData } from './base-map-data'

type MapControl = 'satellite' | 'elevation' | '3d'
interface MapViewActions {
  select(marker: MarkerData): void
  clear(): void
  filter(markers: MarkerData[]): void
  fit(): void
  control(control: MapControl): void
}

const icons = {
  satellite: ['m12 3 10 5-10 5L2 8Z', 'm2 12 10 5 10-5', 'm2 16 10 5 10-5'],
  elevation: ['m3 20 7-16 7 16H3Z', 'm7.5 9.7 2.5 2.8 2.5-2.8M14 13l3-6 5 13h-5'],
  '3d': ['M12 3 20 7.5 12 12 4 7.5 12 3Z', 'M4 7.5v9L12 21v-9', 'M20 7.5v9L12 21'],
  fit: ['M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5', 'M12 8v8M8 12h8'],
  list: ['M9 5h12M9 12h12M9 19h12M3 5h1M3 12h1M3 19h1'],
}

function icon(paths: string[]): string {
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths.map(path => `<path d="${path}"/>`).join('')}</svg>`
}

function distanceKm(a: MarkerData, b: MarkerData): number {
  const rad = Math.PI / 180
  const dLat = (b.lat - a.lat) * rad
  const dLon = (b.lon - a.lon) * rad
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2
  return 6371 * 2 * Math.asin(Math.sqrt(Math.min(1, h)))
}

export function mountBaseMapView(
  container: HTMLElement,
  data: BaseMapData,
  actions: MapViewActions,
  signal: AbortSignal,
) {
  container.innerHTML = `<div class="base-map-canvas"></div>
    <div class="base-map-tools" role="group" aria-label="Map controls">
      <button type="button" data-map-control="list" aria-label="Show saved spots" title="Show saved spots" aria-expanded="true">${icon(icons.list)}</button>
      <button type="button" data-map-control="fit" aria-label="Fit matching spots" title="Fit matching spots" disabled>${icon(icons.fit)}</button>
      <button type="button" data-map-control="elevation" aria-label="Elevation contours and hill shading" title="Elevation contours and hill shading" aria-pressed="false" disabled>${icon(icons.elevation)}</button>
      <button type="button" data-map-control="3d" aria-label="3D terrain and buildings" title="3D terrain and buildings" aria-pressed="false" disabled>${icon(icons['3d'])}</button>
      <button type="button" data-map-control="satellite" aria-label="Satellite" title="Satellite" aria-pressed="false" disabled>${icon(icons.satellite)}</button>
    </div>
    <aside class="base-map-browser" aria-label="Saved spots">
      <div class="base-map-browser-head"><span class="base-map-kicker">collected places</span><h2>Find a spot.</h2><span class="base-map-count" role="status"></span></div>
      <div class="base-map-categories" role="group" aria-label="Filter spots by category"></div>
      <div class="base-map-selection" hidden></div>
      <div class="base-map-spot-list" aria-label="Matching spots"></div>
    </aside>`
  const canvas = container.querySelector<HTMLElement>('.base-map-canvas')
  const browser = container.querySelector<HTMLElement>('.base-map-browser')
  const categoryList = container.querySelector<HTMLElement>('.base-map-categories')
  const list = container.querySelector<HTMLElement>('.base-map-spot-list')
  const selection = container.querySelector<HTMLElement>('.base-map-selection')
  const count = container.querySelector<HTMLElement>('.base-map-count')
  const search =
    container.closest('.base-embed')?.querySelector<HTMLInputElement>('input.base-search-input') ??
    document.querySelector<HTMLInputElement>('#base-search-input')
  let category: string | undefined
  let selected: MarkerData | undefined
  let filtered = data.markers
  const categoryCounts = new Map<string, number>()
  for (const marker of data.markers) {
    for (const item of marker.categories ?? [])
      categoryCounts.set(item, (categoryCounts.get(item) ?? 0) + 1)
  }
  const categoryEntries = [...categoryCounts].sort((a, b) => b[1] - a[1])
  if (categoryList) {
    categoryList.innerHTML =
      `<button type="button" data-category="" aria-pressed="true"><span class="base-map-category-label">all</span> <span class="base-map-category-count">${data.markers.length}</span></button>` +
      categoryEntries
        .map(
          ([name, total]) =>
            `<button type="button" data-category="${escapeHTML(name)}" aria-pressed="false"><span class="base-map-swatch" style="background:${markerColor({ lat: 0, lon: 0, title: '', slug: data.currentSlug, categories: [name], popupFields: {} })}"></span><span class="base-map-category-label">${escapeHTML(name)}</span> <span class="base-map-category-count">${total}</span></button>`,
        )
        .join('')
  }

  function show(marker: MarkerData, focus = false): void {
    selected = marker
    if (!selection) return
    selection.hidden = false
    const nearby = data.markers
      .filter(item => item.slug !== marker.slug)
      .map(item => ({ item, distance: distanceKm(marker, item) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 3)
    selection.innerHTML =
      `<button type="button" class="base-map-selection-close" aria-label="Close selected spot">×</button>` +
      createPopupContent(marker, data.currentSlug, data.properties) +
      (nearby.length
        ? `<div class="base-map-nearby"><span class="base-map-kicker">nearby · straight-line distance</span>${nearby.map(({ item, distance }) => `<button type="button" data-spot="${escapeHTML(item.slug)}"><span>${escapeHTML(item.title)}</span><span>${distance < 1 ? `${Math.round(distance * 1000)} m` : `${distance.toFixed(1)} km`}</span></button>`).join('')}</div>`
        : '')
    for (const button of container.querySelectorAll<HTMLButtonElement>(
      '.base-map-spot-list [data-spot]',
    )) {
      button.setAttribute('aria-pressed', String(button.dataset.spot === marker.slug))
    }
    if (browser) {
      browser.hidden = false
      container.classList.remove('base-map--folded')
      container.querySelector('[data-map-control="list"]')?.setAttribute('aria-expanded', 'true')
      browser.scrollTo({
        top:
          canvas && browser.offsetTop >= canvas.offsetHeight
            ? selection.offsetTop - browser.offsetTop
            : 0,
      })
    }
    if (focus)
      selection
        .querySelector<HTMLAnchorElement>('.base-map-popup-title')
        ?.focus({ preventScroll: true })
  }

  function update(): void {
    const query = search?.value.toLowerCase().trim() ?? ''
    filtered = data.markers.filter(
      marker =>
        (!category || marker.categories?.includes(category)) &&
        [
          marker.title,
          marker.description,
          marker.address,
          ...(marker.categories ?? []),
          JSON.stringify(marker.popupFields),
        ]
          .join(' ')
          .toLowerCase()
          .includes(query),
    )
    if (selected && !filtered.some(marker => marker.slug === selected?.slug)) {
      selected = undefined
      if (selection) selection.hidden = true
    }
    if (count) count.textContent = `${filtered.length} of ${data.markers.length} saved spots`
    if (list)
      list.innerHTML = filtered.length
        ? [...filtered]
            .sort((a, b) => a.title.localeCompare(b.title))
            .map(
              marker =>
                `<button type="button" class="base-map-spot" data-spot="${escapeHTML(marker.slug)}" aria-pressed="${marker.slug === selected?.slug}"><span class="base-map-swatch" style="background:${markerColor(marker)}"></span><span><strong>${escapeHTML(marker.title)}</strong><span class="base-map-spot-meta">${escapeHTML((marker.categories ?? []).join(' · '))}${marker.rating !== undefined ? ` · ☆ ${escapeHTML(String(marker.rating))}` : ''}</span>${marker.description ? `<span class="base-map-spot-description">${escapeHTML(marker.description)}</span>` : ''}</span></button>`,
            )
            .join('')
        : `<p class="base-map-no-results">No spots match. Try another search or category.</p>`
    actions.filter(filtered)
  }

  container.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof Element)) return
      const button = event.target.closest<HTMLButtonElement>('button')
      if (!button) return
      const control = button.dataset.mapControl
      if (control === 'list') {
        if (!browser) return
        browser.hidden = !browser.hidden
        container.classList.toggle('base-map--folded', browser.hidden)
        button.setAttribute('aria-expanded', String(!browser.hidden))
      } else if (control === 'fit') actions.fit()
      else if (control === 'satellite' || control === 'elevation' || control === '3d')
        actions.control(control)
      else if (button.dataset.category !== undefined) {
        category = button.dataset.category || undefined
        for (const item of categoryList?.querySelectorAll<HTMLButtonElement>('button') ?? [])
          item.setAttribute(
            'aria-pressed',
            String((item.dataset.category || undefined) === category),
          )
        update()
      } else if (button.dataset.spot) {
        const marker = data.markers.find(item => item.slug === button.dataset.spot)
        if (marker) {
          show(marker, true)
          actions.select(marker)
        }
      } else if (button.classList.contains('base-map-selection-close') && selection) {
        const previous = selected
        selection.hidden = true
        selected = undefined
        actions.clear()
        for (const item of list?.querySelectorAll<HTMLButtonElement>('button') ?? [])
          item.setAttribute('aria-pressed', 'false')
        if (previous)
          list
            ?.querySelector<HTMLButtonElement>(`[data-spot="${CSS.escape(previous.slug)}"]`)
            ?.focus()
      }
    },
    { signal },
  )
  search?.addEventListener('input', update, { signal })
  update()
  return {
    canvas,
    show,
    markers: () => filtered,
    ready() {
      for (const button of container.querySelectorAll<HTMLButtonElement>('[data-map-control]'))
        button.disabled = false
    },
    pressed(control: MapControl, enabled: boolean) {
      container
        .querySelector(`[data-map-control="${control}"]`)
        ?.setAttribute('aria-pressed', String(enabled))
    },
  }
}
