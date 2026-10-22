import type { BasesPropertyId } from 'obsidian'
import type { ViewFilter } from '../covers'
import type { ViewHost } from './view-host'
import { authorLabel, type BookModel } from '../model'
import { readConfig, type BookcaseConfig } from '../options'
import { buildScene, type ShelfGroup, type ShelfMaterial } from '../scene-model'
import { materialMarkup } from '../svg'
import { ALL, lookupTopic, readTopicConfig, topicMembers, topicTabs } from '../topics'

/** The tags a book carries, for tag topics. */
const TAGS = 'file.tags' as BasesPropertyId

export function rebuild(this: ViewHost): void {
  if (!this.data) return
  // Until the view has a box there is nothing to measure; the resize observer calls again.
  const scroller = this.stage.scroller ?? this.stage.discover()
  if (!scroller) return
  if (!this.scrollers.has(scroller)) {
    // A view that went full screen and back has two scrollers; only the current one counts.
    this.scrollers.add(scroller)
    this.registerDomEvent(scroller, 'scroll', evt => this.onScroll(evt), { passive: true })
    if (scroller !== this.root) this.observer?.observe(scroller)
  }

  const settings = readConfig(this.config)
  this.settings = settings
  if (this.linked && !settings.editorPane) {
    this.editor.forget()
    this.setLinked(false, false)
  }
  this.dark = document.body.hasClass('theme-dark')
  const { groups, style, material, continuous, showTags } = this.filtered(settings)
  const scene = buildScene(
    groups,
    {
      bookHeight: settings.bookHeight,
      shelfWidth: settings.shelfWidth,
      continuous,
      style,
      material,
      showTags,
      rows: settings.shelfRows,
    },
    model => ({ title: model.spine, author: authorLabel(model.authors) || undefined }),
  )
  const count = scene.books.length
  this.root.toggleClass('is-empty', count === 0)
  const materials = new Set<ShelfMaterial>()
  for (const layer of scene.layers) {
    for (const solid of layer.solids) materials.add(solid.material)
    for (const halo of layer.halos) materials.add(halo.material)
  }
  const defsKey = [...materials].sort().join(' ')
  if (defsKey !== this.defsKey) {
    this.defsKey = defsKey
    this.defsEl.innerHTML = materialMarkup(materials, this.plugin.grainAssets)
  }

  const [minX, minY, maxX, maxY] = scene.bounds
  const metrics = this.stage.measure(
    count,
    settings.scrollStep,
    scene.fit,
    scene.span,
    minX,
    maxX - minX,
    maxY - minY,
    this.panel.inset(),
    !this.linked && this.marked >= 0,
  )
  const { scale } = metrics
  this.frame = { minX, minY, scale }
  this.size = [(maxX - minX) * scale, (maxY - minY) * scale]
  this.world.style.width = `${this.size[0]}px`
  this.world.style.height = `${this.size[1]}px`
  this.zoom = Math.min(this.zoom, this.maxZoom())

  const previous = this.paints
  this.scene = scene
  this.version++
  this.paints = []
  this.centres = []
  this.reaches = []
  this.shelfLayer.clear()
  this.fromEnd = []
  for (const [i, layer] of scene.layers.entries()) {
    if (layer.kind !== 'shelf') continue
    this.shelfLayer.set(layer.shelf, i)
    for (const [p, index] of layer.books.entries()) this.fromEnd[index] = layer.books.length - p
  }
  this.coverBooks.clear()
  this.recolour.clear()
  const files = scene.books.map(book => {
    const cover = book.item.cover
    if (cover?.kind !== 'vault') return null
    const list = this.coverBooks.get(cover.file.path)
    if (list) list.push(book.index)
    else this.coverBooks.set(cover.file.path, [book.index])
    return cover.file
  })
  this.paintLayers(scene, previous)
  // The first index waits for the idle pass; later ones follow the data at once.
  if (this.buttons.length > 0) this.updateIndex(scene)
  else this.indexPending = true

  // Keep the focused book by path; when it is gone, keep its index. Only a shift moves the
  // scroll position, so a rebuild never cancels a smooth scroll in progress.
  let focus = this.focus
  let top: number | undefined = metrics.scrollTop
  if (this.filterChanged) {
    // A new topic starts at its first book, from the top, with the whole case in view.
    this.filterChanged = false
    this.focus = focus = -1
    this.focusPath = null
    this.clearSelection()
    this.zoom = 1
    this.pan = [0, 0]
    this.releasing = false
    scroller.scrollTop = 0
    top = 0
  }
  if (this.restoreFocus) {
    this.restoreFocus = false
    if (focus >= 0) {
      scroller.scrollTop = metrics.trackTop + Math.min(focus, count - 1) * settings.scrollStep
      top = undefined
    }
  }
  if (count > 0 && focus >= 0) {
    focus = Math.min(focus, count - 1)
    if (this.focusPath && scene.books[focus]?.path !== this.focusPath) {
      let best = -1
      for (const book of scene.books)
        if (
          book.path === this.focusPath &&
          (best < 0 || Math.abs(book.index - this.focus) < Math.abs(best - this.focus))
        )
          best = book.index
      if (best >= 0) focus = best
    }
    if (focus !== this.focus) {
      // The browser may clamp the offset, so the camera reads it back (a rare layout read).
      scroller.scrollTop = metrics.trackTop + focus * settings.scrollStep
      top = undefined
    }
  }
  this.plugin.covers.need(this, files, Math.max(0, focus))
  this.resolveSelection(scene)
  this.camera(true, top)
  this.setControls()
  this.dataChanged = false
  // Shelves outside the stage, and the markup of the others, are left to idle time.
  this.requestSettle()
}

export function filtered(
  this: ViewHost,
  settings: BookcaseConfig,
): {
  groups: ShelfGroup<BookModel>[]
  style: any
  material:
    | 'paper'
    | 'walnut'
    | 'birch-ply'
    | 'black-lacquer'
    | 'steel'
    | 'copper'
    | 'concrete'
    | null
  continuous: boolean
  showTags: boolean
} {
  const data = this.data
  const config = readTopicConfig(this.config)
  const entries = data.data
  const tagsOf = new Map(
    entries.map(e => [e.file.path, this.reader.strings(e, TAGS).map(tag => tag.replace(/^#/, ''))]),
  )
  const tags = (path: string) => tagsOf.get(path) ?? []
  const pairs: [string | null, string][] = []
  let total = 0
  for (const group of data.groupedData) {
    const key = group.hasKey() && group.key ? group.key.toString() : null
    for (const entry of group.entries) pairs.push([key, entry.file.path])
    total += group.entries.length
  }
  const members = topicMembers(pairs, tags, config)
  const tabs = topicTabs(members, total, config)
  this.filterKey ??= this.baseKey()
  const saved = this.filter ?? this.plugin.covers.filter(this.filterKey)
  const topic = saved && tabs.some(tab => tab.id === saved.topic) ? saved.topic : ALL
  this.filter = { topic }
  this.tabs = tabs
  this.panel.render(tabs, topic)
  this.reserve = this.panel.reserve()

  const shelf = lookupTopic(topic, config).shelf
  const style = shelf?.style ?? settings.shelfStyle
  // As the garden: the topic's shelf, else the view's options, which also style "all".
  const material = shelf?.material ?? settings.shelfMaterial
  const showTags = shelf?.showTags ?? settings.showTags
  if (this.root.dataset.topic !== topic) this.root.dataset.topic = topic
  if (this.root.dataset.shelfStyle !== style) this.root.dataset.shelfStyle = style
  // A topic's shelves group by small subcategories, so they share runs, as in the garden.
  const continuous = topic !== ALL
  const set = members.get(topic)
  const groups = continuous
    ? this.reader.topicGroups(
        entries.filter(entry => !set || set.has(entry.file.path)),
        config.shelvesBy,
        settings,
      )
    : this.reader.groups(data.groupedData, settings)
  return { groups, style, material, continuous, showTags }
}

export function baseKey(this: ViewHost): string {
  let path = ''
  const embed = this.containerEl.closest('.internal-embed, .bases-embed')
  const src = embed?.getAttribute('src')
  if (src) {
    const link = src.split('#')[0]
    path = this.app.metadataCache.getFirstLinkpathDest(link, '')?.path ?? link
  } else
    for (const leaf of this.app.workspace.getLeavesOfType('bases')) {
      const view = leaf.view as { containerEl?: HTMLElement; file?: { path: string } | null }
      if (view.containerEl?.contains(this.containerEl)) path = view.file?.path ?? ''
    }
  return `${path}#${this.config.name}`
}

export function setFilter(this: ViewHost, filter: ViewFilter): void {
  this.filter = filter
  this.filterChanged = true
  if (this.filterKey !== null) this.plugin.covers.setFilter(this.filterKey, filter)
  this.lift.clear()
  this.dataChanged = true
  this.schedule()
}

export function selectTopic(this: ViewHost, id: string): void {
  if (this.filter && id !== this.filter.topic) this.setFilter({ topic: id })
}
