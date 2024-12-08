import micromorph from "micromorph"
import {
  FullSlug,
  RelativeURL,
  SimpleSlug,
  getFullSlug,
  normalizeRelativeURLs,
  resolveRelative,
} from "../../util/path"
import { removeAllChildren, Dag } from "./util"
import { ContentIndex, ContentDetails } from "../../plugins"
import { unescapeHTML } from "../../util/escape"
import { formatDate } from "../Date"

// adapted from `micromorph`
// https://github.com/natemoo-re/micromorph
const NODE_TYPE_ELEMENT = 1
let announcer = document.createElement("route-announcer")
const isElement = (target: EventTarget | null): target is Element =>
  (target as Node)?.nodeType === NODE_TYPE_ELEMENT
const isLocalUrl = (href: string) => {
  try {
    const url = new URL(href)
    if (window.location.origin === url.origin) {
      return true
    }
  } catch (e) {}
  return false
}

const isSamePage = (url: URL): boolean => {
  const sameOrigin = url.origin === window.location.origin
  const samePath = url.pathname === window.location.pathname
  return sameOrigin && samePath
}

const getOpts = ({ target }: Event): { url: URL; scroll?: boolean } | undefined => {
  if (!isElement(target)) return
  if (target.attributes.getNamedItem("target")?.value === "_blank") return
  const a = target.closest("a")
  if (!a) return
  if ("routerIgnore" in a.dataset) return
  const { href } = a
  if (!isLocalUrl(href)) return
  return { url: new URL(href), scroll: "routerNoscroll" in a.dataset ? false : undefined }
}

function notifyNav(url: FullSlug) {
  const event: CustomEventMap["nav"] = new CustomEvent("nav", { detail: { url } })
  document.dispatchEvent(event)
}

const cleanupFns: Set<(...args: any[]) => void> = new Set()
window.addCleanup = (fn) => cleanupFns.add(fn)

// Additional interfaces and types

interface StackedNote {
  slug: string
  contents: HTMLElement[]
  title: string
  hash?: string
}

let p: DOMParser
class StackedNoteManager {
  private dag: Dag = new Dag()

  container: HTMLElement
  column: HTMLElement
  main: HTMLElement

  private styled: CSSStyleDeclaration

  private scrollHandler: (() => void) | null = null

  private baseSlug: FullSlug
  private isActive: boolean = false

  constructor() {
    this.container = document.getElementById("stacked-notes-container") as HTMLDivElement
    this.main = this.container.querySelector("#stacked-notes-main") as HTMLDivElement
    this.column = this.main.querySelector(".stacked-notes-column") as HTMLDivElement

    this.styled = getComputedStyle(this.main)

    this.baseSlug = getFullSlug(window)

    this.setupScrollHandlers()
  }

  private setupScrollHandlers() {
    if (!this.column) return

    const titleWidth = parseInt(this.styled.getPropertyValue("--note-title-width"))
    const contentWidth = parseInt(this.styled.getPropertyValue("--note-content-width"))

    const updateNoteStates = () => {
      const notes = [...this.column.children] as HTMLElement[]
      const clientWidth = document.documentElement.clientWidth

      notes.forEach((note, idx, arr) => {
        const rect = note.getBoundingClientRect()

        if (idx === notes.length - 1) {
          const shouldCollapsed = clientWidth - rect.left <= 50 // 40px + padding
          note.classList.toggle("collapsed", shouldCollapsed)
          if (shouldCollapsed) {
            note.scrollTo({ top: 0 })
          }
          return
        }

        const nextNote = notes[idx + 1]
        if (!nextNote) return

        const nextRect = nextNote.getBoundingClientRect()

        // Calculate right position based on client width and buffer
        const fromRightPosition = clientWidth - rect.left < titleWidth * (arr.length - idx + 1)
        if (fromRightPosition) {
          note.style.right = `-${contentWidth - titleWidth - (arr.length - idx - 1) * titleWidth}px`
        }

        // Check overlay - when next note starts overlapping current note
        nextNote.classList.toggle("overlay", nextRect.left < rect.right)

        // Check collapse - when next note fully overlaps (leaving title space)
        const shouldCollapsed = nextRect.left <= rect.left + titleWidth
        if (shouldCollapsed) {
          note.scrollTo({ top: 0 })
        }
        note.classList.toggle("collapsed", shouldCollapsed)
      })
    }

    this.scrollHandler = () => {
      requestAnimationFrame(updateNoteStates)
    }

    this.main.addEventListener("scroll", this.scrollHandler)
    window.addEventListener("resize", this.scrollHandler)
    this.scrollHandler()

    window.addCleanup(() => {
      if (this.scrollHandler) {
        this.main.removeEventListener("scroll", this.scrollHandler)
        window.removeEventListener("resize", this.scrollHandler)
      }
    })
  }

  private async initFromParams() {
    const url = new URL(window.location.toString())
    const stackedNotes = url.searchParams.getAll("stackedNotes")

    if (stackedNotes.length > 0) {
      // Load each stacked note
      for (const noteHash of stackedNotes) {
        const slug = this.decodeHash(noteHash)
        if (slug) {
          const href = new URL(`/${slug}`, window.location.toString())

          if (this.dag.has(slug)) {
            // NOTE: we still have to notifyNav to register events correctly if we initialized from searchParams
            notifyNav(href.pathname as FullSlug)
            continue
          }

          const res = await this.fetchContent(href)
          if (!res) continue

          const dagNode = this.dag.addNode({
            ...res,
            slug,
            anchor: null,
            note: undefined!,
          })
          dagNode.note = await this.createNote(this.dag.getOrderedNodes().length, {
            slug,
            ...res,
          })
          notifyNav(href.pathname as FullSlug)
        }
      }
    }
  }

  private updateURL() {
    const url = new URL(window.location.toString())

    // Clear existing stackednotes params
    url.searchParams.delete("stackedNotes")

    // Add current stack state
    this.dag.getOrderedNodes().forEach((node) => {
      url.searchParams.append("stackedNotes", this.hashSlug(node.slug))
    })

    // Update URL without reloading
    window.history.replaceState({}, "", url)
    this.updateAnchorHighlights()
  }

  getChain() {
    return this.dag
      .getOrderedNodes()
      .map((el) => `stackedNotes=${this.hashSlug(el.slug)}`)
      .join("&")
  }

  /** Generates URL-safe hash for a slug. Probably use base64 for easier reconstruction */
  private generateHash(slug: string): string {
    return btoa(slug.toString()).replace(/=+$/, "")
  }

  private decodeHash(hash: string): string {
    return atob(hash).match(/^[a-zA-Z0-9/-]+$/)![0]
  }

  // Map to store hash -> slug mappings
  private hashes: Map<string, string> = new Map()
  private slugs: Map<string, string> = new Map()

  private hashSlug(slug: string): string {
    // Check if we already have a hash for this slug
    if (this.slugs.has(slug)) {
      return this.slugs.get(slug)!
    }

    // Generate new hash
    const hash = this.generateHash(slug)
    this.hashes.set(hash, slug)
    this.slugs.set(slug, hash)
    return hash
  }

  private async fetchContent(url: URL): Promise<Omit<StackedNote, "slug"> | undefined> {
    p = p || new DOMParser()

    const hash = decodeURIComponent(url.hash)
    url.hash = ""
    url.search = ""

    const response = await fetch(url.toString()).catch(console.error)
    if (!response) return

    const txt = await response.text()
    const html = p.parseFromString(txt, "text/html")
    normalizeRelativeURLs(html, url)
    const contents = [...html.getElementsByClassName("popover-hint")] as HTMLElement[]
    if (contents.length === 0) return

    const h1 = html.querySelector("h1")
    const title =
      h1?.innerText ??
      h1?.textContent ??
      this.getSlug(url) ??
      html.querySelector("title")?.textContent

    return { hash, contents, title }
  }

  private allFiles: ContentIndex | null = null
  private async loadData() {
    if (!this.allFiles) {
      const data = await fetchData
      this.allFiles = new Map(Object.entries(data) as [FullSlug, ContentDetails][])
    }
  }

  private async getBacklinks(slug: string) {
    await this.loadData()
    // Return empty array if no files loaded
    if (!this.allFiles) return []

    // Find all keys where the slug appears in their links array
    return Array.from(this.allFiles.entries()).filter(([_, value]) =>
      value.links.includes(slug as SimpleSlug),
    ) as [FullSlug, ContentDetails][]
  }

  private async createBacklinks(slug: string) {
    const noteBacklinks = document.createElement("section")
    noteBacklinks.dataset.backlinks = "true"
    noteBacklinks.classList.add("backlinks")

    const data = await this.getBacklinks(slug)
    const hasBacklinks = data.length > 0

    const title = document.createElement("h2")
    title.textContent = "Liens retour"
    const overflow = document.createElement("div")
    overflow.classList.add("overflow")
    if (hasBacklinks) {
      for (const [fullSlug, details] of data) {
        const anchor = document.createElement("a")
        anchor.classList.add("internal")
        anchor.dataset.backlink = fullSlug
        anchor.href = resolveRelative(slug as FullSlug, fullSlug)

        const title = document.createElement("div")
        title.classList.add("small")
        title.textContent = details.title

        const description = document.createElement("div")
        description.classList.add("description")
        description.textContent = unescapeHTML(details.description ?? "...")

        anchor.append(title, description)
        overflow.appendChild(anchor)
      }
    } else {
      const nonce = document.createElement("div")
      nonce.textContent = "Aucun lien retour trouvé"
      overflow.appendChild(nonce)
    }

    noteBacklinks.append(title, overflow)

    return noteBacklinks
  }

  private async createNote(
    i: number,
    { contents, title, slug }: StackedNote,
  ): Promise<HTMLElement> {
    const width = parseInt(this.styled.getPropertyValue("--note-content-width"))
    const left = parseInt(this.styled.getPropertyValue("--note-title-width"))
    const right = width - left

    const note = document.createElement("div")
    note.className = "stacked-note"
    note.style.left = `${i * left}px`
    note.style.right = `-${right}px`
    note.dataset.slug = slug

    // Create note contents...
    const noteTitle = document.createElement("div")
    noteTitle.classList.add("stacked-title")
    noteTitle.textContent = title

    const elView = () => {
      this.main.scrollTo({ left: note.getBoundingClientRect().left - i * left, behavior: "smooth" })
    }
    noteTitle.addEventListener("click", elView)
    window.addCleanup(() => noteTitle.removeEventListener("click", elView))

    const noteContent = document.createElement("div")
    noteContent.className = "stacked-content"
    noteContent.append(...contents)
    if (!["tags"].some((v) => slug.includes(v))) {
      const backlinks = await this.createBacklinks(slug)
      noteContent.append(backlinks)
    }

    await this.loadData()
    // NOTE: some pages are auto-generated, so we don't have access here in allFiles
    const el = this.allFiles!.get(slug as FullSlug)
    if (el) {
      const date = el.fileData
        ? new Date(el.fileData.dates!.modified)
        : el.date
          ? new Date(el.date)
          : new Date()
      if (date) {
        const dateContent = document.createElement("div")
        dateContent.classList.add("published")
        dateContent.innerHTML = `<span lang="fr" class="metadata" dir="auto">dernière modification par <time datetime=${date.toISOString()}>${formatDate(date)}</time></span>`
        noteContent.append(dateContent)
      }
    }

    note.append(noteContent, noteTitle)

    const links = [...noteContent.getElementsByClassName("internal")] as HTMLAnchorElement[]

    for (const link of links) {
      const href = link.href
      const slug = link.dataset.slug as string
      if (this.dag.has(slug)) {
        link.classList.add("dag")
      }

      const onClick = async (e: MouseEvent) => {
        if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return

        e.preventDefault()

        if (this.dag.has(slug)) {
          return await this.focus(slug)
        }
        await this.add(new URL(href), link)
        notifyNav(slug as FullSlug)
      }

      const onMouseEnter = (ev: MouseEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (this.dag.has(link.dataset.slug!)) {
          const note = this.dag.get(link.dataset.slug!)?.note
          const header = note!.querySelector("h1") as HTMLHeadElement
          const stackedTitle = note!.querySelector(".stacked-title") as HTMLDivElement
          header.classList.toggle("dag", true)
          stackedTitle.classList.toggle("dag", true)
        }
      }
      const onMouseLeave = (ev: MouseEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (this.dag.has(link.dataset.slug!)) {
          const note = this.dag.get(link.dataset.slug!)?.note
          const header = note!.querySelector("h1") as HTMLHeadElement
          const stackedTitle = note!.querySelector(".stacked-title") as HTMLDivElement
          header.classList.toggle("dag", false)
          stackedTitle.classList.toggle("dag", false)
        }
      }

      link.addEventListener("click", onClick)
      link.addEventListener("mouseenter", onMouseEnter)
      link.addEventListener("mouseleave", onMouseLeave)
      window.addCleanup(() => {
        link.removeEventListener("click", onClick)
        link.removeEventListener("mouseenter", onMouseEnter)
        link.removeEventListener("mouseleave", onMouseLeave)
      })
    }

    queueMicrotask(() => this.scrollHandler?.())
    return note
  }

  private async render() {
    const width = parseInt(this.styled.getPropertyValue("--note-content-width"))
    const currentChildren = Array.from(this.column.children) as HTMLElement[]

    // Remove notes not in DAG
    currentChildren.forEach((child) => {
      const slug = child.dataset.slug!
      if (!this.dag.has(slug)) {
        this.column.removeChild(child)
      }
    })

    // Add missing notes from DAG path
    for (const [i, node] of this.dag.getOrderedNodes().entries()) {
      if (!currentChildren.some((child) => child.dataset.slug === node.slug)) {
        node.note = await this.createNote(i, {
          slug: node.slug,
          title: node.title,
          contents: node.contents,
        })
        this.column.appendChild(node.note)

        if (node.hash) {
          const heading = node.note.querySelector(node.hash) as HTMLElement | null
          if (heading) {
            // leave ~12px of buffer when scrolling to a heading
            node.note.scroll({ top: heading.offsetTop - 12, behavior: "smooth" })
          }
        }
      }
    }

    this.column.style.width = `${this.column.children.length * width}px`
    this.container.classList.toggle("active", this.isActive)

    // Always scroll to rightmost note
    if (this.column.lastElementChild) {
      requestAnimationFrame(() => {
        // Calculate full scroll width
        const scrollWidth = this.column.scrollWidth - this.main.clientWidth
        this.main.scrollTo({ left: scrollWidth, behavior: "smooth" })
      })
    }
  }

  private async focus(slug: string) {
    const notes = [...this.column.children] as HTMLElement[]
    const note = notes.find((note) => note.dataset.slug === slug)
    if (!note) return false

    requestAnimationFrame(() => {
      this.main.scrollTo({ left: note.getBoundingClientRect().left, behavior: "smooth" })
    })
    note.classList.add("highlights")
    setTimeout(() => {
      note.classList.remove("highlights")
    }, 500)
    return true
  }

  private async updateAnchorHighlights() {
    for (const el of this.dag.getOrderedNodes()) {
      Array.from(el.note.getElementsByClassName("internal")).forEach((el) =>
        el.classList.toggle("dag", this.dag.has((el as HTMLAnchorElement).href)),
      )
    }
  }

  async add(href: URL, anchor?: HTMLElement) {
    let slug = this.getSlug(href)

    // handle default url by appending index for uniqueness
    if (href.pathname === "/") {
      if (slug === "") {
        slug = "index" as FullSlug
      } else {
        slug = `${slug}/index` as FullSlug
      }
    }

    if (!anchor) anchor = document.activeElement as HTMLAnchorElement
    const clickedNote = document.activeElement?.closest(".stacked-note") as HTMLDivElement
    anchor.classList.add("dag")

    this.baseSlug = slug
    // If note exists in DAG
    if (this.dag.has(slug)) {
      return await this.focus(slug)
    }

    // Get clicked note's slug
    const clickedSlug = clickedNote?.dataset.slug

    // If we clicked from a note in the DAG, truncate after it
    if (clickedSlug && this.dag.has(clickedSlug)) {
      this.dag.truncateAfter(clickedSlug)
    }

    const res = await this.fetchContent(href)
    if (!res) return false

    // Add new note to DAG before creating DOM element
    // note will be set after creation
    const dagNode = this.dag.addNode({ ...res, slug, anchor, note: undefined! })
    // Add new note to DAG
    dagNode.note = await this.createNote(this.dag.getOrderedNodes().length, {
      slug,
      ...res,
    })
    this.updateURL()
    return true
  }

  async open() {
    // We will need to construct the results from the current page, so no need to fetch here.
    const contents = [
      ...Array.from(document.getElementsByClassName("popover-hint")).map((el) =>
        el.cloneNode(true),
      ),
    ] as HTMLElement[]
    const h1 = document.querySelector("h1")
    const title =
      h1?.innerText ??
      h1?.textContent ??
      getFullSlug(window) ??
      document.querySelector("title")?.textContent
    const hash = decodeURIComponent(window.location.hash)
    window.location.hash = ""
    const res = { contents, title, hash }

    const note = await this.createNote(0, { slug: this.baseSlug, ...res })
    this.dag.addNode({ ...res, slug: this.baseSlug, anchor: null, note })

    this.isActive = true
    await this.initFromParams()
    this.updateURL()
    await this.render()
    return true
  }

  destroy() {
    this.isActive = false

    this.dag.clear()
    removeAllChildren(this.column)

    // Clear stackednotes from URL
    const url = new URL(window.location.toString())
    url.searchParams.delete("stackedNotes")
    window.history.replaceState({}, "", url)

    cleanupFns.forEach((fn) => fn())
    cleanupFns.clear()
    notifyNav(getFullSlug(window))
  }

  async navigate(url: URL) {
    if (!this.active) {
      return await this.open()
    } else {
      await this.add(url)
    }
    notifyNav(this.getSlug(url))
    await this.render()
    return true
  }

  private getSlug(url: URL): FullSlug {
    return url.pathname.slice(1) as FullSlug
  }

  get active() {
    return this.isActive
  }
}

const stacked = new StackedNoteManager()
window.stacked = stacked

async function navigate(url: URL, isBack: boolean = false) {
  const stackedContainer = document.getElementById("stacked-notes-container")
  if (stackedContainer?.classList.contains("active")) {
    return await stacked.navigate(url)
  }

  p = p || new DOMParser()
  const contents = await fetch(`${url}`)
    .then((res) => {
      const contentType = res.headers.get("content-type")
      if (contentType?.startsWith("text/html")) {
        return res.text()
      } else {
        window.location.assign(url)
      }
    })
    .catch(() => {
      window.location.assign(url)
    })

  if (!contents) return

  // cleanup old
  cleanupFns.forEach((fn) => fn())
  cleanupFns.clear()

  const html = p.parseFromString(contents, "text/html")
  normalizeRelativeURLs(html, url)

  let title = html.querySelector("title")?.textContent
  if (title) {
    document.title = title
  } else {
    const h1 = document.querySelector("h1")
    title = h1?.innerText ?? h1?.textContent ?? url.pathname
  }
  if (announcer.textContent !== title) {
    announcer.textContent = title
  }
  announcer.dataset.persist = ""
  html.body.appendChild(announcer)

  // morph body
  micromorph(document.body, html.body)

  // scroll into place and add history
  if (!isBack) {
    if (url.hash) {
      const el = document.getElementById(decodeURIComponent(url.hash.substring(1)))
      el?.scrollIntoView()
    } else {
      window.scrollTo({ top: 0 })
    }
  }

  // now, patch head
  const elementsToRemove = document.head.querySelectorAll(":not([spa-preserve])")
  elementsToRemove.forEach((el) => el.remove())
  const elementsToAdd = html.head.querySelectorAll(":not([spa-preserve])")
  elementsToAdd.forEach((el) => document.head.appendChild(el))

  // delay setting the url until now
  // at this point everything is loaded so changing the url should resolve to the correct addresses
  if (!isBack) {
    history.pushState({}, "", url)
  }
  notifyNav(getFullSlug(window))
  delete announcer.dataset.persist
}

window.spaNavigate = navigate

function createRouter() {
  if (typeof window !== "undefined") {
    window.addEventListener("click", async (event) => {
      const { url } = getOpts(event) ?? {}
      // dont hijack behaviour, just let browser act normally
      if (!url || event.ctrlKey || event.metaKey) return
      event.preventDefault()

      if (isSamePage(url) && url.hash) {
        const el = document.getElementById(decodeURIComponent(url.hash.substring(1)))
        el?.scrollIntoView()
        history.pushState({}, "", url)
        return
      }

      try {
        navigate(url, false)
      } catch (e) {
        window.location.assign(url)
      }
    })

    window.addEventListener("popstate", (event) => {
      const { url } = getOpts(event) ?? {}
      if (window.location.hash && window.location.pathname === url?.pathname) return
      try {
        navigate(new URL(window.location.toString()), true)
      } catch (e) {
        window.location.reload()
      }
      return
    })
  }

  return new (class Router {
    go(pathname: RelativeURL) {
      const url = new URL(pathname, window.location.toString())
      return navigate(url, false)
    }

    back() {
      return window.history.back()
    }

    forward() {
      return window.history.forward()
    }
  })()
}

function pruneNotesElement() {
  document
    .querySelectorAll(
      'section[class~="page-footer"], footer, span#stacked-note-toggle, nav.breadcrumb-container, .keybind',
    )
    .forEach((el) => el.remove())
}

document.addEventListener("nav", () => {
  if (window.location.hostname.startsWith("notes.aarnphm.xyz")) {
    pruneNotesElement()
  }
})

createRouter()
notifyNav(getFullSlug(window))

if (!customElements.get("route-announcer")) {
  const attrs = {
    "aria-live": "assertive",
    "aria-atomic": "true",
    style:
      "position: absolute; left: 0; top: 0; clip: rect(0 0 0 0); clip-path: inset(50%); overflow: hidden; white-space: nowrap; width: 1px; height: 1px",
  }

  customElements.define(
    "route-announcer",
    class RouteAnnouncer extends HTMLElement {
      constructor() {
        super()
      }
      connectedCallback() {
        for (const [key, value] of Object.entries(attrs)) {
          this.setAttribute(key, value)
        }
      }
    },
  )
}

// NOTE: navigate first if there are stackedNotes
const baseUrl = new URL(document.location.toString())
const stackedNotes = baseUrl.searchParams.get("stackedNotes")
const container = document.getElementById("stacked-notes-container")

// If there's a stackedNotes parameter and stacked mode isn't active, activate it
if (stackedNotes && !container?.classList.contains("active")) {
  const button = document.getElementById("stacked-note-toggle") as HTMLSpanElement
  const header = document.getElementsByClassName("header")[0] as HTMLElement

  button.setAttribute("aria-checked", "true")
  container?.classList.add("active")
  document.body.classList.add("stack-mode")
  header.classList.add("grid", "all-col")
  header.classList.remove(header.dataset.column!)

  if (window.location.hash) {
    window.history.pushState("", document.title, baseUrl.toString().split("#")[0])
  }
  stacked.navigate(baseUrl)
  // NOTE: we need to call this once more to register all existing handler
  notifyNav(getFullSlug(window))
}

// remove elements on notes.aarnphm.xyz
if (window.location.hostname.startsWith("notes.aarnphm.xyz")) {
  if (!stackedNotes || stackedNotes.length === 0) {
    const slug = "notes"
    baseUrl.searchParams.set("stackedNotes", btoa(slug.toString()).replace(/=+$/, ""))
    baseUrl.pathname = `/${slug}`
    stacked.navigate(baseUrl)
  }
}
