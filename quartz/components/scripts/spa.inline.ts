import micromorph from "micromorph"
import { FullSlug, RelativeURL, getFullSlug, normalizeRelativeURLs } from "../../util/path"
import { removeAllChildren, Dag } from "./util"

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

  private container: HTMLElement
  private column: HTMLElement
  private main: HTMLElement
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

    const updateNoteStates = () => {
      const notes = [...this.column.children] as HTMLElement[]

      notes.forEach((note, idx) => {
        // Skip last note
        if (idx === notes.length - 1) return

        const nextNote = notes[idx + 1]
        if (!nextNote) return

        const rect = note.getBoundingClientRect()
        const nextRect = nextNote.getBoundingClientRect()

        // Check overlay - when next note starts overlapping current note
        note.classList.toggle("overlay", nextRect.left < rect.right)

        // Check collapse - when next note fully overlaps (leaving title space)
        note.classList.toggle("collapsed", nextRect.left <= rect.left + titleWidth)
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
      // Enable stacked mode
      const button = document.getElementById("stacked-note-toggle") as HTMLButtonElement
      const container = document.getElementById("stacked-notes-container")
      if (button && container) {
        button.setAttribute("aria-checked", "true")
        container.classList.add("active")
        document.body.classList.add("stack-mode")
      }

      // Load each stacked note
      for (const noteHash of stackedNotes) {
        const slug = this.getSlugFromHash(noteHash)
        if (slug) {
          await this.add(new URL(`/${slug}`, window.location.toString()))
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
  }

  /** Generates URL-safe hash for a slug */
  private generateHash(slug: string): string {
    const str = slug.toString()
    let h1 = 0xdeadbeef ^ 0
    for (let i = 0; i < str.length; i++) {
      h1 = Math.imul(h1 ^ str.charCodeAt(i), 2654435761)
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507)

    // Convert to base62 (use shorter 8-char hashes)
    const chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    let result = ""
    let n = h1 >>> 0
    while (result.length < 8) {
      result = chars[n % 62] + result
      n = Math.floor(n / 62)
    }
    return result
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

  private getSlugFromHash(hash: string): string | undefined {
    return this.hashes.get(hash)
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

  private createNote(i: number, { contents, title, slug }: StackedNote): HTMLElement {
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

    const noteContent = document.createElement("div")
    noteContent.className = "stacked-content"
    noteContent.append(...contents)

    note.append(noteContent, noteTitle)

    const links = [...noteContent.getElementsByClassName("internal")] as HTMLAnchorElement[]

    for (const link of links) {
      const href = link.href
      const traverseLink = async (e: MouseEvent) => {
        if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return

        e.preventDefault()
        const res = await this.add(new URL(href), link)
        if (res) {
          const children = [...this.column.children] as HTMLElement[]
          this.main.scrollTo({
            left: (
              this.column.children.item(
                children.findIndex((node) => node.dataset.slug === href),
              ) as HTMLElement
            ).getBoundingClientRect().left,
            behavior: "smooth",
          })
        }
      }

      link.addEventListener("click", traverseLink)
      window.addCleanup(() => link.removeEventListener("click", traverseLink))
    }

    queueMicrotask(() => this.scrollHandler?.())
    return note
  }

  private render() {
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
    this.dag.getOrderedNodes().forEach((node, i) => {
      if (!currentChildren.some((child) => child.dataset.slug === node.slug)) {
        node.note = this.createNote(i, {
          slug: node.slug,
          title: node.title,
          contents: node.contents,
        })
        this.column.appendChild(node.note)
      }
    })

    this.column.style.width = `${this.column.children.length * width}px`
    this.container.classList.toggle("active", this.isActive)
  }

  private focus(slug: string) {
    const notes = [...this.column.children] as HTMLElement[]
    const note = notes.find((note) => note.dataset.slug === slug)
    if (!note) return

    this.main.scrollTo({
      left: (this.column.children.item(notes.indexOf(note)) as HTMLElement).getBoundingClientRect()
        .left,
      behavior: "smooth",
    })
  }
  async add(href: URL, anchor?: HTMLElement) {
    const slug = this.getSlug(href)
    if (!anchor) anchor = document.activeElement as HTMLAnchorElement
    const clickedNote = document.activeElement?.closest(".stacked-note") as HTMLDivElement

    Array.from(clickedNote.getElementsByClassName("dag")).forEach((anchor) =>
      anchor.classList.toggle("dag", false),
    )
    anchor.classList.add("dag")

    // If note exists in DAG
    if (this.dag.has(slug)) {
      this.focus(slug)
      return true
    }

    // Get clicked note's slug
    const clickedSlug = clickedNote?.dataset.slug

    // If we clicked from a note in the DAG, truncate after it
    if (clickedSlug && this.dag.has(clickedSlug)) {
      this.dag.truncateAfter(clickedSlug)
    }

    const res = await this.fetchContent(href)
    if (!res) return false

    const { hash } = res
    // Add new note to DAG
    const note = this.createNote(this.dag.getOrderedNodes().length, {
      slug,
      ...res,
    })

    if (hash) {
      const heading = note.querySelector(hash) as HTMLElement | null
      if (heading) {
        // leave ~12px of buffer when scrolling to a heading
        note.scroll({ top: heading.offsetTop - 12, behavior: "smooth" })
      }
    }

    this.dag.addNode({ slug, title: res.title, anchor, note, contents: res.contents })
    this.focus(slug)
    this.updateURL()
    return true
  }

  async open() {
    const res = await this.fetchContent(new URL(`/${this.baseSlug}`, window.location.toString()))
    if (!res) return false

    const { hash } = res
    const note = this.createNote(0, { slug: this.baseSlug, ...res })
    this.dag.addNode({
      slug: this.baseSlug,
      title: res.title,
      anchor: null,
      note,
      contents: res.contents,
    })

    if (hash) {
      const heading = note.querySelector(hash) as HTMLElement | null
      if (heading) {
        // leave ~12px of buffer when scrolling to a heading
        note.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
      }
    }

    this.isActive = true
    this.render()
    this.updateURL()
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
      await this.open()
      await this.initFromParams()
    } else {
      await this.add(url)
    }
    this.render()
    notifyNav(this.getSlug(url))

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
  // Check for stackednotes parameter
  const hasStackedNotes = url.searchParams.has("stackedNotes")

  if (hasStackedNotes) {
    // Enable stacked notes view
    const container = document.getElementById("stacked-notes-container")
    const button = document.getElementById("stacked-note-toggle") as HTMLButtonElement
    if (container && button) {
      button.setAttribute("aria-checked", "true")
      container.classList.add("active")
      document.body.classList.add("stack-mode")

      // Let stacked notes manager handle the navigation
      notifyNav(getFullSlug(window))
      return stacked.navigate(url)
    }
  }

  const stackedContainer = document.getElementById("stacked-notes-container")
  if (stackedContainer?.classList.contains("active")) {
    return stacked.navigate(url)
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

      // Preserve stackednotes params when navigating
      const currentParams = new URL(window.location.toString()).searchParams
      const stackedNotes = currentParams.getAll("stackedNotes")
      if (stackedNotes.length > 0) {
        stackedNotes.forEach((note) => {
          url.searchParams.append("stackedNotes", note)
        })
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
