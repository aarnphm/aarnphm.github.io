import micromorph from "micromorph"
import {
  FullSlug,
  RelativeURL,
  getFullSlug,
  normalizeRelativeURLs,
  sluggify,
} from "../../util/path"
import { removeAllChildren, Dag, DagNode } from "./util"
import { ContentDetails } from "../../plugins"
import { formatDate } from "../Date"
import { fetchCanonical, startViewTransition } from "./util"
import { Toast } from "./toast"

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

if (!window.quartzToast) {
  const toast = new Toast()
  window.quartzToast = toast

  const handleToast = (event: CustomEventMap["toast"]) => {
    const detail = event.detail
    if (!detail?.message) return
    toast.show(detail.message, {
      durationMs: detail.durationMs,
      styles: detail.styles,
      containerId: detail.containerId,
      containerStyles: detail.containerStyles,
    })
  }

  document.addEventListener("toast", handleToast)
  document.addEventListener("prenav", () => toast.destroy())
}

function startLoading() {
  const loadingBar = document.createElement("div")
  loadingBar.className = "navigation-progress"
  loadingBar.style.width = "0"
  if (!document.body.contains(loadingBar)) {
    document.body.appendChild(loadingBar)
  }

  setTimeout(() => {
    loadingBar.style.width = "100%"
  }, 100)
}

// Additional interfaces and types

interface StackedNote {
  slug: string
  contents: [...HTMLDivElement[]]
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

  private isActive: boolean = false

  constructor() {
    this.container = document.getElementById("stacked-notes-container") as HTMLDivElement
    this.main = this.container.querySelector("#stacked-notes-main") as HTMLDivElement
    this.column = this.main.querySelector(".stacked-notes-column") as HTMLDivElement

    this.styled = getComputedStyle(this.main)

    this.setupScrollHandlers()
  }

  private mobile() {
    return window.innerWidth <= 800
  }

  private setupScrollHandlers() {
    if (!this.column) return

    if (this.mobile()) {
      this.scrollHandler = () => {}
      return
    }

    const titleWidth = parseInt(this.styled.getPropertyValue("--note-title-width"))
    const contentWidth = parseInt(this.styled.getPropertyValue("--note-content-width"))

    const updateNoteStates = () => {
      const notes = [...this.column.children].filter(
        (el) => !el.classList.contains("popover"),
      ) as HTMLElement[]
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

    if (stackedNotes.length === 0) return

    // Check if notes already rendered by server
    const existingNotes = [...this.column.querySelectorAll(".stacked-note")] as HTMLElement[]

    if (existingNotes.length > 0) {
      // Server-rendered path: hydrate existing notes
      for (const [index, noteElement] of existingNotes.entries()) {
        const slug = noteElement.dataset.slug!
        const noteHash = stackedNotes[index]
        const decodedSlug = this.decodeHash(noteHash)

        if (slug !== decodedSlug) {
          console.warn(`slug mismatch at index ${index}: ${slug} vs ${decodedSlug}`)
          continue
        }

        // Extract title and contents from existing DOM (no fetch needed)
        const noteTitle = noteElement.querySelector(".stacked-title") as HTMLDivElement
        const noteContent = noteElement.querySelector(".stacked-content") as HTMLDivElement

        const title = noteTitle?.textContent || slug
        // Extract popover-hint elements from rendered content
        const contents = noteContent
          ? (Array.from(noteContent.querySelectorAll(".popover-hint")) as HTMLDivElement[])
          : []

        // Add to DAG with reference to existing DOM element
        this.dag.addNode({
          slug,
          title,
          contents,
          hash: "",
          anchor: null,
          note: noteElement,
        })

        // Attach event listeners
        this.hydrateNote(noteElement)

        notifyNav(slug as FullSlug)
      }

      // Update URL and DAG classes
      this.updateURL()
      return
    }

    // Client-rendered path: fetch and build notes

    // Create an array to store all fetch promises
    const fetchPromises = stackedNotes.map(async (noteHash) => {
      const slug = this.decodeHash(noteHash)
      if (!slug) return null

      const href = new URL(`/${slug}`, window.location.toString())

      if (this.dag.has(slug)) {
        // Still notify for navigation events
        notifyNav(href.pathname as FullSlug)
        return null
      }

      const res = await this.fetchContent(href)
      if (!res) return null

      return {
        slug,
        href,
        res,
      }
    })

    // Wait for all fetches to complete in parallel
    const results = await Promise.all(fetchPromises)

    // Process the results in order
    for (const result of results.filter(Boolean)) {
      if (!result) continue
      const { slug, href, res } = result

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

  private updateURL() {
    const url = new URL(window.location.toString())

    // Clear existing stackednotes params
    url.searchParams.delete("stackedNotes")

    // Add current stack state and right position
    this.dag.getOrderedNodes().forEach((node, index, _) => {
      url.searchParams.append("stackedNotes", this.hashSlug(node.slug))

      const width = parseInt(this.styled.getPropertyValue("--note-content-width"))
      const left = parseInt(this.styled.getPropertyValue("--note-title-width"))
      const right = width - left
      node.note.style.right = `${-right + (this.dag.getOrderedNodes().length - index - 1) * left}px`
    })

    // Update URL without reloading
    window.history.replaceState({}, "", url)
    // Update anchor highlights
    for (const el of this.dag.getOrderedNodes()) {
      Array.from(el.note.getElementsByClassName("internal")).forEach((el) =>
        el.classList.toggle("dag", this.dag.has((el as HTMLAnchorElement).dataset.slug!)),
      )
    }
  }

  getChain() {
    return this.dag
      .getOrderedNodes()
      .map((el) => `stackedNotes=${this.hashSlug(el.slug)}`)
      .join("&")
  }

  /** Generates URL-safe hash for a slug. Uses base64 with special handling for dots */
  private generateHash(slug: string): string {
    // Replace dots with a safe character sequence before encoding
    const safePath = slug.toString().replace(/\./g, "___DOT___")
    return btoa(safePath).replace(/=+$/, "")
  }

  private decodeHash(hash: string): string {
    try {
      const decoded = atob(hash)
      // Restore dots after decoding
      const restoredPath = decoded.replace(/___DOT___/g, ".")
      // Validate the path only contains allowed characters
      if (restoredPath.match(/^[a-zA-Z0-9/.-]+$/)) {
        return restoredPath
      }
      throw new Error("Invalid path characters")
    } catch (e) {
      console.error("Failed to decode hash:", e)
      return ""
    }
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

    const response = await fetchCanonical(url).catch(console.error)
    if (!response) return

    const txt = await response.text()
    const html = p.parseFromString(txt, "text/html")
    normalizeRelativeURLs(html, url)
    transformHostInternalLinks(html)

    // check if the page is protected
    const protectedArticle = html.querySelector('article[data-protected="true"]')
    if (protectedArticle) {
      // create a protected indicator element
      const protectedDiv = document.createElement("div")
      protectedDiv.className = "protected-stacked-note"
      protectedDiv.innerHTML = `
        <div class="protected-overlay">
          <div class="protected-message">
            <p>ce contenu est protégé</p>
            <p class="protected-hint">visitez la page principale pour y accéder</p>
          </div>
        </div>
      `

      const h1 = html.querySelector("h1")
      const title =
        h1?.innerText ??
        h1?.textContent ??
        this.getSlug(url) ??
        html.querySelector("title")?.textContent

      return { hash, contents: [protectedDiv], title }
    }

    const contents = new Set<HTMLDivElement>()
    for (const el of Array.from(html.getElementsByClassName("popover-hint"))) {
      if (el.classList.contains("page-footer") && !el.hasChildNodes()) {
        el.remove()
        continue
      }
      contents.add(el as HTMLDivElement)
    }
    if (contents.size == 0) return

    const h1 = html.querySelector("h1")
    const title =
      h1?.innerText ??
      h1?.textContent ??
      this.getSlug(url) ??
      html.querySelector("title")?.textContent

    return { hash, contents: [...contents], title }
  }

  private allFiles: ContentIndex | null = null
  private async loadData() {
    if (!this.allFiles) {
      const data = await fetchData
      this.allFiles = new Map(Object.entries(data) as [FullSlug, ContentDetails][])
    }
    return this.allFiles
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
    note.id = this.hashSlug(slug)
    note.style.left = `${i * left}px`
    note.style.right = `${-right + (this.dag.getOrderedNodes().length - i - 1) * left}px`
    note.dataset.slug = slug

    // Create note contents...
    const noteTitle = document.createElement("div")
    noteTitle.classList.add("stacked-title")
    noteTitle.textContent = title

    const elView = () => {
      // Calculate full scroll width and note's relative position
      const scrollWidth = this.column.scrollWidth - this.main.clientWidth
      const noteLeft = note.offsetLeft
      const scrollPosition = Math.min(noteLeft, scrollWidth)
      this.main.scrollTo({ left: scrollPosition, behavior: "smooth" })
    }
    noteTitle.addEventListener("click", elView)
    window.addCleanup(() => noteTitle.removeEventListener("click", elView))

    const noteContent = document.createElement("div")
    noteContent.className = "stacked-content"
    noteContent.append(...contents)
    transformHostInternalLinks(noteContent)

    await this.loadData().then((allFiles) => {
      // NOTE: some pages are auto-generated, so we don't have access here in allFiles
      const el = allFiles.get(slug as FullSlug)
      if (el) {
        const date = el.fileData
          ? new Date(el.fileData.dates!.modified)
          : el.date
            ? new Date(el.date)
            : new Date()
        if (date) {
          const dateContent = document.createElement("div")
          dateContent.classList.add("published")
          dateContent.innerHTML = `<span lang="fr" class="metadata" dir="auto">dernière modification par <time datetime=${date.toISOString()}>${formatDate(date)}</time> (${el.readingTime?.minutes!} min de lecture)</span>`
          noteContent.append(dateContent)
        }
      }
      note.append(noteContent, noteTitle)
    })

    const links = [...noteContent.getElementsByClassName("internal")] as HTMLAnchorElement[]

    for (const link of links) {
      if ("routerIgnore" in link.dataset) continue

      const href = link.href
      const slug = link.dataset.slug as string
      if (this.dag.has(slug)) {
        link.classList.add("dag")
      }

      const onClick = async (e: MouseEvent) => {
        if (e.ctrlKey || e.metaKey || e.shiftKey) return

        e.preventDefault()
        if (e.altKey) {
          // When alt/option is pressed, add to the end without truncating
          const slug = link.dataset.slug as string
          if (!this.dag.has(slug)) {
            const res = await this.fetchContent(new URL(href))
            if (!res) return
            const dagNode = this.dag.addNode({ ...res, slug, anchor: link, note: undefined! })
            dagNode.note = await this.createNote(this.dag.getOrderedNodes().length, {
              slug,
              ...res,
            })
            this.updateURL()
            await this.render()
            notifyNav(slug as FullSlug)
          }
          return
        }
        await this.add(new URL(href), link)
      }

      const onMouseEnter = (ev: MouseEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (this.dag.has(link.dataset.slug!)) {
          const note = this.dag.get(link.dataset.slug!)?.note
          const header = note!.querySelector<HTMLHeadElement>("h1")
          const stackedTitle = note!.querySelector<HTMLDivElement>(".stacked-title")
          if (header) header!.classList.toggle("dag", true)
          if (stackedTitle) stackedTitle!.classList.toggle("dag", true)
        }
      }
      const onMouseLeave = (ev: MouseEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (this.dag.has(link.dataset.slug!)) {
          const note = this.dag.get(link.dataset.slug!)?.note
          const header = note!.querySelector<HTMLHeadElement>("h1")
          const stackedTitle = note!.querySelector<HTMLDivElement>(".stacked-title")
          if (header) header!.classList.toggle("dag", false)
          if (stackedTitle) stackedTitle!.classList.toggle("dag", false)
        }
      }

      const onKeyDown = (ev: KeyboardEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (ev.altKey && !link.title) {
          link.title = "pour ajouter à la fin de la pile"
        }
      }
      const onKeyUp = (ev: KeyboardEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (!ev.altKey && link.title === "pour ajouter à la fin de la pile") {
          link.title = ""
        }
      }

      link.addEventListener("click", onClick)
      link.addEventListener("mouseenter", onMouseEnter)
      link.addEventListener("mouseleave", onMouseLeave)
      link.addEventListener("keydown", onKeyDown)
      link.addEventListener("keyup", onKeyUp)
      window.addCleanup(() => {
        link.removeEventListener("click", onClick)
        link.removeEventListener("mouseenter", onMouseEnter)
        link.removeEventListener("mouseleave", onMouseLeave)
        link.removeEventListener("keydown", onKeyDown)
        link.removeEventListener("keyup", onKeyUp)
      })
    }

    queueMicrotask(() => this.scrollHandler?.())
    return note
  }

  /**
   * Hydrate a server-rendered note by attaching event listeners
   * Does not create DOM, only binds interactivity
   */
  private hydrateNote(note: HTMLElement): void {
    const slug = note.dataset.slug!
    const noteTitle = note.querySelector(".stacked-title") as HTMLDivElement
    const noteContent = note.querySelector(".stacked-content") as HTMLDivElement
    transformHostInternalLinks(noteContent)

    if (!noteTitle || !noteContent) {
      console.warn(`missing title or content for note ${slug}`)
      return
    }

    // attach title click handler
    const elView = () => {
      const scrollWidth = this.column.scrollWidth - this.main.clientWidth
      const noteLeft = note.offsetLeft
      const scrollPosition = Math.min(noteLeft, scrollWidth)
      this.main.scrollTo({ left: scrollPosition, behavior: "smooth" })
    }
    noteTitle.addEventListener("click", elView)
    window.addCleanup(() => noteTitle.removeEventListener("click", elView))

    // attach link handlers
    const links = [...noteContent.getElementsByClassName("internal")] as HTMLAnchorElement[]

    for (const link of links) {
      if ("routerIgnore" in link.dataset) continue

      const href = link.href
      const linkSlug = link.dataset.slug as string

      // update DAG class if already in stack
      if (this.dag.has(linkSlug)) {
        link.classList.add("dag")
      }

      const onClick = async (e: MouseEvent) => {
        if (e.ctrlKey || e.metaKey || e.shiftKey) return

        e.preventDefault()
        if (e.altKey) {
          // when alt/option is pressed, add to the end without truncating
          if (!this.dag.has(linkSlug)) {
            const res = await this.fetchContent(new URL(href))
            if (!res) return
            const dagNode = this.dag.addNode({
              ...res,
              slug: linkSlug,
              anchor: link,
              note: undefined!,
            })
            dagNode.note = await this.createNote(this.dag.getOrderedNodes().length, {
              slug: linkSlug,
              ...res,
            })
            this.updateURL()
            await this.render()
            notifyNav(linkSlug as FullSlug)
          }
          return
        }
        await this.add(new URL(href), link)
      }

      const onMouseEnter = (ev: MouseEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (this.dag.has(link.dataset.slug!)) {
          const note = this.dag.get(link.dataset.slug!)?.note
          const header = note!.querySelector<HTMLHeadElement>("h1")
          const stackedTitle = note!.querySelector<HTMLDivElement>(".stacked-title")
          if (header) header!.classList.toggle("dag", true)
          if (stackedTitle) stackedTitle!.classList.toggle("dag", true)
        }
      }
      const onMouseLeave = (ev: MouseEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (this.dag.has(link.dataset.slug!)) {
          const note = this.dag.get(link.dataset.slug!)?.note
          const header = note!.querySelector<HTMLHeadElement>("h1")
          const stackedTitle = note!.querySelector<HTMLDivElement>(".stacked-title")
          if (header) header!.classList.toggle("dag", false)
          if (stackedTitle) stackedTitle!.classList.toggle("dag", false)
        }
      }

      const onKeyDown = (ev: KeyboardEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (ev.altKey && !link.title) {
          link.title = "pour ajouter à la fin de la pile"
        }
      }
      const onKeyUp = (ev: KeyboardEvent) => {
        const link = ev.target as HTMLAnchorElement
        if (!ev.altKey && link.title === "pour ajouter à la fin de la pile") {
          link.title = ""
        }
      }

      link.addEventListener("click", onClick)
      link.addEventListener("mouseenter", onMouseEnter)
      link.addEventListener("mouseleave", onMouseLeave)
      link.addEventListener("keydown", onKeyDown)
      link.addEventListener("keyup", onKeyUp)
      window.addCleanup(() => {
        link.removeEventListener("click", onClick)
        link.removeEventListener("mouseenter", onMouseEnter)
        link.removeEventListener("mouseleave", onMouseLeave)
        link.removeEventListener("keydown", onKeyDown)
        link.removeEventListener("keyup", onKeyUp)
      })
    }

    // metadata footer is now included in server-rendered HTML
    // no need to add it client-side during hydration
  }

  private async render() {
    const currentChildren = Array.from(this.column.children) as HTMLElement[]

    if (this.mobile()) {
      const node = this.dag.getTail() as DagNode
      currentChildren.forEach((child) => {
        if (child.dataset.slug !== node!.slug) this.column.removeChild(child)
      })

      // Create last node if needed
      if (!this.column.children.length) {
        node.note = await this.createNote(0, {
          slug: node.slug,
          title: node.title,
          contents: node.contents as HTMLDivElement[],
        })
        this.column.appendChild(node.note)
      }
      this.container.classList.toggle("active", this.isActive)
      return
    }

    const width = parseInt(this.styled.getPropertyValue("--note-content-width"))

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
          contents: node.contents as HTMLDivElement[],
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

    // Skip mermaid rendering for stacked notes - causes memory issues and failures
    return
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

    // If note exists in DAG
    if (this.dag.has(slug)) {
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
      notifyNav(slug)
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

    // Add new note to DAG before creating DOM element
    // note will be set after creation
    const dagNode = this.dag.addNode({ ...res, slug, anchor, note: undefined! })
    // Add new note to DAG
    dagNode.note = await this.createNote(this.dag.getOrderedNodes().length, {
      slug,
      ...res,
    })
    this.updateURL()
    notifyNav(this.getSlug(href))
    return true
  }

  async open() {
    // We will need to construct the results from the current page, so no need to fetch here.
    const contents = Array.from(document.getElementsByClassName("popover-hint")).map((el) =>
      el.cloneNode(true),
    ) as HTMLDivElement[]
    const h1 = document.querySelector("h1")
    const title =
      h1?.innerText ??
      h1?.textContent ??
      getFullSlug(window) ??
      document.querySelector("title")?.textContent
    const hash = decodeURIComponent(window.location.hash)
    window.location.hash = ""
    const res = { contents, title, hash }

    const note = await this.createNote(0, { slug: getFullSlug(window), ...res })
    this.dag.addNode({ ...res, slug: getFullSlug(window), anchor: null, note })

    this.isActive = true
    await this.initFromParams()
    this.updateURL()
    await this.render().then(() => notifyNav(getFullSlug(window)))

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
  }

  async navigate(url: URL) {
    try {
      if (!this.active) return await this.open()

      // notify about to nav
      const event: CustomEventMap["prenav"] = new CustomEvent("prenav", { detail: {} })
      document.dispatchEvent(event)

      await this.add(url)
      await this.render()
    } catch (e) {
      console.error(`Failed to navigate to ${url}: ${e}`)
      return false
    }
    return true
  }

  private getSlug(url: URL): FullSlug {
    return url.pathname.slice(1) as FullSlug
  }

  get active() {
    return this.isActive
  }
}

const STREAM_HOSTNAME = "stream.aarnphm.xyz"
const NOTES_HOSTNAME = "notes.aarnphm.xyz"
const APEX_ORIGIN = "https://aarnphm.xyz"

function normalizeStreamPath(raw: string): string | null {
  if (!raw) return null
  const trimmed = raw.trim()
  if (trimmed === "") return "/"
  if (trimmed.startsWith("#")) return null

  try {
    const parsed = new URL(trimmed, `https://${STREAM_HOSTNAME}`)
    const hash = parsed.hash ?? ""
    const search = parsed.search ?? ""
    let pathname = parsed.pathname.replace(/^\/+/, "")
    if (pathname === "") return `/${search}${hash}`
    pathname = sluggify(pathname)
    const normalizedPath = `/${pathname}`
    return `${normalizedPath}${search}${hash}`
  } catch {
    return null
  }
}

function transformHostInternalLinks(root: Document | Element) {
  if (typeof window === "undefined") return
  const host = window.location.hostname

  const anchors = root.querySelectorAll<HTMLAnchorElement>("a.internal")
  anchors.forEach((anchor) => {
    if (host === STREAM_HOSTNAME) {
      const preferred = anchor.dataset.slug ?? anchor.getAttribute("href") ?? ""
      const normalizedPath = normalizeStreamPath(preferred)
      if (!normalizedPath) return

      const absoluteHref = `${APEX_ORIGIN}${
        normalizedPath.startsWith("/") ? normalizedPath : `/${normalizedPath}`
      }`
      if (anchor.getAttribute("href") !== absoluteHref) {
        anchor.setAttribute("href", absoluteHref)
      }
      anchor.dataset.noPopover = "true"
      anchor.dataset.routerIgnore = "true"
    } else if (host === NOTES_HOSTNAME) {
      anchor.dataset.noPopover = "true"
    }
  })

  if (host === STREAM_HOSTNAME) {
    const transcludeRefs = root.querySelectorAll<HTMLElement>(".transclude-ref[data-href]")
    transcludeRefs.forEach((ref) => {
      const preferred = ref.dataset.slug ?? ref.dataset.href ?? ""
      const normalizedPath = normalizeStreamPath(preferred)
      if (!normalizedPath) return
      const absoluteHref = `${APEX_ORIGIN}${
        normalizedPath.startsWith("/") ? normalizedPath : `/${normalizedPath}`
      }`
      ref.dataset.href = absoluteHref
      ref.dataset.routerIgnore = "true"
    })
  }
}

const stacked = new StackedNoteManager()
window.stacked = stacked

async function navigate(url: URL, isBack: boolean = false) {
  const stackedContainer = document.getElementById("stacked-notes-container")
  if (stackedContainer?.classList.contains("active")) {
    return await window.stacked.navigate(url)
  }

  startLoading()

  p = p || new DOMParser()
  const contents = await fetchCanonical(new URL(`${url}`))
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

  // notify about to nav
  const event: CustomEventMap["prenav"] = new CustomEvent("prenav", { detail: {} })
  document.dispatchEvent(event)

  // cleanup old
  cleanupFns.forEach((fn) => fn())
  cleanupFns.clear()

  const html = p.parseFromString(contents, "text/html")
  normalizeRelativeURLs(html, url)
  transformHostInternalLinks(html)

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
  startViewTransition(() => {
    micromorph(document.body, html.body)
    transformHostInternalLinks(document)

    // scroll into place and add history
    if (!isBack) {
      if (url.hash) {
        const el = document.getElementById(decodeURIComponent(url.hash.substring(1)))
        el?.scrollIntoView()
      } else {
        window.scrollTo({ top: 0 })
      }
    }

    // now, patch head, re-executing scripts
    const elementsToRemove = document.head.querySelectorAll(":not([data-persist])")
    elementsToRemove.forEach((el) => el.remove())
    const elementsToAdd = html.head.querySelectorAll(":not([data-persist])")
    elementsToAdd.forEach((el) => document.head.appendChild(el))

    // delay setting the url until now
    // at this point everything is loaded so changing the url should resolve to the correct addresses
    if (!isBack) {
      history.pushState({}, "", url)
    }
    notifyNav(getFullSlug(window))
    delete announcer.dataset.persist
  })
}

window.spaNavigate = navigate
window.notifyNav = notifyNav

document.addEventListener("nav", () => {
  transformHostInternalLinks(document)
})

function createRouter() {
  if (typeof window !== "undefined") {
    window.addEventListener("click", async (event) => {
      const { url } = getOpts(event) ?? {}
      // dont hijack behaviour, just let browser act normally
      if (!url || event.ctrlKey || event.metaKey || event.altKey) return
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
        console.error(e)
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
        console.error(e)
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

// NOTE: navigate first if there are stackedNotes
const baseUrl = new URL(document.location.toString())
const stackedNotes = baseUrl.searchParams.get("stackedNotes")

// remove elements on notes.aarnphm.xyz
if (window.location.host === "notes.aarnphm.xyz") {
  if (!stackedNotes || stackedNotes.length === 0) {
    const slug = "notes"
    baseUrl.searchParams.set("stackedNotes", btoa(slug.toString()).replace(/=+$/, ""))
    baseUrl.pathname = `/${slug}`

    window.stacked.navigate(baseUrl).then((data) => {
      if (data) window.location.reload()
      document
        .querySelectorAll(
          'main > section[class~="page-footer"], main > section[class~="page-header"], main > section[class~="page-content"], nav.breadcrumb-container, header > .keybind, header > .search, header > .graph',
        )
        .forEach((el) => el.remove())
    })
  }
}
