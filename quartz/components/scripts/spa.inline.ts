import micromorph from "micromorph"
import {
  FullSlug,
  RelativeURL,
  getFullSlug,
  normalizeRelativeURLs,
  resolveRelative,
} from "../../util/path"
import { removeAllChildren } from "./util"

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

interface StackedNote {
  slug: string
  contents: HTMLElement[]
  hash?: string
}

let p: DOMParser
class StackedNoteManager {
  private stack: StackedNote[]

  private container: HTMLElement
  private column: HTMLElement
  private main: HTMLElement
  private focusIdx: number = -1

  private isActive: boolean = false
  private baseSlug: FullSlug

  constructor() {
    this.stack = []
    this.container = document.getElementById("stacked-notes-container") as HTMLDivElement
    this.main = this.container.querySelector("#stacked-notes-main") as HTMLDivElement
    this.column = this.main.querySelector(".stacked-notes-column") as HTMLDivElement

    this.baseSlug = getFullSlug(window)

    window.addEventListener("popstate", async (e) => {
      if (e.state?.stackState) {
        await this.restore(e.state.stackState)
      } else {
        this.destroy()
      }
    })
  }

  private createNote(elts: HTMLElement[], slug: string): HTMLElement {
    const note = document.createElement("div")
    note.className = "stacked-note"
    note.dataset.slug = slug

    const noteContent = document.createElement("div")
    noteContent.className = "stacked-content"
    noteContent.append(...elts)

    note.appendChild(noteContent)

    const links = [...noteContent.getElementsByClassName("internal")] as HTMLAnchorElement[]

    for (const link of links) {
      const href = link.href
      const traverseLink = async (e: MouseEvent) => {
        if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
        e.preventDefault()

        if (!this.isActive) {
          await this.open()
        } else {
          await this.add(new URL(href))
        }
      }

      link.addEventListener("click", traverseLink)
      window.addCleanup(() => link.removeEventListener("click", traverseLink))
    }

    return note
  }

  private setFocus(index: number) {
    const notes = this.column.querySelectorAll(".stacked-note")
    notes.forEach((note, idx) => {
      note.classList.toggle("active", idx === index)
    })

    if (index >= 0 && index < notes.length) {
      const activeNote = notes[index] as HTMLElement
      this.main.scrollTo({
        left: activeNote.offsetLeft - 20,
        behavior: "smooth",
      })
    }

    this.focusIdx = index
  }

  private resolveRelative(targetSlug: FullSlug): URL {
    return new URL(resolveRelative(this.baseSlug, targetSlug), location.toString())
  }

  private async fetchContent(slug: FullSlug | URL, isUrl: boolean = false) {
    p = p || new DOMParser()

    const url = isUrl ? new URL(slug) : new URL(this.resolveRelative(slug as FullSlug))
    const hash = decodeURIComponent(url.hash)
    url.hash = ""
    url.search = ""

    // TODO: prevent fetching the same page

    const response = await fetch(url.toString()).catch(console.error)
    if (!response) return

    const txt = await response.text()
    const html = p.parseFromString(txt, "text/html")
    normalizeRelativeURLs(html, url)
    const contents = [...html.getElementsByClassName("popover-hint")] as HTMLElement[]
    if (contents.length === 0) return

    return { hash, contents }
  }

  // TODO: FIX THIS RIGHT NOW BREOKEN
  private hashed() {
    const stackState = this.stack.map((note) => note.slug)
    const url = new URL(window.location.toString())
    if (stackState.length > 0) {
      url.hash = `stack=${stackState.join(",")}`
    } else {
      url.hash = ""
    }
    history.pushState({ stackState }, "", url)
  }

  private render() {
    removeAllChildren(this.column)

    for (const { hash, slug, contents } of this.stack) {
      const note = this.createNote(contents, slug)
      this.column.appendChild(note)

      if (hash) {
        const heading = note.querySelector(hash) as HTMLElement | null
        if (heading) {
          note.scroll({ top: heading.offsetTop - 12, behavior: "instant" })
        }
      }
    }
    this.container.classList.toggle("active", this.isActive)

    if (this.active) {
      if (this.stack.length > 0) this.setFocus(this.stack.length - 1)

      const keydown = (e: KeyboardEvent) => {
        if (e.key === "ArrowLeft" && this.focusIdx > 0) {
          this.setFocus(this.focusIdx - 1)
        } else if (e.key === "ArrowRight" && this.focusIdx < this.stack.length - 1) {
          this.setFocus(this.focusIdx + 1)
        }
      }

      document.addEventListener("keydown", keydown)
      window.addCleanup(() => document.removeEventListener("keydown", keydown))
    }
  }

  async open() {
    const res = await this.fetchContent(this.baseSlug)
    if (!res) return

    this.stack = [{ slug: this.baseSlug, ...res }]
    this.isActive = true
    this.render()
    this.hashed()
  }

  async add(href: URL) {
    const res = await this.fetchContent(href, true)
    const slug = href.pathname.slice(1)
    if (!res) return

    // Remove any notes after the current one if we're branching
    const existing = this.stack.findIndex((note) => note.slug === slug)
    if (existing !== -1) {
      this.stack = this.stack.slice(0, existing)
    }

    this.baseSlug = slug as FullSlug

    this.stack.push({ slug, ...res })
  }

  close(slug: string) {
    const idx = this.stack.findIndex((note) => note.slug === slug)
    if (idx !== -1) {
      this.stack = this.stack.slice(0, idx)
      if (this.stack.length === 0) {
        this.destroy()
      } else {
        // FIXME: OK this is not ideal, we should just remove the diff of the slice instead of re-rendering everything
        this.render()
        this.hashed()
      }
    }
  }

  destroy() {
    this.stack = []
    this.isActive = false
    this.focusIdx = -1
    removeAllChildren(this.column)
  }

  async restore(slugs: string[]) {
    this.stack = []
    for (const slug of slugs) {
      const res = await this.fetchContent(slug as FullSlug)
      if (!res) continue

      this.stack.push({ slug, ...res })
    }
    this.isActive = this.stack.length > 0
    this.render()
  }

  async navigate(url: URL) {
    if (!this.active) {
      await this.open()
    } else {
      await this.add(url)
    }
    this.render()
    return true
  }

  get active() {
    return this.isActive
  }
}

const stacked = new StackedNoteManager()
window.stacked = stacked

async function navigate(url: URL, isBack: boolean = false) {
  const stackedContainer = document.getElementById("stacked-notes-container") as HTMLButtonElement
  if (stackedContainer.classList.contains("active")) {
    stacked.navigate(url)
    return
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
