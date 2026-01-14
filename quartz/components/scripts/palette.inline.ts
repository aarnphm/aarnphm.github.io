import { FilePath, FullSlug, normalizeRelativeURLs, resolveRelative } from "../../util/path"
import {
  highlight,
  registerEscapeHandler,
  removeAllChildren,
  fetchCanonical,
  createSidePanel,
  getOrCreateSidePanel,
} from "./util"
import { populateSearchIndex, querySearchIndex, SearchItem } from "./search-index"

interface Item extends SearchItem {
  target: string
}

const numSearchResults = 10

const localStorageKey = "recent-notes"
function getRecents(): Set<FullSlug> {
  return new Set(JSON.parse(localStorage.getItem(localStorageKey) ?? "[]"))
}

function addToRecents(slug: FullSlug) {
  const visited = getRecents()
  visited.add(slug)
  localStorage.setItem(localStorageKey, JSON.stringify([...visited]))
}

const commentAuthorKey = "comment-author"
const commentAuthorSourceKey = "comment-author-source"
const commentAuthorLastRenameKey = "comment-author-last-rename"
const commentAuthorGithubLoginKey = "comment-author-github-login"
const commentAuthorRenameWindowMs = 1000 * 60 * 60 * 24 * 90

function notifyToast(message: string) {
  document.dispatchEvent(new CustomEvent("toast", { detail: { message } }))
}

function dispatchCommentAuthorUpdated(oldAuthor: string, newAuthor: string) {
  document.dispatchEvent(
    new CustomEvent("commentauthorupdated", {
      detail: { oldAuthor, newAuthor },
    }),
  )
}

function getLastRenameTime(): number | null {
  const raw = localStorage.getItem(commentAuthorLastRenameKey)
  if (!raw) return null
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) ? parsed : null
}

function isRenameWindowActive() {
  const last = getLastRenameTime()
  if (last === null) return false
  return Date.now() - last < commentAuthorRenameWindowMs
}

async function requestCommentAuthorRename(oldAuthor: string, newAuthor: string): Promise<boolean> {
  const githubLogin = localStorage.getItem(commentAuthorGithubLoginKey)
  const payload: { oldAuthor: string; newAuthor: string; githubLogin?: string } = {
    oldAuthor,
    newAuthor,
  }
  if (githubLogin) {
    payload.githubLogin = githubLogin
  }
  try {
    const response = await fetch("/comments/author/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
    if (response.ok) return true
    if (response.status === 429) {
      notifyToast("comment name can change every 3 months")
      return false
    }
    const text = await response.text()
    notifyToast(text || "failed to update comment author")
  } catch {
    notifyToast("failed to update comment author")
  }
  return false
}

async function updateCommentAuthor(author: string, source: "manual" | "github") {
  const next = author.trim()
  if (!next) return
  const existing = localStorage.getItem(commentAuthorKey) ?? ""
  if (!existing) {
    localStorage.setItem(commentAuthorKey, next)
    localStorage.setItem(commentAuthorSourceKey, source)
    notifyToast(`name set to ${next}`)
    return
  }

  if (existing === next) {
    localStorage.setItem(commentAuthorSourceKey, source)
    notifyToast(`name set to ${next}`)
    return
  }

  if (isRenameWindowActive()) {
    notifyToast("name can change every 3 months")
    return
  }

  const renamed = await requestCommentAuthorRename(existing, next)
  if (!renamed) return

  localStorage.setItem(commentAuthorLastRenameKey, `${Date.now()}`)
  localStorage.setItem(commentAuthorKey, next)
  localStorage.setItem(commentAuthorSourceKey, source)
  dispatchCommentAuthorUpdated(existing, next)
  notifyToast(`name set to ${next}`)
}

function promptForCommentAuthor() {
  const existing = localStorage.getItem(commentAuthorKey) ?? ""
  const hint = "suggest: use the username that matches your gravatar"
  const promptText = existing
    ? `current comment name: ${existing}\nset comment name (${hint})`
    : `set comment name (${hint})`
  const raw = window.prompt(promptText, existing)
  if (raw === null) return
  void updateCommentAuthor(raw, "manual")
}

function startGithubCommentLogin(returnTo: string) {
  const target = new URL("/comments/github/login", window.location.origin)
  target.searchParams.set("returnTo", returnTo)
  const existing = localStorage.getItem(commentAuthorKey)
  if (existing) {
    target.searchParams.set("author", existing)
  }
  window.location.assign(target.toString())
}

const p = new DOMParser()
const fetchContentCache: Map<FullSlug, HTMLElement[]> = new Map()
async function fetchContent(currentSlug: FullSlug, slug: FullSlug): Promise<HTMLElement[]> {
  if (fetchContentCache.has(slug)) {
    return fetchContentCache.get(slug) as HTMLElement[]
  }

  const targetUrl = new URL(resolveRelative(currentSlug, slug), location.toString())
  const contents = await fetchCanonical(targetUrl)
    .then((res) => res.text())
    .then((contents) => {
      if (contents === undefined) {
        throw new Error(`Could not fetch ${targetUrl}`)
      }
      const html = p.parseFromString(contents ?? "", "text/html")
      normalizeRelativeURLs(html, targetUrl)
      return [...html.getElementsByClassName("popover-hint")] as HTMLElement[]
    })

  fetchContentCache.set(slug, contents)
  return contents
}

type ActionType = "quick_open" | "command"
interface Action {
  name: string
  onClick: (e: MouseEvent) => void
  auxInnerHtml: string
}

let actionType: ActionType = "quick_open"
let currentSearchTerm: string = ""
document.addEventListener("nav", async (e) => {
  const data = await fetchData
  const currentSlug = e.detail.url
  const idDataMap = Object.keys(data) as FullSlug[]

  const container = document.getElementById("palette-container")
  if (!container) return

  const bar = container.querySelector("#bar") as HTMLInputElement
  const output = container.getElementsByTagName("output")[0]
  const helper = container.querySelector("ul#helper") as HTMLUListElement
  let currentHover: HTMLDivElement | null = null

  function hidePalette() {
    container?.classList.remove("active")
    if (bar) {
      bar.value = "" // clear the input when we dismiss the search
    }
    if (output) {
      removeAllChildren(output)
    }

    actionType = "quick_open" // reset search type after closing
    helper.querySelectorAll<HTMLLIElement>("li[data-quick-open]").forEach((el) => {
      el.style.display = ""
    })
    recentItems = []
  }

  function showPalette(actionTypeNew: ActionType) {
    actionType = actionTypeNew
    container?.classList.add("active")
    if (actionType === "command") {
      helper.querySelectorAll<HTMLLIElement>("li[data-quick-open]").forEach((el) => {
        el.style.display = "none"
        getCommandItems(ACTS)
      })
    } else if (actionType === "quick_open") {
      getRecentItems()
    }

    bar?.focus()
  }

  const ACTS: Action[] = [
    {
      name: "x.com (formerly Twitter)",
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#twitter-icon" /></svg>`,
      onClick: () => {
        window.location.href = "https://x.com/aarnphm"
      },
    },
    {
      name: "bsky.app",
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#bsky-icon" /></svg>`,
      onClick: () => {
        window.location.href = "https://bsky.app/profile/aarnphm.xyz"
      },
    },
    {
      name: "substack",
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#substack-icon" /></svg>`,
      onClick: () => {
        window.location.href = "https://livingalonealone.com"
      },
    },
    {
      name: "github",
      auxInnerHtml: `<svg width="1em" height="1em"><use href="#github-icon" /></svg>`,
      onClick: () => {
        window.location.href = "https://github.com/aarnphm"
      },
    },
    {
      name: "commenter name",
      auxInnerHtml: "<kbd>↵</kbd> set comment handle",
      onClick: () => {
        promptForCommentAuthor()
      },
    },
    {
      name: "commenter login with github",
      auxInnerHtml: "<kbd>↵</kbd> verify via github",
      onClick: () => {
        startGithubCommentLogin(window.location.toString())
      },
    },
    {
      name: "curius",
      auxInnerHtml: "<kbd>↵</kbd> links",
      onClick: () => {
        window.spaNavigate(
          new URL(resolveRelative(currentSlug, "/curius" as FullSlug), window.location.toString()),
        )
      },
    },
    {
      name: "research",
      auxInnerHtml: "<kbd>↵</kbd> a peak into my research interests",
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, "/research" as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
    {
      name: "are.na",
      auxInnerHtml: "<kbd>↵</kbd> a rundown version of are.na",
      onClick: () => {
        window.spaNavigate(
          new URL(resolveRelative(currentSlug, "/arena" as FullSlug), window.location.toString()),
        )
      },
    },
    {
      name: "stream",
      auxInnerHtml: "<kbd>↵</kbd> microblog",
      onClick: () => {
        window.spaNavigate(
          new URL(resolveRelative(currentSlug, "/stream" as FullSlug), window.location.toString()),
        )
      },
    },
    {
      name: "dating me",
      auxInnerHtml: "<kbd>↵</kbd> as love",
      onClick: () => {
        window.spaNavigate(
          new URL(resolveRelative(currentSlug, "/dating" as FullSlug), window.location.toString()),
        )
      },
    },
    {
      name: "coffee chat",
      auxInnerHtml: "<kbd>↵</kbd> on calendly",
      onClick: () => {
        window.location.href = "https://calendly.com/aarnphm/30min"
      },
    },
    {
      name: "current work",
      auxInnerHtml: "<kbd>↵</kbd> as craft",
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, "/thoughts/craft" as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
    {
      name: "cool people",
      auxInnerHtml: "<kbd>↵</kbd> as inspiration",
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, "/influence" as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
    {
      name: "old fashioned resume (maybe not up-to-date)",
      auxInnerHtml: "<kbd>↵</kbd>",
      onClick: () => {
        window.spaNavigate(
          new URL(
            resolveRelative(currentSlug, "/thoughts/pdfs/2025q1-resume.pdf" as FullSlug),
            window.location.toString(),
          ),
        )
      },
    },
  ]

  const createActComponent = ({ name, auxInnerHtml, onClick }: Action) => {
    const item = document.createElement("div")
    item.classList.add("suggestion-item")

    const content = document.createElement("div")
    content.classList.add("suggestion-content")
    const title = document.createElement("div")
    title.classList.add("suggestion-title")
    title.innerHTML = name
    content.appendChild(title)

    const aux = document.createElement("div")
    aux.classList.add("suggestion-aux")
    aux.innerHTML = `<span class="suggestion-action">${auxInnerHtml}</span>`
    item.append(content, aux)

    function mainOnClick(e: MouseEvent) {
      e.preventDefault()
      onClick(e)
      hidePalette()
    }
    item.addEventListener("click", mainOnClick)
    window.addCleanup(() => item.removeEventListener("click", mainOnClick))
    return item
  }

  function getCommandItems(acts: Action[]) {
    if (output) {
      removeAllChildren(output)
    }
    if (acts.length === 0) {
      if (bar.matches(":focus") && currentSearchTerm === "") {
        output.append(...ACTS.map(createActComponent))
      } else {
        output.append(createActComponent(ACTS[0]))
      }
    } else {
      output.append(...acts.map(createActComponent))
    }
    setFocusFirstChild()
  }

  let recentItems: Item[] = []
  function getRecentItems() {
    const visited = getRecents()

    if (output) {
      removeAllChildren(output)
    }

    const visitedArray = [...visited]
    const els =
      visited.size > numSearchResults
        ? visitedArray.slice(-numSearchResults).reverse()
        : visitedArray.reverse()

    // If visited >= 10, then we get the first recent 10 items
    // Otherwise, we will choose randomly from the set of data
    els.forEach((slug) => {
      const id = idDataMap.findIndex((s) => s === slug)
      if (id !== -1) {
        //@ts-ignore
        recentItems.push({
          id,
          slug,
          name: data[slug].fileName,
          title: data[slug].title ?? "",
          content: data[slug].content ?? "",
          aliases: data[slug].aliases,
          target: "",
        })
      }
    })
    // Fill with random items from data
    const needed = numSearchResults - els.length
    if (needed != 0) {
      const availableSlugs = idDataMap.filter((slug) => !els.includes(slug))

      // Then add random items
      for (let i = 0; i < needed && availableSlugs.length > 0; i++) {
        const randomIndex = Math.floor(Math.random() * availableSlugs.length)
        const slug = availableSlugs[randomIndex]
        const id = idDataMap.findIndex((s) => s === slug)

        //@ts-ignore
        recentItems.push({
          id,
          slug: slug as FullSlug,
          name: data[slug].fileName,
          title: data[slug].title ?? "",
          content: data[slug].content ?? "",
          aliases: data[slug].aliases,
          target: "",
        })

        // Remove used slug to avoid duplicates
        availableSlugs.splice(randomIndex, 1)
      }
    }

    output.append(...recentItems.map(toHtml))
    setFocusFirstChild()
  }

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    const searchOpen = document.querySelector<HTMLDivElement>("search.search-container")
    const noteContainer = document.getElementById("stacked-notes-container") as HTMLDivElement
    if (
      (searchOpen && searchOpen.classList.contains("active")) ||
      (noteContainer && noteContainer.classList.contains("active"))
    )
      return

    if (e.key === "o" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const barOpen = container?.classList.contains("active")
      barOpen ? hidePalette() : showPalette("quick_open")
      return
    } else if (e.key === "p" && (e.altKey || e.metaKey)) {
      e.preventDefault()
      const barOpen = container?.classList.contains("active")
      barOpen ? hidePalette() : showPalette("command")
      return
    } else if (
      e.key.startsWith("Esc") &&
      container?.classList.contains("active") &&
      bar.matches(":focus")
    ) {
      // Handle Escape key when input is focused
      e.preventDefault()
      hidePalette()
      return
    }

    if (currentHover) currentHover.classList.remove("focus")
    if (!container?.classList.contains("active")) return

    if (e.metaKey && e.altKey && e.key === "Enter") {
      if (!currentHover) return
      const slug = currentHover.dataset.slug
      if (!slug) return

      try {
        const asidePanel = getOrCreateSidePanel()
        await fetchContent(currentSlug, slug as FullSlug).then((innerDiv) => {
          asidePanel.dataset.slug = slug
          createSidePanel(asidePanel, ...innerDiv)
          window.notifyNav(slug as FullSlug)
          hidePalette()
        })
      } catch (error) {
        console.error("Failed to create side panel:", error)
      }
      return
    } else if (e.key === "Enter") {
      // If result has focus, navigate to that one, otherwise pick first result
      if (output?.contains(currentHover)) {
        e.preventDefault()
        currentHover!.click()
      } else {
        const anchor = output.getElementsByClassName("suggestion-item")[0] as HTMLDivElement
        e.preventDefault()
        anchor.click()
      }
    } else if (e.key === "ArrowUp" || (e.ctrlKey && e.key === "p")) {
      e.preventDefault()
      const items = output.querySelectorAll<HTMLDivElement>(".suggestion-item")
      if (items.length === 0) return

      const focusedElement = currentHover
        ? currentHover
        : output.querySelector<HTMLDivElement>(".suggestion-item.focus")

      // Remove focus from current element
      if (focusedElement) {
        focusedElement.classList.remove("focus")
        // Get the previous element or cycle to the last
        const currentIndex = Array.from(items).indexOf(focusedElement)
        const prevIndex = currentIndex <= 0 ? items.length - 1 : currentIndex - 1
        currentHover = items[prevIndex]
        items[prevIndex].classList.add("focus")
        items[prevIndex].focus()
      } else {
        // If no element is focused, start from the last one
        const lastIndex = items.length - 1
        items[lastIndex].classList.add("focus")
        items[lastIndex].focus()
      }
    } else if (e.key === "Tab") {
      e.preventDefault()
      const focusedElement = currentHover
        ? currentHover
        : output.querySelector<HTMLDivElement>(".suggestion-item.focus")
      bar.value = currentSearchTerm =
        focusedElement?.querySelector<HTMLDivElement>(".suggestion-title")!.textContent ?? ""
      return await querySearch(currentSearchTerm)
    } else if (e.key === "ArrowDown" || (e.ctrlKey && e.key === "n")) {
      e.preventDefault()
      const items = output.querySelectorAll<HTMLDivElement>(".suggestion-item")
      if (items.length === 0) return

      const focusedElement = currentHover
        ? currentHover
        : output.querySelector<HTMLDivElement>(".suggestion-item.focus")

      // Remove focus from current element
      if (focusedElement) {
        focusedElement.classList.remove("focus")
        // Get the next element or cycle to the first
        const currentIndex = Array.from(items).indexOf(focusedElement)
        const nextIndex = currentIndex >= items.length - 1 ? 0 : currentIndex + 1
        currentHover = items[nextIndex]
        items[nextIndex].classList.add("focus")
        items[nextIndex].focus()
      } else {
        // If no element is focused, start from the first one
        items[0].classList.add("focus")
        items[0].focus()
      }
    }
  }

  async function querySearch(currentSearchTerm: string) {
    if (actionType === "quick_open") {
      const searchResults = await querySearchIndex(currentSearchTerm, numSearchResults)

      displayResults(
        searchResults
          .map((item) => {
            const target =
              item.aliases.find((alias) =>
                alias.toLowerCase().includes(currentSearchTerm.toLowerCase()),
              ) || ""
            return {
              ...item,
              name: highlight(currentSearchTerm, item.name) as FilePath,
              target,
            }
          })
          .sort((a, b) => {
            if ((!a?.target && !b?.target) || (a?.target && b?.target)) return 0
            if (a?.target && !b?.target) return -1
            if (!a?.target && b?.target) return 1
            return 0
          }),
        currentSearchTerm,
      )
    } else {
      // Search actions directly (simple string matching)
      const query = currentSearchTerm.toLowerCase().trim()
      const matchedActions = query
        ? ACTS.filter(
            (action) =>
              action.name.toLowerCase().includes(query) ||
              action.auxInnerHtml.toLowerCase().includes(query),
          )
        : ACTS

      getCommandItems(
        matchedActions.map(({ name, onClick, auxInnerHtml }) => ({
          name: query ? highlight(currentSearchTerm, name) : name,
          onClick,
          auxInnerHtml,
        })),
      )
    }
  }

  async function onType(e: HTMLElementEventMap["input"]) {
    currentSearchTerm = (e.target as HTMLInputElement).value
    await querySearch(currentSearchTerm)
  }

  function displayResults(finalResults: Item[], currentSearchTerm: string) {
    if (!finalResults) return

    removeAllChildren(output)

    const noMatchEl = document.createElement("div")
    noMatchEl.classList.add("suggestion-item", "no-match")
    noMatchEl.innerHTML = `<div class="suggestion-content"><div class="suggestion-title">${currentSearchTerm}</div></div><div class="suggestion-aux"><span class="suggestion-action">enter to schedule a chat</span></div>`

    const onNoMatchClick = () => {
      window.location.href = `mailto:contact@aarnphm.xyz?subject=Chat about: ${encodeURIComponent(currentSearchTerm)}`
      hidePalette()
    }

    noMatchEl.addEventListener("click", onNoMatchClick)
    window.addCleanup(() => noMatchEl.removeEventListener("click", onNoMatchClick))
    if (finalResults.length === 0) {
      if (bar.matches(":focus") && currentSearchTerm === "") {
        output.append(...recentItems.map(toHtml))
      } else {
        output.appendChild(noMatchEl)
      }
    } else {
      output.append(...finalResults.map(toHtml))
    }
    setFocusFirstChild()
  }

  function setFocusFirstChild() {
    // focus on first result, then also dispatch preview immediately
    const firstChild = output.firstElementChild as HTMLElement
    firstChild.classList.add("focus")
    currentHover = firstChild as HTMLInputElement
  }

  function toHtml({ name, slug, target }: Item) {
    const item = document.createElement("div")
    item.classList.add("suggestion-item")
    item.dataset.slug = slug

    const content = document.createElement("div")
    content.classList.add("suggestion-content")
    const title = document.createElement("div")
    title.classList.add("suggestion-title")
    const titleContent = target ? highlight(currentSearchTerm, target) : name
    const subscript = target ? `${slug}` : ``
    title.innerHTML = `${titleContent}<br/><span class="subscript">${subscript}</span>`
    content.appendChild(title)

    const aux = document.createElement("div")
    aux.classList.add("suggestion-aux")

    item.append(content, aux)

    const onClick = () => {
      addToRecents(slug)
      window.spaNavigate(new URL(resolveRelative(currentSlug, slug), location.toString()))
      hidePalette()
    }

    const onMouseEnter = () => {
      // Remove focus class from all other items
      output.querySelectorAll<HTMLDivElement>(".suggestion-item.focus").forEach((el) => {
        el.classList.remove("focus")
      })
      // Add focus to current item
      item.classList.add("focus")
      currentHover = item
    }

    item.addEventListener("click", onClick)
    item.addEventListener("mouseenter", onMouseEnter)
    window.addCleanup(() => {
      item.removeEventListener("click", onClick)
      item.removeEventListener("mouseenter", onMouseEnter)
    })

    return item
  }

  document.addEventListener("keydown", shortcutHandler)
  bar.addEventListener("input", onType)
  window.addCleanup(() => {
    document.removeEventListener("keydown", shortcutHandler)
    bar.removeEventListener("input", onType)
  })

  registerEscapeHandler(container, hidePalette)
  await fillDocument(data)
})

async function fillDocument(data: ContentIndex) {
  await populateSearchIndex(data)
}
