import FlexSearch from "flexsearch"
import { ContentIndex } from "../../plugins"
import { FilePath, FullSlug, resolveRelative, simplifySlug } from "../../util/path"
import { highlight, registerEscapeHandler, removeAllChildren, encode } from "./util"

interface Item {
  id: number
  slug: FullSlug
  name: FilePath
  aliases: string[]
}

let index = new FlexSearch.Document<Item>({
  charset: "latin:extra",
  encode,
  document: {
    id: "id",
    index: [
      {
        field: "name",
        tokenize: "forward",
      },
      {
        field: "aliases",
        tokenize: "forward",
      },
    ],
  },
})

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

type ActionType = "quick_open" | "command"
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
    isActive = false
    recentItems = []
  }

  function showPalette(actionTypeNew: ActionType) {
    actionType = actionTypeNew
    container?.classList.add("active")
    if (actionType === "command") {
      helper.querySelectorAll<HTMLLIElement>("li[data-quick-open]").forEach((el) => {
        el.style.display = "none"
      })
    } else if (actionType === "quick_open") {
      getRecentItems()
    }

    bar?.focus()
    isActive = true
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
        recentItems.push({
          id,
          slug,
          name: data[slug].fileName,
          aliases: data[slug].aliases,
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

        recentItems.push({
          id,
          slug: slug as FullSlug,
          name: data[slug].fileName,
          aliases: data[slug].aliases,
        })

        // Remove used slug to avoid duplicates
        availableSlugs.splice(randomIndex, 1)
      }
    }

    output.append(...recentItems.map(toHtml))
  }

  let isActive: boolean = false
  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    const searchOpen = document.querySelector("search#search-container") as HTMLDivElement
    if (searchOpen && searchOpen.classList.contains("active")) return

    if (e.key === "o" && (e.ctrlKey || e.metaKey) && !isActive) {
      e.preventDefault()
      const barOpen = container?.classList.contains("active")
      barOpen ? hidePalette() : showPalette("quick_open")
      return
    } else if (e.key === "p" && (e.ctrlKey || e.metaKey) && !isActive) {
      e.preventDefault()
      const barOpen = container?.classList.contains("active")
      barOpen ? hidePalette() : showPalette("command")
      return
    }

    // If search is active, then we will render the first result and display accordingly
    if (!container?.classList.contains("active")) return

    if (e.key === "Enter") {
      // If result has focus, navigate to that one, otherwise pick first result
      let anchor: HTMLAnchorElement | undefined
      if (output?.contains(document.activeElement)) {
        anchor = document.activeElement as HTMLAnchorElement
        if (anchor.classList.contains("no-match")) return
        e.preventDefault()
        anchor.click()
      } else {
        anchor = output.getElementsByClassName("suggestion-item")[0] as HTMLAnchorElement
        if (!anchor || anchor?.classList.contains("no-match")) return
        e.preventDefault()
        anchor.click()
      }
    } else if ((e.metaKey || e.altKey) && e.key === "Enter") {
      // Find the focused item
      const focusedElement = output.querySelector<HTMLDivElement>(".suggestion-item.focus")
      if (!focusedElement || focusedElement.classList.contains("no-match")) return

      e.preventDefault()
      // Get the slug from the focused element's data
      const items = output.querySelectorAll<HTMLDivElement>(".suggestion-item")
      const currentIndex = Array.from(items).indexOf(focusedElement)
      const slug = idDataMap[currentIndex]

      // Add to recents and open in new tab
      addToRecents(slug)
      window.open(resolveRelative(currentSlug, slug), "_blank")
      hidePalette()
    } else if (
      e.key === "ArrowUp" ||
      (e.shiftKey && e.key === "Tab") ||
      (e.ctrlKey && e.key === "p")
    ) {
      e.preventDefault()
      const items = output.querySelectorAll<HTMLDivElement>(".suggestion-item")
      if (items.length === 0) return

      const focusedElement = output.querySelector<HTMLDivElement>(".suggestion-item.focus")

      // Remove focus from current element
      if (focusedElement) {
        focusedElement.classList.remove("focus")
        // Get the previous element or cycle to the last
        const currentIndex = Array.from(items).indexOf(focusedElement)
        const nextIndex = currentIndex <= 0 ? items.length - 1 : currentIndex - 1
        items[nextIndex].classList.add("focus")
        items[nextIndex].focus()
      } else {
        // If no element is focused, start from the last one
        const lastIndex = items.length - 1
        items[lastIndex].classList.add("focus")
        items[lastIndex].focus()
      }
    } else if (e.key === "ArrowDown" || e.key === "Tab" || (e.ctrlKey && e.key === "n")) {
      e.preventDefault()
      const items = output.querySelectorAll<HTMLDivElement>(".suggestion-item")
      if (items.length === 0) return

      const focusedElement = output.querySelector<HTMLDivElement>(".suggestion-item.focus")

      // Remove focus from current element
      if (focusedElement) {
        focusedElement.classList.remove("focus")
        // Get the next element or cycle to the first
        const currentIndex = Array.from(items).indexOf(focusedElement)
        const nextIndex = currentIndex >= items.length - 1 ? 0 : currentIndex + 1
        items[nextIndex].classList.add("focus")
        items[nextIndex].focus()
      } else {
        // If no element is focused, start from the first one
        items[0].classList.add("focus")
        items[0].focus()
      }
    }
  }

  async function onType(e: HTMLElementEventMap["input"]) {
    currentSearchTerm = (e.target as HTMLInputElement).value

    let searchResults: FlexSearch.SimpleDocumentSearchResultSetUnit[]
    if (actionType === "quick_open") {
      searchResults = await index.searchAsync({
        query: currentSearchTerm,
        limit: numSearchResults,
        index: ["name", "aliases"],
      })
    }

    const getByField = (field: string): number[] => {
      const results = searchResults.filter((x) => x.field === field)
      return results.length === 0 ? [] : ([...results[0].result] as number[])
    }

    // order titles ahead of content
    const allIds: Set<number> = new Set([...getByField("name"), ...getByField("aliases")])
    displayResults(
      [...allIds].map((id) => {
        const slug = idDataMap[id]
        return {
          id,
          slug,
          name: highlight(currentSearchTerm, data[slug].fileName) as FilePath,
          aliases: data[slug].aliases,
        }
      }),
      currentSearchTerm,
    )
  }

  function displayResults(finalResults: Item[], currentSearchTerm: string) {
    if (!finalResults) return

    removeAllChildren(output)

    if (finalResults.length === 0) {
      output.innerHTML = `<div class="suggestion-item no-match"><div class="suggestion-content"><div class="suggestion-title">${currentSearchTerm}</div></div><div class="suggestion-aux"><span class="suggestion-action">enter to schedule a chat</span></div></div>`
    } else {
      output.append(...finalResults.map(toHtml))
    }

    // focus on first result, then also dispatch preview immediately
    const firstChild = output.firstElementChild as HTMLElement
    firstChild.classList.add("focus")
  }

  function toHtml({ name, slug }: Item) {
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
  let id = 0
  const promises = []
  for (const [slug, fileData] of Object.entries(data)) {
    promises.push(
      index.addAsync(id++, {
        id,
        slug: slug as FullSlug,
        name: fileData.fileName,
        aliases: fileData.aliases,
      }),
    )
  }

  return await Promise.all(promises)
}
