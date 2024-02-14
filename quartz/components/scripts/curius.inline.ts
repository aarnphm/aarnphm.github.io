import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"
import { computePosition, arrow as arrowFloating, inline, offset } from "@floating-ui/dom"
import { Link } from "../types"
import { fetchCuriusLinks } from "./curius-data.inline"

const refetchTimeout = 2 * 60 * 1000 // 2 minutes
const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/

const _IconMapping = {
  favourite: `<svg fill="currentColor" preserveAspectRatio="xMidYMid meet" height="1em" width="1em" viewBox="0 0 40 40" data-tip="Unfavorite" data-for="links" style="vertical-align: unset;"><g><path d="m5.2 18.8l6 5.5-1.7 7.7c-0.2 1 0.2 2 1 2.5 0.3 0.3 0.8 0.5 1.3 0.5 0.4 0 0.7 0 1-0.2 0 0 0.2 0 0.2-0.1l6.8-3.9 6.9 3.9s0.1 0 0.1 0.1c0.9 0.4 1.9 0.4 2.5-0.1 0.9-0.5 1.2-1.5 1-2.5l-1.6-7.7c0.6-0.5 1.6-1.5 2.6-2.5l3.2-2.8 0.2-0.2c0.6-0.7 0.8-1.7 0.5-2.5s-1-1.5-2-1.7h-0.2l-7.8-0.8-3.2-7.2s0-0.1-0.2-0.1c-0.1-1.2-1-1.7-1.8-1.7s-1.7 0.5-2.2 1.3c0 0 0 0.2-0.1 0.2l-3.2 7.2-7.8 0.8h-0.2c-0.8 0.2-1.7 0.8-2 1.7-0.2 1 0 2 0.7 2.6z"></path></g></svg>`,
}

type Keys = keyof typeof _IconMapping

const Icons = (name: Keys) => _IconMapping[name] ?? null

function timeSince(date: Date | string) {
  const now = new Date()
  const dateObject = date instanceof Date ? date : new Date(date)
  const diff = Math.floor((now.getTime() - dateObject.getTime()) / 1000)
  const days = Math.floor(diff / (3600 * 24))
  const hours = Math.floor((diff % (3600 * 24)) / 3600)
  const minutes = Math.floor((diff % 3600) / 60)

  if (days > 1) {
    return `${days} days ago`
  } else if (days === 1) {
    return `1 day ago`
  } else if (hours > 1) {
    return `${hours} hours ago`
  } else if (hours === 1) {
    return `1 hour ago`
  } else if (minutes > 1) {
    return `${minutes} minutes ago`
  } else if (minutes === 1) {
    return `1 minute ago`
  } else {
    return `just now`
  }
}

function extractApexDomain(url: string) {
  const match = url.match(externalLinkRegex)
  return match ? match[1] : ""
}

let currentActive: HTMLLIElement | null = null
function createLinkEl(Link: Link): HTMLLIElement {
  const curiusItem = document.createElement("li")
  curiusItem.id = `curius-item-${Link.id}`
  curiusItem.classList.add("curius-item")

  const createTitle = (Link: Link): HTMLDivElement => {
    const item = document.createElement("div")
    item.classList.add("curius-item-title")

    const header = document.createElement("div")
    header.classList.add("curius-item-link")

    const link = document.createElement("a")
    Object.assign(link, {
      href: Link.link,
      target: "_blank",
      rel: "noopener noreferrer",
      innerHTML: `<span class="curius-item-span">${Link.title}</span>`,
    })
    header.appendChild(link)

    const address = document.createElement("div")
    address.classList.add("curius-item-address")
    address.textContent = extractApexDomain(Link.link)

    const icons = document.createElement("div")
    icons.classList.add("curius-item-icons")
    if (Link.favorite) {
      const icon = document.createElement("span")
      icon.classList.add("curius-favourite")
      icon.innerHTML = Icons("favourite")
      icons.appendChild(icon)
    }

    item.append(header, address, icons)
    return item
  }

  const createMetadata = (Link: Link): HTMLDivElement => {
    const item = document.createElement("div")
    item.classList.add("curius-item-metadata")

    const tags = document.createElement("ul")
    tags.classList.add("curius-item-tags")
    tags.innerHTML =
      Link.topics.length > 0
        ? `${Link.topics
            .map((topic) =>
              topic.public
                ? `<li><a href="https://curius.app/aaron-pham/${topic.slug}" target="_blank">${topic.topic}</a></li>`
                : ``,
            )
            .join("")}`
        : ``

    const misc = document.createElement("div")
    misc.id = `curius-misc-${Link.id}`
    const time = document.createElement("span")
    time.id = `curius-span-${Link.id}`
    const modifiedDate = new Date(Link.modifiedDate)
    time.innerHTML = `<time datetime=${
      Link.modifiedDate
    } title="${modifiedDate.toUTCString()}">${timeSince(Link.createdDate)}</time>`
    misc.appendChild(time)

    if (Link.highlights.length > 0) {
      const highlights = document.createElement("div")
      highlights.id = `curius-highlights-${Link.id}`
      highlights.innerHTML = `${Link.highlights.length} highlight`
      misc.appendChild(highlights)

      const modal = document.getElementById("highlight-modal")
      const modalList = document.getElementById("highlight-modal-list")

      const onMouseEnter = () => {
        const highlightsData = Link.highlights

        if (!modal || !modalList) return
        // clear the previous modal
        modalList.innerHTML = ""
        curiusItem.classList.remove("focus")

        highlightsData.forEach((highlight) => {
          let hiItem = document.createElement("li")
          hiItem.textContent = highlight.highlight
          modalList.appendChild(hiItem)
        })
        modal.style.visibility = "visible"
        modal.classList.add("active")
      }

      const onMouseLeave = () => {
        curiusItem.classList.add("focus")

        if (!modal) return
        modal.style.visibility = "hidden"
        modal.classList.remove("active")
      }

      const onMouseMove = ({ pageX, pageY }: MouseEvent) => {
        curiusItem.classList.remove("focus")

        if (!modal) return
        modal.classList.add("active")
        modal.style.left = `${pageX + 10}px`
        modal.style.top = `${pageY + 10}px`
      }

      registerEvents(
        highlights,
        ["mouseenter", onMouseEnter],
        ["mouseleave", onMouseLeave],
        ["mousemove", onMouseMove],
      )
    }

    item.append(tags, misc)
    return item
  }

  curiusItem.append(createTitle(Link), createMetadata(Link))
  curiusItem.dataset.items = JSON.stringify(true)

  const onMouseEnter = () => {
    curiusItem.classList.add("focus")
  }

  const onMouseLeave = () => {
    curiusItem.classList.remove("focus")
  }

  const onClick = (e: HTMLElementEventMap["click"]) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
    if (currentActive) currentActive.classList.remove("active")

    const note = document.getElementsByClassName("curius-notes")[0] as HTMLDivElement | null
    if (!note) return

    currentActive = curiusItem
    currentActive.classList.add("active")
    note.classList.add("active")
    updateNotePanel(Link, note, currentActive)
  }

  registerEscapeHandler(curiusItem, () => curiusItem.classList.remove("active"))
  registerEvents(
    curiusItem,
    ["mouseenter", onMouseEnter],
    ["mouseleave", onMouseLeave],
    ["click", onClick],
  )

  return curiusItem
}

function updateNotePanel(Link: Link, note: HTMLDivElement, parent: HTMLLIElement) {
  const titleNode = note.querySelector("#note-link") as HTMLAnchorElement
  const snippetNode = note.querySelector(".curius-note-snippet") as HTMLDivElement
  const highlightsNode = note.querySelector(".curius-note-highlights") as HTMLDivElement

  titleNode.innerHTML = `<span class="curius-item-span">${Link.title}</span>`
  titleNode.href = Link.link
  titleNode.target = "_blank"
  titleNode.rel = "noopener noreferrer"

  const close = document.querySelector(".icon-container")

  const cleanUp = () => {
    note.classList.remove("active")
    parent.classList.remove("active")
  }

  close?.addEventListener("click", cleanUp)
  window.addCleanup(() => close?.removeEventListener("click", cleanUp))
  registerEscapeHandler(note, cleanUp)

  removeAllChildren(snippetNode)
  snippetNode.textContent = Link.metadata ? Link.metadata.full_text : Link.snippet

  removeAllChildren(highlightsNode)
  if (Link.highlights.length === 0) return
  for (const hl of Link.highlights) {
    const highlightItem = document.createElement("li")
    const hlLink = document.createElement("a")
    hlLink.dataset.highlight = hl.id.toString()
    hlLink.href = `${Link.link}?curius=${hl.userId}`
    hlLink.target = "_blank"
    hlLink.rel = "noopener noreferrer"
    hlLink.textContent = hl.highlight
    highlightItem.appendChild(hlLink)
    highlightsNode.appendChild(highlightItem)
  }
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const elements = [
    "#curius-container",
    "#curius-fetching-text",
    "#curius-fragments",
    ".navigation-container",
    ".total-links",
  ].map((id) => document.querySelector(id))

  if (elements.some((el) => el === null)) return

  const [container, fetchText, fragment, nav, total] = elements as HTMLElement[]

  fetchText.textContent = "Récupération des liens curius"
  fetchText.classList.toggle("active", true)
  const resp = await fetchCuriusLinks()
  fetchText.classList.toggle("active", false)

  const linksData = resp.links ?? []

  const callIfEmpty = (data: Link[]) => {
    if (data.length === 0) {
      container.innerHTML = `<p>Échec de la récupération des liens.</p>`
      return
    }
  }

  callIfEmpty(linksData)
  fragment.append(...linksData.map(createLinkEl))
  total.textContent = `${linksData.length} éléments`
  nav.classList.toggle("active", true)

  const refetchIcon = document.getElementById("curius-refetch")

  // Ensure refetchIcon exists before adding event listener
  if (refetchIcon) {
    const preventRefreshDefault = (e: HTMLElementEventMap["keydown"]) => {
      const icon = document.getElementById("curius-refetch")
      if (!icon || !icon.classList.contains("disabled")) return
      if ((e.key === "r" || e.key === "R") && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        e.stopPropagation()
      }
    }

    document.addEventListener("keydown", preventRefreshDefault)
    window.addCleanup(() => document.removeEventListener("keydown", preventRefreshDefault))

    let isTimeout = false

    const onClick = async () => {
      if (isTimeout) return

      const searchContainer = document.getElementById(
        "curius-search-container",
      ) as HTMLDivElement | null

      if (searchContainer?.classList.contains("active")) {
        searchContainer.classList.remove("active")
      }

      refetchIcon.classList.add("disabled")
      refetchIcon.style.opacity = "0.5"

      removeAllChildren(fragment)
      nav.classList.toggle("active", false)

      fetchText.classList.toggle("active", true)
      fetchText.textContent = "Rafraîchissement des liens curius"
      const refetched = await fetchCuriusLinks(true)
      fetchText.classList.toggle("active", false)

      const newData = refetched.links ?? []
      callIfEmpty(newData)
      fragment.append(...newData.map(createLinkEl))
      total.textContent = `${newData.length} éléments`
      nav.classList.toggle("active", true)

      isTimeout = true
      setTimeout(() => {
        refetchIcon.classList.remove("disabled")
        refetchIcon.style.opacity = "0"
        isTimeout = false
      }, refetchTimeout)
    }

    refetchIcon.addEventListener("click", onClick)
    window.addCleanup(() => refetchIcon.removeEventListener("click", onClick))

    registerEvents(
      refetchIcon,
      [
        "mouseenter",
        () =>
          (refetchIcon.style.opacity = refetchIcon.classList.contains("disabled") ? "0.5" : "1"),
      ],
      [
        "mouseleave",
        () =>
          (refetchIcon.style.opacity = refetchIcon.classList.contains("disabled") ? "0.5" : "0"),
      ],
    )
  }
})
