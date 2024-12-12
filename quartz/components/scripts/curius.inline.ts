import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"
import { Link } from "../types"
import {
  fetchCuriusLinks,
  createTitle,
  timeSince,
  createTrailMetadata,
  createTrailList,
  curiusSearch,
} from "./curius"

const refetchTimeout = 2 * 60 * 1000 // 2 minutes

let currentActive: HTMLLIElement | null = null
function createLinkEl(Link: Link): HTMLLIElement {
  const curiusItem = document.createElement("li")
  curiusItem.id = `curius-item-${Link.id}`
  curiusItem.classList.add("curius-item")

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
      highlights.innerHTML = `${Link.highlights.length} ${Link.highlights.length > 0 ? "highlights" : "highlight"}`
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

  curiusItem.append(createTitle({ Link, addFaIcon: true }), createMetadata(Link))
  curiusItem.dataset.items = JSON.stringify(true)

  const onClick = (e: HTMLElementEventMap["click"]) => {
    const note = document.getElementsByClassName("curius-notes")[0] as HTMLDivElement | null

    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
    if (currentActive) currentActive.classList.remove("active")
    if (note) note.classList.remove("active")

    currentActive = curiusItem
    currentActive.classList.add("active")

    if (Link.highlights.length > 0) {
      if (!note) return
      note.classList.add("active")
      updateNotePanel(Link, note, currentActive)
    }

    if (e.target instanceof HTMLAnchorElement || note?.classList.contains("active")) return
    window.open(Link.link, "_blank")
  }

  registerEscapeHandler(curiusItem, () => curiusItem.classList.remove("active"))

  const onMouseEnter = () => {
    const favoriteDiv = curiusItem.querySelector(".curius-item-favorite") as HTMLDivElement | null

    if (favoriteDiv) favoriteDiv.classList.add("focus")
    curiusItem.classList.add("focus")
  }

  const onMouseLeave = () => {
    const favoriteDiv = curiusItem.querySelector(".curius-item-favorite") as HTMLDivElement | null

    if (favoriteDiv) favoriteDiv.classList.remove("focus")
    curiusItem.classList.remove("focus")
  }

  registerEvents(
    curiusItem,
    ["click", onClick],
    ["mouseenter", onMouseEnter],
    ["mouseleave", onMouseLeave],
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

document.addEventListener("nav", async (e) => {
  if (e.detail.url !== "curius") return

  const elements = [".curius-page-container", "#curius-fetching-text", "#curius-fragments"].map(
    (id) => document.querySelector(id),
  )

  if (elements.some((el) => el === null)) return

  const [container, fetchText, fragment] = elements as HTMLElement[]

  const friends = document.querySelector(".curius-friends") as HTMLUListElement | null
  const trails = document.getElementsByClassName("curius-trail")[0] as HTMLDivElement | null

  fetchText.textContent = "Récupération des liens curius"
  fetchText.classList.toggle("active", true)
  const resp = await fetchCuriusLinks()
  fetchText.classList.toggle("active", false)

  const callIfEmpty = (data: Link[]) => {
    if (data.length === 0) {
      container.innerHTML = `<p>Échec de la récupération des liens.</p>`
      return []
    }
    return data.filter((link) => link.trails.length === 0)
  }

  // show trails separately
  const linksData = callIfEmpty(resp.links!)
  if (linksData.length === 0) return

  createTrailList(createTrailMetadata(resp))

  await curiusSearch(linksData)

  fragment.append(...linksData.map(createLinkEl))
  if (friends) friends.classList.toggle("active", true)
  if (trails) trails.classList.toggle("active", true)

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

      const trail = document.getElementById("trail-list") as HTMLUListElement | null
      const note = document.getElementsByClassName("curius-notes")[0] as HTMLDivElement | null

      removeAllChildren(fragment)
      removeAllChildren(trail!)

      note?.classList.toggle("active", false)
      nav.classList.toggle("active", false)
      friends?.classList.toggle("active", false)
      trails?.classList.toggle("active", false)

      fetchText.classList.toggle("active", true)
      fetchText.textContent = "Rafraîchissement des liens curius"
      const refetched = await fetchCuriusLinks()
      fetchText.classList.toggle("active", false)

      createTrailList(createTrailMetadata(refetched))

      const newData = callIfEmpty(refetched.links!)
      if (newData.length === 0) return

      await curiusSearch(refetched.links!)

      fragment.append(...newData.map(createLinkEl))
      friends?.classList.toggle("active", true)
      trails?.classList.toggle("active", true)

      isTimeout = true
      setTimeout(() => {
        refetchIcon.classList.remove("disabled")
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
          (refetchIcon.style.opacity = refetchIcon.classList.contains("disabled") ? "0.5" : "1"),
      ],
    )
  }
})
