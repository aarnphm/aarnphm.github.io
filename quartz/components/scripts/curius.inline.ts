import { pluralize } from "../../util/lang"
import { removeAllChildren } from "./util"

interface Highlight {
  id: number
  userId: number
  linkId: number
  highlight: string
  createdDate: string
  leftContext: string
  rightContext: string
  rawHighlight: string
  comment_ids: string[]
  comment: string
}

interface Topic {
  id: number
  userId: number
  topic: string
  slug: string
  public: boolean
  createdDate: string
  modifiedDate: string
}

interface Link {
  id: number
  link: string
  title: string
  favorite: boolean
  snippet: string
  toRead: any
  createdDate: string
  modifiedDate: string
  lastCrawled: any
  topics: Topic[]
  highlights: Highlight[]
  userIds?: number[]
}

interface Response {
  userSaved: Link[]
}

const CURIUS = "https://curius.app/aaron-pham"

const timeSince = (date: Date | string) => {
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

async function fetchLinks(): Promise<Response> {
  const res = await fetch("https://raw.aarnphm.xyz/api/curius", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  })
    .then((res) => res.json())
    .then((data) => {
      data.userSaved.sort(
        (a: Link, b: Link) => new Date(b.createdDate).getTime() - new Date(a.createdDate).getTime(),
      )
      return data
    })
    .catch((err) => console.error(err))
  return res
}

let prevShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined

const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/
const extractApexDomain = (url: string) => {
  const match = url.match(externalLinkRegex)
  return match ? match[1] : ""
}

document.addEventListener("nav", async (e) => {
  const curius = document.getElementById("curius")
  const container = document.getElementById("curius-container")
  const description = document.getElementById("curius-description")

  if (!curius || !container || !description) return
  curius.innerHTML = `<p id="curius-fetching-text">Fetching curius links</p>`
  const links = await fetchLinks()
  curius.innerHTML = ""
  curius.append(description ?? "", container ?? "")

  const linkToHTML = (curiusLink: Link) => {
    const curiusItem = document.createElement("li")
    curiusItem.id = `curius-item-${curiusLink.id}`
    curiusItem.addEventListener("mouseenter", (e) => {
      e.target.style.backgroundColor = "var(--lightgray)"
    })
    curiusItem.addEventListener("mouseleave", (e) => {
      e.target.style.backgroundColor = ""
    })

    // create title: itemHeader - links
    const itemTitle = document.createElement("div")
    itemTitle.classList.add("curius-item-title")
    const itemHeader = document.createElement("div")
    itemHeader.classList.add("curius-item-link")
    const itemLink = document.createElement("a")
    itemLink.href = curiusLink.link
    itemLink.setAttribute("target", "_blank")
    itemLink.setAttribute("rel", "noopener noreferrer")
    itemLink.innerHTML = `<span class="curius-item-span">${curiusLink.title}</span>`
    itemHeader.appendChild(itemLink)
    const itemAddress = document.createElement("div")
    itemAddress.classList.add("curius-item-address")
    itemAddress.innerHTML = extractApexDomain(curiusLink.link)
    itemTitle.append(itemHeader, itemAddress)

    // metadata: tags\n ul -> {time since, highlights}
    const itemMetadata = document.createElement("div")
    itemMetadata.classList.add("curius-item-metadata")

    const itemTags = document.createElement("div")
    itemTags.classList.add("curius-item-tags")
    itemTags.innerHTML =
      curiusLink.topics.length > 0
        ? `${curiusLink.topics
            .map(
              (topic) =>
                `<ul><a href=${[CURIUS, topic.slug].join("/")} target="_blank">${
                  topic.topic
                }</a></ul>`,
            )
            .join("")}`
        : ``

    const misc = document.createElement("div")
    misc.id = `curius-misc-${curiusLink.id}`
    const itemTime = document.createElement("span")
    itemTime.id = `curius-span-${curiusLink.id}`
    const modifiedDate = new Date(curiusLink.modifiedDate)
    itemTime.innerHTML = `<time datetime=${
      curiusLink.modifiedDate
    } title="${modifiedDate.toUTCString()}">${timeSince(curiusLink.createdDate)}</time>`
    misc.appendChild(itemTime)

    if (curiusLink.highlights.length > 0) {
      const itemHighlights = document.createElement("div")
      itemHighlights.id = `curius-highlights-${curiusLink.id}`
      itemHighlights.innerHTML = `${pluralize(curiusLink.highlights.length, "highlight")}`
      misc.appendChild(itemHighlights)
    }
    itemMetadata.append(itemTags, misc)

    curiusItem.append(itemTitle, itemMetadata)
    return curiusItem
  }

  function displayLinks(finalLinks: Response) {
    if (!container) return
    if (finalLinks.userSaved.length === 0) {
      container.innerHTML = `<p>Failed to fetch links.</p>`
      return
    }

    const fragment = document.createDocumentFragment()
    finalLinks.userSaved.forEach((link) => {
      fragment.appendChild(linkToHTML(link))
    })
    container.append(fragment)
  }

  function displayDescription() {
    if (!description) return
    const item = document.createElement("p")
    item.innerHTML = `${links.userSaved.length} of <a href="${CURIUS}" target="_blank"><em>curius.app/aaron-pham</em></a>`
    description.appendChild(item)
    return description
  }

  function navigationHandler() {
    const navigation = document.createElement("div")
    navigation.classList.add("navigation-container")
    const navigationText = document.createElement("p")
    navigationText.innerHTML = `You might be interested in <a href="/dump/quotes">this</a> or <a href="/">that</a>`
    navigation.appendChild(navigationText)

    function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
      if (e.key === "e" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        window.location.pathname = "/"
      }
    }

    if (prevShortcutHandler) {
      document.removeEventListener("keydown", prevShortcutHandler)
    }

    document.addEventListener("keydown", shortcutHandler)
    prevShortcutHandler = shortcutHandler
    return navigation
  }

  displayDescription()
  displayLinks(links)
  curius.appendChild(navigationHandler())
})
