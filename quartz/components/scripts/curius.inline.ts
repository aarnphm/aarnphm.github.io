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

interface User {
  id: number
  firstName: string
  lastName: string
  major?: string
  interests?: string
  expertise?: string
  school: string
  github?: string
  twitter: string
  website: string
  createdDate: string
  modifiedDate: string
  lastOnline: string
  lastCheckedNotifications: string
  views: number
  numFollowers: number
  followed?: boolean
  followingMe?: boolean
  recentUsers: any[]
  followingUsers: Following[]
}

interface Following {
  id: number
  firstName: string
  lastName: string
  userLink: string
  lastOnline: string
}

interface Response {
  links?: Link[]
  user?: User
  lastFetched: string
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

const fetchLinksHeaders: RequestInit = {
  method: "POST",
  headers: { "Content-Type": "application/json" },
}

const localFetchKey = "curiusLinks"
const localTimeKey = "curiusLastFetch"

const getLocalItem = (key: "curiusLinks" | "curiusLastFetch", value: any): any =>
  localStorage.getItem(key) ?? value

async function fetchLinks(): Promise<Response> {
  try {
    // user metadata
    const user = await fetch("https://raw.aarnphm.xyz/api/curius?query=user", fetchLinksHeaders)
      .then((res): Promise<Response> => res.json())
      .then((data) => {
        if (data === undefined || data.user === undefined) {
          throw new Error("Failed to fetch user")
        }
        return data.user
      })

    const currentTime = new Date()
    const lastFetched = new Date(getLocalItem(localTimeKey, 0))
    // set fetched period to 5 minutes
    const periods = 5 * 60 * 1000

    const getLocalLinks = () => JSON.parse(getLocalItem(localFetchKey, "[]"))

    if (currentTime.getTime() - lastFetched.getTime() < periods) {
      return { links: getLocalLinks(), user: user, lastFetched: lastFetched.toString() }
    }

    localStorage.setItem("curiusLastFetch", currentTime.toString())

    // fetch new links
    const links: Link[] = await fetch(
      "https://raw.aarnphm.xyz/api/curius?query=links",
      fetchLinksHeaders,
    )
      .then((res) => res.json())
      .then((data: Response) => {
        if (data === undefined || data.links === undefined) {
          throw new Error("Failed to fetch links")
        }
        data.links.sort(
          (a: Link, b: Link) =>
            new Date(b.createdDate).getTime() - new Date(a.createdDate).getTime(),
        )
        return data.links
      })

    const existingLinks = getLocalLinks()
    if (JSON.stringify(existingLinks) !== JSON.stringify(links)) {
      localStorage.setItem("curiusLinks", JSON.stringify(links))
    }
    return { links, user, lastFetched: lastFetched.toString() }
  } catch (err) {
    console.error(err)
    throw new Error("Failed to fetch links")
  }
}

const externalLinkRegex = /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/
const extractApexDomain = (url: string) => {
  const match = url.match(externalLinkRegex)
  return match ? match[1] : ""
}

const createLinkEl = (Link: Link): HTMLLIElement => {
  const curiusItem = document.createElement("li")
  curiusItem.id = `curius-item-${Link.id}`
  curiusItem.onmouseenter = () => (curiusItem.style.backgroundColor = "var(--lightgray)")
  curiusItem.onmouseleave = () => (curiusItem.style.backgroundColor = "")

  const createTitle = (Link: Link): HTMLDivElement => {
    const item = document.createElement("div")
    item.classList.add("curius-item-title")

    const header = document.createElement("div")
    const link = document.createElement("a")
    link.href = Link.link
    link.target = "_blank"
    link.rel = "noopener noreferrer"
    link.innerHTML = `<span class="curius-item-span">${Link.title}</span>`
    header.classList.add("curius-item-link")
    header.appendChild(link)

    const address = document.createElement("div")
    address.classList.add("curius-item-address")
    address.textContent = extractApexDomain(Link.link)

    item.append(header, address)
    return item
  }

  const createMetadata = (Link: Link): HTMLDivElement => {
    const item = document.createElement("div")
    item.classList.add("curius-item-metadata")

    const tags = document.createElement("div")
    tags.classList.add("curius-item-tags")
    tags.innerHTML =
      Link.topics.length > 0
        ? `${Link.topics
            .map(
              (topic) =>
                `<ul><a href=${[CURIUS, topic.slug].join("/")} target="_blank">${
                  topic.topic
                }</a></ul>`,
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
      highlights.innerHTML = `${pluralize(Link.highlights.length, "highlight")}`
      misc.appendChild(highlights)
    }

    item.append(tags, misc)
    return item
  }

  ;[createTitle, createMetadata].forEach((fn) => curiusItem.appendChild(fn(Link)))

  return curiusItem
}

document.addEventListener("nav", async (e: unknown) => {
  // create an array of all these elements by Id functionally
  const elements = ["curius", "curius-container", "curius-description"].map((id) =>
    document.getElementById(id),
  )

  if (elements.some((el) => el === null)) return
  const [curius, container, description] = elements as HTMLElement[]

  const fetching = document.createElement("div")
  fetching.id = "curius-fetching-text"
  fetching.textContent = "Fetching curius links"
  curius.appendChild(fetching)
  const resp = await fetchLinks()
  curius.removeChild(fetching)

  const linksData = resp.links ?? []
  const userData = resp.user ?? {}

  const item = document.createElement("p")
  const time = document.createElement("p")
  time.innerHTML = `<em>last fetched: ${new Date(resp.lastFetched).toUTCString()}</em>`
  const titleLink = document.createElement("span")
  titleLink.textContent = `${linksData.length} of `
  const curiusLink = document.createElement("a")
  curiusLink.href = CURIUS
  curiusLink.target = "_blank"
  curiusLink.textContent = "curius.app/aaron-pham"
  titleLink.append(curiusLink)
  item.append(titleLink, time)
  description.appendChild(item)

  const fragment = document.createDocumentFragment()
  if (linksData.length === 0) {
    container.innerHTML = `<p>Failed to fetch links.</p>`
    return
  }
  linksData.map((link) => fragment.appendChild(createLinkEl(link)))
  container.append(fragment)

  const navigation = document.createElement("div")
  navigation.classList.add("navigation-container")
  const navigationText = document.createElement("p")
  navigationText.innerHTML = `You might be interested in <a href="/dump/quotes">this</a> or <a href="/">that</a>`
  navigation.appendChild(navigationText)
  curius.append(navigation)
})
