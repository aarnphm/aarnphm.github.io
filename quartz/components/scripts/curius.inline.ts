import { pluralize } from "../../util/lang"

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
  topics: Object[]
  highlights: Highlight[]
  userIds?: number[]
}

interface Response {
  userSaved: Link[]
}

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
  const res = await fetch("https://curius.aarnphm.xyz/api/curius", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  })
    .then((res) => res.json())
    .then((data) => {
      data.userSaved.sort((a: Link, b: Link) => {
        return new Date(b.createdDate).getTime() - new Date(a.createdDate).getTime()
      })
      return data
    })
    .catch((err) => console.error(err))
  return res
}

document.addEventListener("nav", async (e) => {
  const curius = document.getElementById("curius")
  const curiusContainer = document.getElementById("curius-container")
  const curiusDescription = document.getElementById("curius-description")

  if (!curius) return
  curius.innerHTML = `<p>Fetching curius links...</p>`
  const links = await fetchLinks()
  curius.innerHTML = ""
  curius.append(curiusDescription ?? "", curiusContainer ?? "")

  const linkToHTML = (curiusLink: Link) => {
    const item = document.createElement("li")
    item.id = `curius-item-${curiusLink.id}`

    const itemTitle = document.createElement("h5")
    const itemLink = document.createElement("a")
    itemLink.classList.add("curius-item-link")
    itemLink.href = curiusLink.link
    itemLink.setAttribute("target", "_blank")
    itemLink.setAttribute("rel", "noopener noreferrer")
    itemLink.innerHTML = `${curiusLink.title}`
    itemTitle.appendChild(itemLink)

    const metadata = document.createElement("ul")
    metadata.id = `curius-metadata-${curiusLink.id}`
    const itemSpan = document.createElement("li")
    itemSpan.id = `curius-span-${curiusLink.id}`
    itemSpan.innerHTML = `${timeSince(curiusLink.createdDate)}`
    metadata.appendChild(itemSpan)

    if (curiusLink.highlights.length > 0) {
      const itemHighlights = document.createElement("li")
      itemHighlights.id = `curius-highlights-${curiusLink.id}`
      itemHighlights.innerHTML = `${pluralize(curiusLink.highlights.length, "highlight")}`
      metadata.appendChild(itemHighlights)
    }

    item.append(itemTitle, metadata)
    return item
  }

  function displayLinks(finalLinks: Response) {
    if (!curiusContainer) return
    if (finalLinks.userSaved.length === 0) {
      curiusContainer.innerHTML = `<p>Failed to fetch links.</p>`
    } else {
      curiusContainer.append(...finalLinks.userSaved.map(linkToHTML))
    }
  }

  if (!curiusDescription) return
  const pItem = document.createElement("p")
  pItem.innerHTML = `${links.userSaved.length} of <a href="https://curius.app/aaron-pham" target="_blank"><em>curius.app/aaron-pham</em></a>`
  curiusDescription.appendChild(pItem)

  displayLinks(links)

  const navigation = document.createElement("div")
  navigation.classList.add("navigation-container")
  const navigationText = document.createElement("p")
  navigationText.innerHTML = `You might be interested in <a href="/dump/quotes" class="internal">this</a>`
  navigation.appendChild(navigationText)
  curius.appendChild(navigation)
})
