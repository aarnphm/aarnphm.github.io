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
    .catch((err) => console.error(err))
  return res
}

document.addEventListener("nav", async (e) => {
  const curius = document.getElementById("curius-container")
  const description = document.getElementById("curius-description")

  const linkToHTML = (curiusLink: Link) => {
    const item = document.createElement("li")

    const itemLink = document.createElement("a")
    itemLink.classList.add("curius-item-link")
    itemLink.href = curiusLink.link
    itemLink.setAttribute("target", "_blank")
    itemLink.setAttribute("rel", "noopener noreferrer")
    itemLink.innerHTML = `${curiusLink.title}`

    const itemSpan = document.createElement("span")
    itemSpan.id = `curius-span-${curiusLink.id}`
    itemSpan.innerHTML = `${timeSince(curiusLink.createdDate)}`

    item.id = `curius-item-${curiusLink.id}`
    item.appendChild(itemLink)
    item.appendChild(itemSpan)
    return item
  }

  function displayLinks(finalLinks: Response) {
    if (!curius) return
    if (finalLinks.userSaved.length === 0) {
      curius.innerHTML = `<p>Failed to fetch links.</p>`
    } else {
      curius.append(...finalLinks.userSaved.map(linkToHTML))
    }
  }

  const links = await fetchLinks()
  console.log(links)
  if (!description) return
  const pItem = document.createElement("p")
  pItem.innerHTML = `${links.userSaved.length} of <em>many</em> on <a href="https://curius.app/aaron-pham" target="_blank">curius dot app</a>`
  description.appendChild(pItem)
  displayLinks(links)
})
