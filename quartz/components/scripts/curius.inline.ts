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
}

interface Response {
  links: Link[]
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
  const res = await fetch("http://localhost:8001/api", {
    headers: { "Content-Type": "application/json" },
  })
    .then((res) => res.json())
    .catch((err) => console.error(err))
  return res
}

document.addEventListener("nav", async (e) => {
  const curius = document.getElementById("curius-container")

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
    if (finalLinks.links.length === 0) {
      curius.innerHTML = `<p>Failed to fetch links.</p>`
    } else {
      curius.append(...finalLinks.links.map(linkToHTML))
    }
  }

  const links = await fetchLinks()
  displayLinks(links)
})
