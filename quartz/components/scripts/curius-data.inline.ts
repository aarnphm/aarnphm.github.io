import { Link, User, CuriusResponse } from "../types"

const fetchLinksHeaders: RequestInit = {
  method: "POST",
  headers: { "Content-Type": "application/json" },
}

const localFetchKey = "curiusLinks"
const localTimeKey = "curiusLastFetch"

function getLocalItem(key: "curiusLinks" | "curiusLastFetch", value: any) {
  return localStorage.getItem(key) ?? value
}

export async function fetchCuriusLinks(refetch: boolean = false): Promise<CuriusResponse> {
  // user metadata
  const user = await fetch("https://raw.aarnphm.xyz/api/curius?query=user", fetchLinksHeaders)
    .then((res): Promise<CuriusResponse> => res.json())
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

  const getCachedLinks = () => JSON.parse(getLocalItem(localFetchKey, "[]"))

  if (!refetch && currentTime.getTime() - lastFetched.getTime() < periods) {
    return { links: getCachedLinks(), user }
  }

  localStorage.setItem(localTimeKey, currentTime.toString())

  // fetch new links
  const newLinks: Link[] = await fetch(
    "https://raw.aarnphm.xyz/api/curius?query=links",
    fetchLinksHeaders,
  )
    .then((res) => res.json())
    .then((data: CuriusResponse) => {
      if (data === undefined || data.links === undefined) {
        throw new Error("Failed to fetch links")
      }
      return data.links
    })

  if (JSON.stringify(getCachedLinks()) !== JSON.stringify(newLinks)) {
    localStorage.setItem(localFetchKey, JSON.stringify(newLinks))
  }
  return { links: newLinks, user }
}

document.addEventListener("nav", async () => {
  await fetchCuriusLinks()
})
