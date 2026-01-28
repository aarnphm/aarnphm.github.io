import { Link } from "../../types/curius"
import {
  fetchCuriusLinks,
  fetchSearchLinks,
  createTrailMetadata,
  createTrailList,
  curiusSearch,
  createLinkEl,
} from "./curius"
import { updateNavigation } from "./curius-navigation.inline"

declare global {
  interface Window {
    curiusState?: { currentPage: number; hasMore: boolean }
  }
}

document.addEventListener("nav", async (e) => {
  if (e.detail.url !== "curius") return

  const elements = [".curius-page-container", "#curius-fetching-text", "#curius-fragments"].map(
    (selector) => document.querySelector<HTMLDivElement>(selector),
  )

  if (elements.some((el) => el === null)) return

  const [container, fetchText, fragment] = elements

  const friends = document.querySelector<HTMLUListElement>(".curius-friends")
  const trails = document.getElementsByClassName("curius-trail")[0] as HTMLDivElement

  fetchText!.textContent = "Récupération des liens curius"
  fetchText!.classList.toggle("active", true)

  const [resp, searchLinks] = await Promise.all([fetchCuriusLinks(), fetchSearchLinks()])

  fetchText!.classList.toggle("active", false)

  const callIfEmpty = (data: Link[]) => {
    if (data.length === 0) {
      container!.innerHTML = `<p>Échec de la récupération des liens.</p>`
      return []
    }
    return data.filter((link) => link.trails.length === 0)
  }

  const linksData = callIfEmpty(resp.links!)
  if (linksData.length === 0) return

  const trailMetadata = await createTrailMetadata(resp)
  createTrailList(trailMetadata)

  await curiusSearch(searchLinks)

  fragment!.append(...linksData.map(createLinkEl))

  if (friends) friends.classList.toggle("active", true)
  if (trails) trails.classList.toggle("active", true)

  // Store pagination state
  window.curiusState = {
    currentPage: resp.page ?? 0,
    hasMore: typeof resp.hasMore === "boolean" ? resp.hasMore : linksData.length > 0,
  }

  // Update navigation buttons
  updateNavigation()
})
