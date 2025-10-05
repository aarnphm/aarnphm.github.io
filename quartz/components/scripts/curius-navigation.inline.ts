import { removeAllChildren } from "./util"
import { Link, CuriusResponse } from "../types"
import { createLinkEl } from "./curius"

declare global {
  interface Window {
    curiusState?: {
      currentPage: number
      hasMore: boolean
    }
  }
}

const fetchLinksHeaders: RequestInit = {
  method: "GET",
  headers: { "Content-Type": "application/json" },
}

async function loadPage(page: number) {
  const fetchText = document.getElementById("curius-fetching-text")
  if (fetchText) {
    fetchText.textContent = "Récupération des liens curius"
    fetchText.classList.toggle("active", true)
  }

  const resp = await fetch(
    `https://aarnphm.xyz/api/curius?query=links&page=${page}`,
    fetchLinksHeaders,
  )

  if (fetchText) {
    fetchText.classList.toggle("active", false)
  }

  if (!resp.ok) {
    throw new Error("Failed to load page")
  }

  const data: CuriusResponse = await resp.json()
  return data
}

async function renderPage(page: number) {
  const fragment = document.getElementById("curius-fragments")
  if (!fragment) return

  const data = await loadPage(page)
  if (!data || !data.links) return

  const linksData = data.links.filter((link: Link) => link.trails.length === 0)
  removeAllChildren(fragment)
  fragment.append(...linksData.map(createLinkEl))

  // Update state
  window.curiusState = {
    currentPage: page,
    hasMore: data.hasMore ?? false,
  }

  updateNavigation()
}

export function updateNavigation() {
  const prevButton = document.getElementById("curius-prev")
  const nextButton = document.getElementById("curius-next")

  if (!prevButton || !nextButton) return

  const { currentPage, hasMore } = window.curiusState || { currentPage: 0, hasMore: false }

  prevButton.style.visibility = currentPage > 0 ? "visible" : "hidden"
  prevButton.style.cursor = currentPage > 0 ? "pointer" : "default"

  nextButton.style.visibility = hasMore ? "visible" : "hidden"
  nextButton.style.cursor = hasMore ? "pointer" : "default"
}

document.addEventListener("nav", async (e) => {
  if (e.detail.url !== "curius") return

  const prevButton = document.getElementById("curius-prev")
  const nextButton = document.getElementById("curius-next")

  if (!prevButton || !nextButton) return

  const onPrevClick = async () => {
    const { currentPage } = window.curiusState || { currentPage: 0 }
    if (currentPage > 0) {
      await renderPage(currentPage - 1)
    }
  }

  const onNextClick = async () => {
    const { currentPage, hasMore } = window.curiusState || { currentPage: 0, hasMore: false }
    if (hasMore) {
      await renderPage(currentPage + 1)
    }
  }

  prevButton.addEventListener("click", onPrevClick)
  nextButton.addEventListener("click", onNextClick)

  window.addCleanup(() => prevButton.removeEventListener("click", onPrevClick))
  window.addCleanup(() => nextButton.removeEventListener("click", onNextClick))
})
