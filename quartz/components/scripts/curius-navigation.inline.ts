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

  const resp = await fetch(`/api/curius?query=links&page=${page}`, fetchLinksHeaders)

  if (fetchText) {
    fetchText.classList.toggle("active", false)
  }

  if (!resp.ok) {
    throw new Error("Failed to load page")
  }

  const data: CuriusResponse = await resp.json()
  return data
}

async function renderPage(page: number): Promise<boolean> {
  const fragment = document.getElementById("curius-fragments")
  if (!fragment) return false

  let data: CuriusResponse
  try {
    data = await loadPage(page)
  } catch (error) {
    console.error(error)
    return false
  }

  if (!data || !data.links) {
    return false
  }

  const linksData = data.links.filter((link: Link) => link.trails.length === 0)

  if (linksData.length === 0) {
    if (page === 0) {
      removeAllChildren(fragment)
      fragment.innerHTML = `<p>Échec de la récupération des liens.</p>`
      window.curiusState = { currentPage: 0, hasMore: false }
      updateNavigation()
    } else {
      const fetchText = document.getElementById("curius-fetching-text")
      if (fetchText) {
        fetchText.textContent = "Pas d'autres liens pour le moment"
        fetchText.classList.toggle("active", true)
        window.setTimeout(() => fetchText.classList.toggle("active", false), 1500)
      }

      const prevState = window.curiusState || { currentPage: 0, hasMore: false }
      window.curiusState = { ...prevState, hasMore: false }
      updateNavigation()
    }

    return false
  }

  removeAllChildren(fragment)
  fragment.append(...linksData.map(createLinkEl))

  window.curiusState = {
    currentPage: page,
    hasMore: typeof data.hasMore === "boolean" ? data.hasMore : linksData.length > 0,
  }

  updateNavigation()
  return true
}

export function updateNavigation() {
  const prevButton = document.getElementById("curius-prev")
  const nextButton = document.getElementById("curius-next")

  if (!prevButton || !nextButton) return

  const state = window.curiusState || { currentPage: 0, hasMore: true }
  const canGoPrev = state.currentPage > 0
  const canGoNext = state.hasMore !== false

  prevButton.classList.toggle("disabled", !canGoPrev)
  prevButton.setAttribute("aria-disabled", String(!canGoPrev))

  nextButton.classList.toggle("disabled", !canGoNext)
  nextButton.setAttribute("aria-disabled", String(!canGoNext))
}

let isNavigating = false

document.addEventListener("nav", async (e) => {
  if (e.detail.url !== "curius") return

  const prevButton = document.getElementById("curius-prev")
  const nextButton = document.getElementById("curius-next")

  if (!prevButton || !nextButton) return

  const onPrevClick = async () => {
    if (isNavigating) return

    const { currentPage = 0 } = window.curiusState || {}
    if (currentPage <= 0) return

    isNavigating = true
    try {
      await renderPage(currentPage - 1)
    } finally {
      isNavigating = false
    }
  }

  const onNextClick = async () => {
    if (isNavigating) return

    const state = window.curiusState || { currentPage: 0, hasMore: true }
    if (state.hasMore === false) {
      updateNavigation()
      return
    }

    isNavigating = true
    try {
      const nextPage = (state.currentPage || 0) + 1
      const loaded = await renderPage(nextPage)
      if (!loaded) {
        updateNavigation()
      }
    } finally {
      isNavigating = false
    }
  }

  prevButton.addEventListener("click", onPrevClick)
  nextButton.addEventListener("click", onNextClick)

  window.addCleanup(() => prevButton.removeEventListener("click", onPrevClick))
  window.addCleanup(() => nextButton.removeEventListener("click", onNextClick))
})
