import { fetchCuriusLinks } from "./curius"

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  if (e.detail.url.includes("curius")) await fetchCuriusLinks()
})
