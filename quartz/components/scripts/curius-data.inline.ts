import { fetchCuriusLinks } from "./curius"

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  if (!e.detail.url.includes("curius")) return
  const data = await fetchCuriusLinks()

  const curius = document.getElementsByClassName("curius")[0] as HTMLDivElement | null
  if (!curius) return

  curius.dataset.content = JSON.stringify(data)
})
