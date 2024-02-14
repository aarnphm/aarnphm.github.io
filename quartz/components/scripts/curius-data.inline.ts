import { fetchCuriusLinks } from "./curius"

document.addEventListener("nav", async () => await fetchCuriusLinks())
