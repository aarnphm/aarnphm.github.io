import { fetchTrails } from "./curius"

document.addEventListener("nav", async () => await fetchTrails())
