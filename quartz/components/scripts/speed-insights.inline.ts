import { injectSpeedInsights } from "@vercel/speed-insights"
import { getFullSlug } from "../../util/path"

const insights = injectSpeedInsights()

document.addEventListener("nav", async () => {
  insights?.setRoute(getFullSlug(window))
})
