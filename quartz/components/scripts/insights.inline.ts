const SCRIPT_URL = "https://va.vercel-scripts.com/v1/speed-insights"
const PROD_SCRIPT_URL = `${SCRIPT_URL}/script.js`
const DEV_SCRIPT_URL = `${SCRIPT_URL}/script.debug.js`

function detectEnvironment() {
  try {
    const env = process.env.NODE_ENV
    if (env === "development" || env === "test") {
      return "development"
    }
  } catch (e) {}
  return "production"
}
function isDevelopment() {
  return detectEnvironment() === "development"
}
function isBrowser() {
  return typeof window !== "undefined"
}

if (isBrowser()) {
  const script = document.createElement("script")
  const src = isDevelopment() ? DEV_SCRIPT_URL : PROD_SCRIPT_URL
  script.src = src
  script.defer = true
  script.dataset.sdkv = "1.1.0"
  script.dataset.sdkn = "@vercel/speed-insights"
  script.dataset.endpoint = `https://cdn.aarnphm.xyz/_vercel/speed-insights/vitals`
  script.onerror = () => {
    console.log(
      `[Vercel Speed Insights] Failed to load script from ${src}. Please check if any content blockers are enabled and try again.`,
    )
  }
  document.head.appendChild(script)

  document.addEventListener("nav", (ev) => {
    script.dataset.route = `/${ev.detail.url}`
  })
}
