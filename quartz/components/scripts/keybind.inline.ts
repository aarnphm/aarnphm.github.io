import { registerEscapeHandler, registerEvents } from "./util"

type Browser = "Safari" | "Chrome" | "Firefox" | "Edge" | "Opera" | "Other"

const detectBrowser = (): Browser => {
  const userAgent = window.navigator.userAgent.toLowerCase()

  if (userAgent.includes("safari") && !userAgent.includes("chrome")) {
    return "Safari"
  } else if (userAgent.includes("edg")) {
    return "Edge"
  } else if (userAgent.includes("firefox")) {
    return "Firefox"
  } else if (userAgent.includes("opr") || userAgent.includes("opera")) {
    return "Opera"
  } else if (userAgent.includes("chrome")) {
    return "Chrome"
  }
  return "Other"
}

const isMacOS = (): boolean => {
  return window.navigator.userAgent.toLowerCase().includes("mac")
}

document.addEventListener("nav", async () => {
  const keybind = document.getElementsByClassName("keybind")[0] as HTMLDivElement | null
  if (!keybind) return

  const container = keybind.querySelector("#shortcut-container")
  const shortcutKey = keybind.querySelector("#shortcut-key") as HTMLElement

  const center = document.querySelector(".center") as HTMLElement | null
  const right = document.querySelector(".right.sidebar") as HTMLElement | null
  const left = document.querySelector(".left.sidebar") as HTMLElement | null

  const showContainer = () => {
    if (center) center.style.zIndex = "-1"
    if (right) right.style.zIndex = "-1"
    if (left) left.style.zIndex = "-1"
    container?.classList.add("active")
  }

  const hideContainer = () => {
    if (center) center.style.zIndex = ""
    if (right) right.style.zIndex = ""
    if (left) left.style.zIndex = ""
    container?.classList.remove("active")
  }

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (!shortcutKey) return
    for (const binding of JSON.parse(shortcutKey.dataset.mapping as string)) {
      const [, key] = binding.split("--")

      if (e.key === key && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        const containerOpen = container?.classList.contains("active")
        containerOpen ? hideContainer() : showContainer()
        break
      }
    }
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
  registerEscapeHandler(keybind, hideContainer)
})

const _mapping = new Map([
  ["\\", "/"],
  ["j", "/curius"],
])

const aliases: Record<string, { mac: string; def: string }> = {
  recherche: { mac: "/", def: "k" },
  graphique: { mac: ";", def: "g" },
}

document.addEventListener("nav", () => {
  const container = document.getElementById("shortcut-container")
  if (!container) return

  const shortcuts = container.querySelectorAll(
    'ul[id="shortcut-list"] > li > div[id="shortcuts"]',
  ) as NodeListOf<HTMLElement>
  for (const short of shortcuts) {
    const binding = short.dataset.key as string
    const span = short.dataset.value as string
    let [prefix, key] = binding.split("--")
    const spanAliases = aliases[span]
    prefix = isMacOS() ? "⌘" : "⌃"
    const browser = detectBrowser()
    if (spanAliases) {
      const { mac, def } = spanAliases
      key = browser === "Safari" ? mac : def
    }
    short.innerHTML = `
<kbd class="clickable">${prefix} ${key}</kbd>
<span>${span}</span>
`
  }

  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (_mapping.get(e.key) !== undefined && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const loc = _mapping.get(e.key) as string
      container?.classList.toggle("active", false)
      if (window.location.pathname === loc) return
      window.spaNavigate(new URL(loc, window.location.toString()))
    }
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
})
