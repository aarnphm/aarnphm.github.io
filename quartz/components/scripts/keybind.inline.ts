import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"

document.addEventListener("nav", async () => {
  const modal = document.getElementById("highlight-modal")
  const container = document.getElementById("shortcut-container")
  const shortcutKey = document.getElementById("shortcut-key")
  const keybind = document.getElementsByClassName("keybind")[0] as HTMLDivElement | null

  const center = document.querySelector(".center") as HTMLElement | null
  const sidebar = document.querySelector(".right.sidebar") as HTMLElement | null

  const showContainer = () => {
    if (center) center.style.zIndex = "-1"
    if (sidebar) sidebar.style.zIndex = "-1"
    container?.classList.add("active")
  }

  const hideContainer = () => {
    if (center) center.style.zIndex = "unset"
    if (sidebar) sidebar.style.zIndex = "unset"
    container?.classList.remove("active")
  }

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (!shortcutKey) return
    for (const binding of JSON.parse(shortcutKey.dataset.mapping as string)) {
      const [, key] = binding.split("--")
      if (modal) hideModal()

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

  if (!modal) return

  const onMouseEnter = () => {
    if (container?.classList.contains("active")) return
    modal.classList.add("active")
    modal.style.visibility = "visible"
  }

  const hideModal = () => {
    modal.classList.remove("active")
    modal.style.visibility = "hidden"
  }

  const onMouseLeave = () => {
    if (container?.classList.contains("active")) return
    hideModal()
  }

  const onMouseMove = ({ pageX, pageY }: MouseEvent) => {
    if (container?.classList.contains("active")) return
    modal.classList.add("active")
    Object.assign(modal.style, {
      left: `${pageX + 10}px`,
      top: `${pageY + 10}px`,
    })
  }
  const events = [
    ["mouseenter", onMouseEnter],
    ["mouseleave", onMouseLeave],
    ["mousemove", onMouseMove],
  ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]
  registerEvents(keybind, ...events)
})

const _mapping = new Map([
  ["\\", "/"],
  ["j", "/curius"],
])

document.addEventListener("nav", () => {
  const container = document.getElementById("shortcut-container")
  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (_mapping.get(e.key) !== undefined && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const loc = _mapping.get(e.key) as string
      container?.classList.toggle("active", false)
      if (window.location.pathname === loc) return
      window.location.href = loc
    }
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
})
