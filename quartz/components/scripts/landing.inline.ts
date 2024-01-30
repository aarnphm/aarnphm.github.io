//@ts-ignore
import popoverScript from "./popover.inline"
//@ts-ignore
import searchScript from "./search.inline"
//@ts-ignore
import graphScript from "./graph.inline"
import { registerEscapeHandler, removeAllChildren } from "./util"

function handleKeybindClick(ev: MouseEvent) {
  ev.preventDefault()

  const keybind = (ev?.target as HTMLElement).dataset.keybind
  if (!keybind) return
  const [modifier, key] = keybind.split("--")
  const eventProps = {
    ctrKey: modifier === "ctrl",
    metaKey: modifier === "cmd",
    shiftKey: modifier === "shift",
    altKey: modifier === "alt",
  }
  const sim = new KeyboardEvent("keydown", {
    ...eventProps,
    key: key.length === 1 ? key : key.toLowerCase(),
    bubbles: true,
    cancelable: true,
  })
  document.dispatchEvent(sim)
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  for (const modifier of document.querySelectorAll("#landing-keybind") as NodeListOf<HTMLElement>) {
    modifier.removeEventListener("click", handleKeybindClick)
    modifier.addEventListener("click", handleKeybindClick)
  }

  const slug = e.detail.url
  if (slug === "index") {
    const landingEmail = document.querySelector(
      "ul#socials > li > a#landing-mail.landing-links",
    ) as HTMLAnchorElement | null
    const modal = document.getElementById("email-modal") as HTMLDivElement

    function emailModalEnter() {
      modal.style.display = "block"
    }

    function emailModalMove(ev: MouseEvent) {
      modal.style.left = `${ev.pageX + 20}px`
      modal.style.top = `${ev.pageY - 20}px`
    }

    function emailModalLeave() {
      modal.style.display = "none"
    }

    landingEmail?.removeEventListener("mouseenter", emailModalEnter)
    landingEmail?.addEventListener("mouseenter", emailModalEnter)
    landingEmail?.removeEventListener("mousemove", emailModalEnter)
    landingEmail?.addEventListener("mousemove", emailModalEnter)
    landingEmail?.removeEventListener("mouseleave", emailModalLeave)
    landingEmail?.addEventListener("mouseleave", emailModalLeave)
  }
})

// keybind shortcut
let prevHomeShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
document.addEventListener("nav", (ev: CustomEventMap["nav"]) => {
  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      window.location.pathname = "/"
    }
  }

  if (prevHomeShortcutHandler) {
    document.removeEventListener("keydown", prevHomeShortcutHandler)
  }

  document.addEventListener("keydown", shortcutHandler)
  prevHomeShortcutHandler = shortcutHandler
})
