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

document.addEventListener("nav", async (e: unknown) => {
  for (const modifier of document.querySelectorAll("#landing-keybind") as NodeListOf<HTMLElement>) {
    modifier.removeEventListener("click", handleKeybindClick)
    modifier.addEventListener("click", handleKeybindClick)
  }

  const landingEmail = document.querySelector(
    "ul#socials > li > a#landing-mail.landing-links",
  ) as HTMLAnchorElement
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

  landingEmail.removeEventListener("mouseenter", emailModalEnter)
  landingEmail.addEventListener("mouseenter", emailModalEnter)
  landingEmail.removeEventListener("mousemove", emailModalEnter)
  landingEmail.addEventListener("mousemove", emailModalEnter)
  landingEmail.removeEventListener("mouseleave", emailModalLeave)
  landingEmail.addEventListener("mouseleave", emailModalLeave)
})
