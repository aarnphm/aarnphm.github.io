//@ts-ignore
import popoverScript from "./popover.inline"
//@ts-ignore
import searchScript from "./search.inline"
//@ts-ignore
import graphScript from "./graph.inline"
import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"

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
    registerEvents(modifier, ["click", handleKeybindClick])
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

    const events = [
      ["mouseenter", emailModalEnter],
      ["mousemove", emailModalMove],
      ["mouseleave", emailModalLeave],
    ] as [keyof HTMLElementEventMap, (ev: MouseEvent) => void][]

    registerEvents(
      landingEmail,
      ["mouseenter", () => (modal.style.display = "block")],
      ["mouseleave", () => (modal.style.display = "none")],
      [
        "mousemove",
        ({ pageX, pageY }) =>
          Object.assign(modal.style, { left: `${pageX + 20}px`, top: `${pageY - 20}px` }),
      ],
    )
  }
})
