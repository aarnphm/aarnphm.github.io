//@ts-ignore
import popoverScript from "./popover.inline"
//@ts-ignore
import searchScript from "./search.inline"
//@ts-ignore
import graphScript from "./graph.inline"

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
    modifier.addEventListener("click", handleKeybindClick)
    window.addCleanup(() => modifier.removeEventListener("click", handleKeybindClick))
  }
})
