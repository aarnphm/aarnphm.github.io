export function registerEscapeHandler(outsideContainer: HTMLElement | null, cb: () => void) {
  if (!outsideContainer) return
  function click(this: HTMLElement, e: HTMLElementEventMap["click"]) {
    if (e.target !== this) return
    e.preventDefault()
    e.stopPropagation()
    cb()
  }

  function esc(e: HTMLElementEventMap["keydown"]) {
    if (!e.key.startsWith("Esc")) return
    e.preventDefault()
    cb()
  }

  outsideContainer?.addEventListener("click", click)
  window.addCleanup(() => outsideContainer?.removeEventListener("click", click))
  document.addEventListener("keydown", esc)
  window.addCleanup(() => document.removeEventListener("keydown", esc))
}

export function removeAllChildren(node: HTMLElement) {
  while (node.firstChild) {
    node.removeChild(node.firstChild)
  }
}

export function registerMouseHover(el: HTMLElement, ...classList: string[]) {
  const onMouseEnter = () => el.classList.add(...classList)
  const onMouseLeave = () => el.classList.remove(...classList)

  registerEvents(el, ["mouseenter", onMouseEnter], ["mouseleave", onMouseLeave])
}

type EventType = HTMLElementEventMap[keyof HTMLElementEventMap]
type EventHandlers<E extends EventType> = (evt: E) => any

export function registerEvents<
  T extends Document | HTMLElement | null,
  E extends keyof HTMLElementEventMap,
>(element: T, ...events: [E, EventHandlers<HTMLElementEventMap[E]>][]) {
  if (!element) return

  events.forEach(([event, cb]) => {
    const listener: EventListener = (evt) => cb(evt as HTMLElementEventMap[E])
    element.addEventListener(event, listener)
    window.addCleanup(() => element.removeEventListener(event, listener))
  })
}

export function decodeString(el: HTMLSpanElement, targetString: string, duration: number = 1000) {
  const start = performance.now()
  const end = start + duration

  function update() {
    const current = performance.now()
    const progress = (current - start) / duration
    const currentIndex = Math.floor(progress * targetString.length)

    if (current < end) {
      let decodingString =
        targetString.substring(0, currentIndex) +
        getRandomString(targetString.length - currentIndex)
      el.textContent = decodingString
      requestAnimationFrame(update)
    } else {
      el.textContent = targetString
    }
  }

  requestAnimationFrame(update)
}

export function getRandomString(length: number) {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  let result = ""
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length))
  }
  return result
}
