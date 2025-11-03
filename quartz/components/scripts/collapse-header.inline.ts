function handleToggleClick(event: Event) {
  const toggle = event.currentTarget as HTMLElement | null
  if (!toggle) return

  const anchor = (event.target as HTMLElement | null)?.closest('a[data-role="anchor"]')
  if (anchor) return

  event.stopPropagation()

  const section = toggle.closest<HTMLElement>("section.collapsible-header")
  if (!section) return

  const shell = section.querySelector<HTMLElement>("[data-collapse-shell]")
  if (!shell) return

  shell.classList.toggle("is-open")
  const isOpen = shell.classList.contains("is-open")

  toggle.setAttribute("aria-expanded", isOpen ? "true" : "false")

  localStorage.setItem(
    `${window.document.body.dataset.slug!.replace(/\//g, "--")}-${toggle.id}`,
    isOpen ? "true" : "false",
  )
}

function handleToggleKeydown(event: KeyboardEvent) {
  const key = event.key
  if (key !== "Enter" && key !== " " && key !== "Spacebar") return

  const toggle = event.currentTarget as HTMLElement | null
  if (!toggle) return

  event.preventDefault()
  toggle.click()
}

function hydrateCollapsibleHeaders() {
  document.querySelectorAll<HTMLElement>("section.collapsible-header").forEach((section) => {
    const shell = section.querySelector<HTMLElement>("[data-collapse-shell]")
    if (!shell) return

    const toggle = section.querySelector<HTMLElement>("[data-collapse-toggle]")
    if (!toggle) return

    const initialOpen = shell.dataset.initialOpen !== "false"
    const stored = localStorage.getItem(
      `${window.document.body.dataset.slug!.replace(/\//g, "--")}-${toggle.id}`,
    )
    const isOpen = stored ? stored === "true" : initialOpen

    if (isOpen) {
      shell.classList.add("is-open")
    } else {
      shell.classList.remove("is-open")
    }
    toggle.setAttribute("aria-expanded", isOpen ? "true" : "false")

    if (toggle.tabIndex === -1) {
      toggle.tabIndex = 0
    }

    toggle.addEventListener("click", handleToggleClick)
    toggle.addEventListener("keydown", handleToggleKeydown)
    window.addCleanup?.(() => {
      toggle.removeEventListener("click", handleToggleClick)
      toggle.removeEventListener("keydown", handleToggleKeydown)
    })
  })

  const transcludeButtons = document.querySelectorAll("button.transclude-title-link")
  for (const button of transcludeButtons) {
    const parent = button.parentElement as HTMLElement | null
    if (!parent || !parent.dataset.href) continue

    const href = parent.dataset.href
    const navigate = () => window.spaNavigate(new URL(href, window.location.toString()))

    button.addEventListener("click", navigate)
    window.addCleanup?.(() => button.removeEventListener("click", navigate))
  }
}

document.addEventListener("nav", hydrateCollapsibleHeaders)
window.addEventListener("resize", hydrateCollapsibleHeaders)
