function hydrateStreamInteractions() {
  const el = document.querySelector<HTMLElement>(".stream")
  if (!el) return

  const timeElements = el.querySelectorAll<HTMLTimeElement>(".stream-entry-date[datetime]")

  if (timeElements.length === 0) return

  const formatter = new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: false,
    timeZoneName: "shortOffset",
  })

  timeElements.forEach((timeEl) => {
    const isoDate = timeEl.getAttribute("datetime")
    if (!isoDate) return

    const date = new Date(isoDate)
    if (Number.isNaN(date.getTime())) return

    timeEl.textContent = formatter.format(date)
  })

  const entries = Array.from(el.querySelectorAll<HTMLElement>(".stream-entry"))
  if (entries.length === 0) return

  const interactiveLinks = Array.from(
    el.querySelectorAll<HTMLAnchorElement>(
      ".stream-entry-date[data-stream-link][data-stream-timestamp]",
    ),
  )

  if (interactiveLinks.length === 0) return

  const timestampHrefMap = new Map<string, string>()
  interactiveLinks.forEach((link) => {
    const timestamp = link.dataset.streamTimestamp
    const href = link.dataset.streamHref
    if (timestamp && href) {
      timestampHrefMap.set(timestamp, href)
    }
  })

  if (timestampHrefMap.size <= 1) {
    el.removeAttribute("data-stream-active-timestamp")
    return
  }

  const canonicalPath = el.dataset.streamCanonical || "/stream"
  const originalUrl = new URL(window.location.href)
  const originalSearch = originalUrl.search
  const originalHash = originalUrl.hash

  let activeTimestamp: string | null = null

  const applyHistory = (targetPath: string | null) => {
    const url = new URL(window.location.href)
    url.pathname = targetPath ?? canonicalPath
    url.search = originalSearch
    url.hash = originalHash
    window.history.replaceState(window.history.state, "", url)
  }

  const updateEntries = (
    targetTimestamp: string | null,
    opts: { updateHistory?: boolean } = {},
  ) => {
    const { updateHistory = true } = opts
    activeTimestamp = targetTimestamp

    if (!targetTimestamp) {
      el.removeAttribute("data-stream-active-timestamp")
    } else {
      el.dataset.streamActiveTimestamp = targetTimestamp
    }

    entries.forEach((entry) => {
      const entryTimestamp = entry.dataset.streamTimestamp ?? null
      const matches = targetTimestamp !== null && entryTimestamp === targetTimestamp

      if (!targetTimestamp || matches) {
        entry.hidden = false
      } else {
        entry.hidden = true
      }

      entry.classList.toggle("stream-entry-active", matches)
    })

    interactiveLinks.forEach((link) => {
      const matches = targetTimestamp !== null && link.dataset.streamTimestamp === targetTimestamp
      link.classList.toggle("is-active", matches)
      if (matches) {
        link.setAttribute("aria-current", "page")
      } else {
        link.removeAttribute("aria-current")
      }
    })

    if (!updateHistory) return

    if (targetTimestamp) {
      const targetPath = timestampHrefMap.get(targetTimestamp)
      if (targetPath) {
        applyHistory(targetPath)
      }
    } else if (window.location.pathname !== canonicalPath) {
      applyHistory(null)
    }
  }

  const clearIfOutside = () => {
    window.setTimeout(() => {
      const activeElement = document.activeElement as HTMLElement | null
      if (!activeElement || !el.contains(activeElement)) {
        if (activeTimestamp !== null) {
          updateEntries(null)
        }
      }
    }, 0)
  }

  const handleRootKeydown = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      updateEntries(null)
    }
  }

  el.addEventListener("keydown", handleRootKeydown)
  el.addEventListener("focusout", clearIfOutside, true)
  window.addCleanup(() => el.removeEventListener("keydown", handleRootKeydown))
  window.addCleanup(() => el.removeEventListener("focusout", clearIfOutside, true))

  const getTimestampForPath = (path: string): string | null => {
    for (const [timestamp, href] of timestampHrefMap.entries()) {
      if (href === path) {
        return timestamp
      }
    }
    return null
  }

  interactiveLinks.forEach((link) => {
    const timestamp = link.dataset.streamTimestamp
    if (!timestamp) return

    const onClick = (event: MouseEvent) => {
      event.preventDefault()
      updateEntries(activeTimestamp === timestamp ? null : timestamp)
      link.focus()
    }

    const onKeydown = (event: KeyboardEvent) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault()
        updateEntries(activeTimestamp === timestamp ? null : timestamp)
      } else if (event.key === "Escape" && activeTimestamp !== null) {
        event.preventDefault()
        updateEntries(null)
      }
    }

    link.addEventListener("click", onClick)
    link.addEventListener("keydown", onKeydown)

    window.addCleanup(() => link.removeEventListener("click", onClick))
    window.addCleanup(() => link.removeEventListener("keydown", onKeydown))
  })

  const handlePopstate = () => {
    const timestamp = getTimestampForPath(window.location.pathname)
    updateEntries(timestamp, { updateHistory: false })
  }

  window.addEventListener("popstate", handlePopstate)
  window.addCleanup(() => window.removeEventListener("popstate", handlePopstate))

  const initialTimestamp = getTimestampForPath(window.location.pathname)
  if (initialTimestamp) {
    updateEntries(initialTimestamp, { updateHistory: false })
  } else if (window.location.pathname !== canonicalPath) {
    updateEntries(null, { updateHistory: true })
  } else {
    updateEntries(null, { updateHistory: false })
  }
}

document.addEventListener("nav", hydrateStreamInteractions)
