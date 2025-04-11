document.addEventListener("nav", () => {
  const noteTags = document.querySelectorAll(".note-tag") as NodeListOf<HTMLElement>
  const notesList = document.querySelector(".section-ul") as HTMLUListElement
  let activeTag: HTMLElement | null = null
  let hiddenNotes: HTMLElement[] = []

  function filterByTag(tag: HTMLElement) {
    const tagValue = tag.dataset.tag
    if (!tagValue) return

    // Store and remove non-matching notes
    const notes = [...notesList.children] as HTMLElement[]
    notes.forEach((li) => {
      const link = li.querySelector(".note-link") as HTMLAnchorElement
      const tags = link?.dataset.tags?.split(",") ?? []
      if (!tags.includes(tagValue)) {
        hiddenNotes.push(li)
        notesList.removeChild(li)
      }
    })

    noteTags.forEach((otherTag) => {
      if (otherTag !== tag) {
        otherTag.classList.add("fade-out")
      }
    })

    activeTag = tag
    tag.classList.add("active")
  }

  function resetFilter() {
    if (hiddenNotes.length > 0) {
      // Get current visible notes
      const visibleNotes = [...notesList.children] as HTMLElement[]

      // Remove all notes
      while (notesList.firstChild) {
        notesList.removeChild(notesList.firstChild)
      }

      // Combine and sort all notes by their original order
      const allNotes = [...visibleNotes, ...hiddenNotes].sort((a, b) => {
        const aIndex = parseInt(a.dataset.index ?? "0")
        const bIndex = parseInt(b.dataset.index ?? "0")
        return aIndex - bIndex
      })

      // Add all notes back
      allNotes.forEach((note) => notesList.appendChild(note))
      hiddenNotes = []
    }

    noteTags.forEach((tag) => {
      tag.classList.remove("fade-out", "active")
    })
    activeTag = null
  }

  noteTags.forEach((tag) => {
    const tagValue = tag.dataset.tag

    const onMouseEnter = () => {
      if (tagValue && !activeTag) {
        removeFadeOut()
        fadeOutOtherTags(tag)
        fadeOutNonMatchingNotes(tagValue)
      }
    }

    const onMouseLeave = () => {
      if (!activeTag) {
        removeFadeOut()
      }
    }

    const onClick = () => {
      if (activeTag === tag) {
        resetFilter()
      } else {
        resetFilter()
        filterByTag(tag)
      }
    }

    tag.addEventListener("mouseenter", onMouseEnter)
    tag.addEventListener("mouseleave", onMouseLeave)
    tag.addEventListener("click", onClick)
    tag.style.cursor = "pointer"

    window.addCleanup(() => {
      tag.removeEventListener("mouseenter", onMouseEnter)
      tag.removeEventListener("mouseleave", onMouseLeave)
      tag.removeEventListener("click", onClick)
    })
  })

  function fadeOutOtherTags(currentTag: HTMLElement) {
    noteTags.forEach((tag) => {
      if (tag !== currentTag && tag !== activeTag) {
        tag.classList.add("fade-out")
      }
    })
  }

  function fadeOutNonMatchingNotes(tag: string) {
    const notes = [...notesList.children] as HTMLElement[]

    notes.forEach((li) => {
      const link = li.querySelector(".note-link") as HTMLAnchorElement
      const tags = link?.dataset.tags?.split(",") ?? []
      if (!tags.includes(tag)) {
        li.classList.add("fade-out")
      }
    })
  }

  function removeFadeOut() {
    noteTags.forEach((tag) => {
      if (tag !== activeTag) {
        tag.classList.remove("fade-out")
      }
    })
    const notes = [...notesList.children] as HTMLElement[]
    notes.forEach((li) => {
      li.classList.remove("fade-out")
    })
  }

  // Setup sidepanel layout toggle
  const setupSidepanelLayoutToggle = () => {
    const quartzRoot = document.getElementById("quartz-root")
    const sidepanel = document.querySelector(".sidepanel-container")
    const slugData = document.body.getAttribute("data-slug")

    // Skip for index page since it has special layout
    if (slugData === "index" || !quartzRoot || !sidepanel) return

    // Save original styles to restore them later
    const originalStyle = {
      display: quartzRoot.style.display,
      flexDirection: quartzRoot.style.flexDirection,
      minHeight: quartzRoot.style.minHeight,
    }

    // Create a mutation observer to watch for class changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === "attributes" && mutation.attributeName === "class") {
          if (sidepanel.classList.contains("active")) {
            // When sidepanel is active, switch to grid layout
            quartzRoot.classList.add("grid")
            quartzRoot.style.display = ""
            quartzRoot.style.flexDirection = ""
            quartzRoot.style.minHeight = ""
          } else {
            // When sidepanel is inactive, restore flex layout
            quartzRoot.classList.remove("grid")
            quartzRoot.style.display = originalStyle.display
            quartzRoot.style.flexDirection = originalStyle.flexDirection
            quartzRoot.style.minHeight = originalStyle.minHeight
          }
        }
      })
    })

    // Start observing the sidepanel for class changes
    observer.observe(sidepanel, { attributes: true })

    // Add cleanup for the observer
    window.addCleanup(() => observer.disconnect())
  }

  // Run the setup
  setupSidepanelLayoutToggle()
})
