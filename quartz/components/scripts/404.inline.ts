document.addEventListener("DOMContentLoaded", () => {
  const slug = document.body.dataset.slug
  if (slug === "404") {
    const tooltip = document.querySelector(".home-tooltip") as HTMLElement

    const handleMouseMove = (e: MouseEvent) => {
      const x = e.clientX
      const y = e.clientY
      // Offset tooltip slightly from cursor
      tooltip!.style.left = `${x}px`
      tooltip!.style.top = `${y + 30}px`
    }

    const handleClick = (e: MouseEvent) => {
      e.preventDefault()
      window.spaNavigate(new URL("/", window.location.toString()))
    }

    document.body.addEventListener("click", handleClick)
    document.body.addEventListener("mousemove", handleMouseMove)

    // Show/hide tooltip on mouse enter/leave
    document.body.addEventListener("mouseenter", () => {
      tooltip!.classList.add("visible")
    })
    document.body.addEventListener("mouseleave", () => {
      tooltip!.classList.remove("visible")
    })

    // Cleanup function
    window.addCleanup(() => {
      document.body.removeEventListener("click", handleClick)
      document.body.removeEventListener("mousemove", handleMouseMove)
    })
  }
})
