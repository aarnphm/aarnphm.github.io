document.addEventListener("nav", async () => {
  const popover = document.getElementById("content-popover")
  if (!popover) return
  popover.dataset.show = true.toString()
})
