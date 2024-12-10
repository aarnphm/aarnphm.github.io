document.addEventListener("nav", () => {
  document.querySelectorAll("div#telescope").forEach((el) => {
    el.querySelectorAll('span[class~="details"]').forEach((closed) => {
      function onClick() {
        closed.classList.remove("close")
        closed.classList.add("open")
      }
      closed.addEventListener("click", onClick)
      window.addCleanup(() => closed.removeEventListener("click", onClick))
    })
  })
})
