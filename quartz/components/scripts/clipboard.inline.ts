document.addEventListener("nav", () => {
  const els = document.getElementsByTagName("pre")
  for (let i = 0; i < els.length; i++) {
    const codeBlock = els[i].getElementsByTagName("code")[0]
    const button = els[i].querySelector("span.clipboard-button")
    if (codeBlock) {
      const source = (
        codeBlock.dataset.clipboard ? codeBlock.dataset.clipboard : codeBlock.innerText
      ).replace(/\n\n/g, "\n")
      function onClick() {
        navigator.clipboard.writeText(source).then(
          () => {
            button?.classList.add("check")
            setTimeout(() => {
              button?.classList.remove("check")
            }, 2000)
          },
          (error) => console.error(error),
        )
      }
      button?.addEventListener("click", onClick)
      window.addCleanup(() => button?.removeEventListener("click", onClick))
    }
  }
})
