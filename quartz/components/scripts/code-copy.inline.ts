document.addEventListener("nav", () => {
  const codeElements = document.querySelectorAll<HTMLElement>("code:not(pre > code):not(.no-copy)")

  const copyToClipboard = async (text: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch (err) {
      console.error("Failed to copy:", err)
      return false
    }
  }

  const showCopyToast = () => {
    const event: CustomEventMap["toast"] = new CustomEvent("toast", {
      detail: { message: "copied to clipboard", containerId: "code-copy-toast-container" },
    })
    document.dispatchEvent(event)
  }

  codeElements.forEach((code) => {
    // Add visual indication that code is clickable
    code.style.cursor = "pointer"
    code.title = "Click to copy"

    const onClick = async (e: MouseEvent) => {
      e.preventDefault()
      const text = code.textContent || ""
      const success = await copyToClipboard(text)
      if (success) {
        showCopyToast()
      }
    }

    code.addEventListener("click", onClick)
    window.addCleanup(() => code.removeEventListener("click", onClick))
  })
})
