document.addEventListener("nav", () => {
  const root = document.documentElement
  const lang = document.body.dataset.language
  const computedStyle = getComputedStyle(root)
  const defaultBodyFont = computedStyle.getPropertyValue("--bodyFont")
  const defaultCodeFont = computedStyle.getPropertyValue("--codeFont")
  const defaultHeaderFont = computedStyle.getPropertyValue("--headerFont")
  switch (lang) {
    case "vi":
      root.style.setProperty("--headerFont", `"Playfair Display", ${defaultHeaderFont}`)
      root.style.setProperty("--bodyFont", `"Playfair Display", ${defaultBodyFont}`)
      root.style.setProperty("--codeFont", `"Playfair Display", ${defaultCodeFont}`)
      break
    default:
      root.style.removeProperty("--headerFont")
      root.style.removeProperty("--bodyFont")
      root.style.removeProperty("--codeFont")
      break
  }
})
