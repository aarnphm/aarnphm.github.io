import { renderDiagrams } from "./util"

document.addEventListener("DOMContentLoaded", function () {
  if (window.__MERMAID_RENDERED__) return
  window.__MERMAID_RENDERED__ = true

  const nodes = document.querySelectorAll<HTMLDivElement>(
    "pre > code.mermaid:not([data-processed])",
  )
  if (nodes.length == 0) return

  let timeoutId: NodeJS.Timeout | undefined
  const checkMermaid = () => {
    if (timeoutId) clearTimeout(timeoutId)
    timeoutId = setTimeout(() => {
      if (window.mermaid) {
        window.mermaid.run({ nodes }).then(async () => await renderDiagrams(nodes))
      } else {
        checkMermaid()
      }
    }, 300)
  }
  checkMermaid()
})
