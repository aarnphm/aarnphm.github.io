import { renderDiagrams } from "./util"

document.addEventListener("nav", async () => {
  const mainContent = document.querySelector("article") as HTMLElement
  if (!mainContent) return

  const nodes = mainContent.querySelectorAll<HTMLDivElement>("pre > code.mermaid")
  if (nodes.length === 0 || !window.mermaid) return

  await window.mermaid.run({ nodes }).then(async () => await renderDiagrams(nodes))
})
