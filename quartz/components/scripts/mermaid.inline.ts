import { removeAllChildren } from "./util"

const svgExpand =
  '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 1.06L2.56 7h10.88l-2.22-2.22a.75.75 0 011.06-1.06l3.5 3.5a.75.75 0 010 1.06l-3.5 3.5a.75.75 0 11-1.06-1.06l2.22-2.22H2.56l2.22 2.22a.75.75 0 11-1.06 1.06l-3.5-3.5a.75.75 0 010-1.06l3.5-3.5z"></path></svg>'

for (let i = 0; i < nodes.length; i++) {
  const codeBlock = nodes[i] as HTMLElement
  const pre = codeBlock.parentElement as HTMLPreElement
  const clipboardBtn = pre.querySelector(".clipboard-button") as HTMLButtonElement

  const expandBtn = document.createElement("button")
  expandBtn.className = "expand-button"
  expandBtn.type = "button"
  expandBtn.innerHTML = svgExpand
  expandBtn.ariaLabel = "Expand mermaid diagram"
  const clipboardStyle = window.getComputedStyle(clipboardBtn)
  const clipboardWidth =
    clipboardBtn.offsetWidth +
    parseFloat(clipboardStyle.marginLeft || "0") +
    parseFloat(clipboardStyle.marginRight || "0")

  // Set expand button position
  expandBtn.style.right = `calc(${clipboardWidth}px + 0.3rem)`
  pre.prepend(expandBtn)

  // Create popup container
  const popupContainer = document.createElement("div")
  popupContainer.id = "mermaid-container"
  popupContainer.innerHTML = `
  <div id="mermaid-space">
    <div class="mermaid-header">
      <button class="close-button" aria-label="Close">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
    <div class="mermaid-content"></div>
  </div>
  `
  pre.appendChild(popupContainer)

  function showMermaid() {
    const content = popupContainer.querySelector(".mermaid-content") as HTMLElement
    if (!content) return
    removeAllChildren(content)

    // Clone the mermaid content
    const mermaidContent = codeBlock.querySelector("svg")!.cloneNode(true)
    content.appendChild(mermaidContent)

    // Show container
    popupContainer.classList.add("active")
  }

  function hideMermaid() {
    popupContainer.classList.remove("active")
  }

  function handleEscape(e: any) {
    if (e.key === "Escape") {
      hideMermaid()
    }
  }

  const closeBtn = popupContainer.querySelector(".close-button") as HTMLButtonElement

  closeBtn.addEventListener("click", hideMermaid)
  expandBtn.addEventListener("click", showMermaid)
  document.addEventListener("keydown", handleEscape)

  window.addCleanup(() => {
    closeBtn.removeEventListener("click", hideMermaid)
    expandBtn.removeEventListener("click", showMermaid)
    document.removeEventListener("keydown", handleEscape)
  })
}
