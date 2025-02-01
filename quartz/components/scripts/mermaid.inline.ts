import { mermaidViewer } from "./util"

const cssVars = [
  "--secondary",
  "--tertiary",
  "--gray",
  "--light",
  "--lightgray",
  "--highlight",
  "--dark",
  "--darkgray",
  "--codeFont",
] as const

let mermaidImport = undefined
document.addEventListener("nav", async () => {
  const nodes = document.querySelectorAll<HTMLDivElement>("pre > code.mermaid")
  if (nodes.length === 0) return

  mermaidImport ||= await import(
    //@ts-ignore
    "https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.4.0/mermaid.esm.min.mjs"
  ).then((mod) => {
    const computedStyleMap = cssVars.reduce(
      (acc, key) => {
        acc[key] = getComputedStyle(document.documentElement).getPropertyValue(key)
        return acc
      },
      {} as Record<(typeof cssVars)[number], string>,
    )

    // The actual mermaid instance is the default export:
    const mermaid: typeof import("mermaid/dist/mermaid").default = mod.default

    const darkMode = document.documentElement.getAttribute("saved-theme") === "dark"
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: darkMode ? "dark" : "base",
      themeVariables: {
        fontFamily: computedStyleMap["--codeFont"],
        primaryColor: computedStyleMap["--light"],
        primaryTextColor: computedStyleMap["--darkgray"],
        primaryBorderColor: computedStyleMap["--tertiary"],
        lineColor: computedStyleMap["--darkgray"],
        secondaryColor: computedStyleMap["--secondary"],
        tertiaryColor: computedStyleMap["--tertiary"],
        clusterBkg: computedStyleMap["--light"],
        edgeLabelBackground: computedStyleMap["--highlight"],
      },
    })
    mermaid.run({ nodes }).then(() => {
      mermaidViewer(nodes)
    })
    window.mermaid = mermaid
    return
  })
})
