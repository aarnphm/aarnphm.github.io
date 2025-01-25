import { isBrowser } from "./util"

let mermaidPromise: Promise<any> | null = null

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

if (isBrowser()) {
  if (!mermaidPromise) {
    mermaidPromise ||= await import(
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
      const mermaid = mod.default

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

      window.mermaid = mermaid
      return mermaid
    })
  }
}
