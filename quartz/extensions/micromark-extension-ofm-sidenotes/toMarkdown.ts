import type { Options } from "mdast-util-to-markdown"
import type { Sidenote, SidenoteReference, SidenoteDefinition } from "./types"

export function sidenoteToMarkdown(): Options {
  return {
    handlers: {
      sidenote: handleSidenote as any,
      sidenoteReference: handleReference as any,
      sidenoteDefinition: handleDefinition as any,
    } as any,
    unsafe: [
      { character: "{", inConstruct: ["phrasing"] },
      { character: "}", inConstruct: ["phrasing"] },
    ],
  }
}

function handleSidenote(node: Sidenote, _: any, state: any, info: any): string {
  const parsed = node.data?.sidenoteParsed

  if (!parsed) {
    return node.value || ""
  }

  let output = "{{sidenotes"

  if (parsed.properties && Object.keys(parsed.properties).length > 0) {
    const props = Object.entries(parsed.properties)
      .map(([k, v]) => {
        if (Array.isArray(v)) {
          return `${k}: ${v.join(", ")}`
        }
        return `${k}: ${v}`
      })
      .join(", ")
    output += `<${props}>`
  }

  if (parsed.label) {
    output += `[${parsed.label}]`
  }

  output += ": "

  const exit = state.enter("sidenote")
  output += state.containerPhrasing(node, info)
  exit()

  output += "}}"

  return output
}

function handleReference(node: SidenoteReference): string {
  return `{{sidenotes[^${node.label}]}}`
}

function handleDefinition(
  node: SidenoteDefinition,
  _: any,
  state: any,
  info: any,
): string {
  const exit = state.enter("sidenoteDefinition")
  const value = state.containerFlow(node, info)
  exit()

  return `{{sidenotes[${node.label}]}}:\n${value}`
}
