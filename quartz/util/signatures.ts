import { h, s } from "hastscript"

interface SvgOptions {
  viewBox: string
  width: string
  height: string
}

interface SvgBankElement {
  d: string
  opts: SvgOptions
}

interface LetterBank {
  [key: string]: {
    upper: SvgBankElement
    lower: SvgBankElement
  }
}

interface GlyphBank {
  [key: string]: SvgBankElement & { name: string }
}

const glyphScript = (char: string, idx: number, glyphs: GlyphBank) => {
  const path = glyphs[char]
  if (!path) return h("span", { class: "char" }, char)

  return h("span", { class: `glyph ${path.name}` }, [
    s(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        fill: "none",
        ...path.opts,
      },
      [h("path", { d: path.d, style: `animation: drawPath 1s ease forwards ${idx * 0.1}s;` })],
    ),
  ])
}

const charScript = (char: string, idx: number, paths: LetterBank, glyphs: GlyphBank) => {
  if (char === " ") {
    return h("span", { class: "space", style: "margin-right: 0.6em;" })
  }

  // Handle glyphs first
  if (glyphs[char]) {
    return glyphScript(char, idx, glyphs)
  }

  const isUpper = char === char.toUpperCase()
  const key = char.toLowerCase()

  if (!paths[key]) {
    return h("span", { class: "char" }, char)
  }

  const path = isUpper ? paths[key].upper : paths[key].lower

  return h("span", { class: `${key} ${isUpper ? "up" : "lo"}` }, [
    s(
      "svg",
      {
        xmlns: "http://www.w3.org/2000/svg",
        fill: "none",
        ...path.opts,
      },
      s("path", { d: path.d, style: `animation: drawPath 1s ease forwards ${idx * 0.1}s;` }),
    ),
  ])
}

export default (paths: LetterBank, glyphs: GlyphBank) => (text: string) =>
  text.split("").map((char, index) => charScript(char, index, paths, glyphs))
