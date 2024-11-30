const onClickGenerator = (button: HTMLButtonElement, source: string) => {
  return function onClick() {
    navigator.clipboard.writeText(source).then(
      () => {
        button.blur()
        button.classList.add("check")
        setTimeout(() => {
          button.style.borderColor = ""
          button.classList.remove("check")
        }, 2000)
      },
      (error) => console.error(error),
    )
  }
}

const renderer = {
  indentSize: "0.6em",
  commentDelimiter: "  ▷",
  lineNumberPunc: ":",
  lineNumber: true,
  noEnd: false,
  scopeLines: false,
  captionCount: undefined,
  titlePrefix: "Algorithm",
  mathEngine: "katex",
  mathRenderer: undefined,
}

document.addEventListener("nav", () => {
  const els = document.getElementsByClassName("tikz") as HTMLCollectionOf<HTMLElement>
  if (els.length == 0) return
  for (let i = 0; i < els.length; i++) {
    const tikzBlock = els[i]

    const button = tikzBlock.querySelector(
      "figcaption > button.source-code-button",
    ) as HTMLButtonElement
    if (!button) continue

    const mathML = tikzBlock.querySelector(".tikz-mathml")
    if (!mathML) continue
    const source = JSON.parse(mathML.querySelector("annotation")!.textContent ?? "")

    const onClick = onClickGenerator(button, source)
    button.addEventListener("click", onClick)
    window.addCleanup(() => button.removeEventListener("click", onClick))
  }
})

document.addEventListener("nav", () => {
  const els = document.getElementsByClassName("ps-root") as HTMLCollectionOf<HTMLElement>
  if (els.length == 0) return

  for (let i = 0; i < els.length; i++) {
    const psBlock = els[i]
    const button = psBlock.getElementsByClassName("ps-clipboard")[0] as HTMLButtonElement
    let inlineMacros: string | undefined = undefined
    if (psBlock.dataset.inlineMacros && psBlock.dataset.inlineMacros !== "") {
      inlineMacros = JSON.parse(psBlock.dataset.inlineMacros as string)
    }
    const mathML = psBlock.querySelector(".ps-mathml")
    if (!mathML) continue

    const blockContent = JSON.parse(mathML.querySelector("annotation")!.textContent ?? "")
    const source =
      "\\documentclass{article}\n" +
      macros(inlineMacros) +
      "\n" +
      "\\begin{document}\n" +
      processing(blockContent) +
      "\n\\end{document}"

    const onClick = onClickGenerator(button, source)
    button.addEventListener("click", onClick)
    window.addCleanup(() => button.removeEventListener("click", onClick))
  }
})

const macros = (inlineMacros: string | undefined): string => {
  // Split inline macros into lines and remove heading or trailing spaces
  const inlineMacrosLine =
    inlineMacros !== undefined ? inlineMacros.split("\n").map((line) => line.trim()) : ""

  return `
\\usepackage{algorithm}
\\usepackage[noEnd=${renderer.noEnd},indLines=${renderer.noEnd}]{algpseudocodex}

\\newcommand{\\And}{\\textbf{and~}}
\\newcommand{\\Or}{\\textbf{or~}}
\\newcommand{\\Xor}{\\textbf{xor~}}
\\newcommand{\\Not}{\\textbf{not~}}
\\newcommand{\\To}{\\textbf{to~}}
\\newcommand{\\DownTo}{\\textbf{downto~}}
\\newcommand{\\True}{\\textbf{true~}}
\\newcommand{\\False}{\\textbf{false~}}
\\newcommand{\\Input}{\\item[\\textbf{Input:}]}
\\renewcommand{\\Output}{\\item[\\textbf{Output:}]}
\\newcommand{\\Print}{\\State \\textbf{print~}}
\\renewcommand{\\Return}{\\State \\textbf{return~}}

\\usepackage{amsmath}
${inlineMacrosLine}
`
}

const processing = (block: string): string => {
  if (renderer.lineNumber)
    // Replace "\begin{algorithmic}" with "\begin{algorithmic}[1]"
    block = block.replace("\\begin{algorithmic}", "\\begin{algorithmic}[1]")
  else;
  return block
}
