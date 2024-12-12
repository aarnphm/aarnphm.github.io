import { QuartzComponent, QuartzComponentConstructor } from "./types"
import { htmlToJsx } from "../util/jsx"
import { Root } from "hast"
// @ts-ignore
import script from "./scripts/mermaid.inline"
import style from "./styles/mermaid.scss"
import { h, s } from "hastscript"

const Mermaid: QuartzComponent = ({ fileData }) => {
  return htmlToJsx(fileData.filePath!, {
    type: "root",
    children: [
      h(
        "#mermaid-container",
        h(".mermaid-backdrop"),
        h(
          "#mermaid-space",
          h(
            ".mermaid-header",
            h(
              "button.close-button",
              { arialabel: "close button", title: "close button", type: "button" },
              [
                s(
                  "svg",
                  {
                    ariaHidden: true,
                    xmlns: "http://www.w3.org/2000/svg",
                    width: 24,
                    height: 24,
                    viewbox: "0 0 24 24",
                    fill: "none",
                    stroke: "currentColor",
                    strokewidth: 2,
                    strokelinecap: "round",
                    strokelinejoin: "round",
                  },
                  [
                    s("line", { x1: 18, y1: 6, x2: 6, y2: 18 }),
                    s("line", { x1: 6, y1: 6, x2: 18, y2: 18 }),
                  ],
                ),
              ],
            ),
          ),
          h(".mermaid-content"),
        ),
      ),
    ],
  } as Root)
}

Mermaid.css = style
Mermaid.afterDOMLoaded = script

export default (() => Mermaid) satisfies QuartzComponentConstructor
