import { QuartzComponent, QuartzComponentConstructor } from "./types"
import { htmlToJsx } from "../util/jsx"
import { Root } from "hast"
// @ts-ignore
import script from "./scripts/mermaid.inline"
import style from "./styles/mermaid.scss"

const Mermaid: QuartzComponent = ({ fileData }) => {
  return htmlToJsx(fileData.filePath!, {
    type: "root",
    children: [
      {
        type: "element",
        tagName: "div",
        properties: { id: "mermaid-container" },
        children: [
          {
            type: "element",
            tagName: "div",
            properties: { className: ["mermaid-backdrop"] },
            children: [],
          },
          {
            type: "element",
            tagName: "div",
            properties: { id: "mermaid-space" },
            children: [
              {
                type: "element",
                tagName: "div",
                properties: { className: ["mermaid-header"] },
                children: [
                  {
                    type: "element",
                    tagName: "button",
                    properties: {
                      className: ["close-button"],
                      "aria-label": "close button",
                      title: "close button",
                    },
                    children: [
                      {
                        type: "element",
                        tagName: "svg",
                        properties: {
                          "aria-hidden": "true",
                          xmlns: "http://www.w3.org/2000/svg",
                          width: 24,
                          height: 24,
                          viewBox: "0 0 24 24",
                          fill: "none",
                          stroke: "currentColor",
                          "stroke-width": "2",
                          "stroke-linecap": "round",
                          "stroke-linejoin": "round",
                        },
                        children: [
                          {
                            type: "element",
                            tagName: "line",
                            properties: {
                              x1: 18,
                              y1: 6,
                              x2: 6,
                              y2: 18,
                            },
                            children: [],
                          },
                          {
                            type: "element",
                            tagName: "line",
                            properties: {
                              x1: 6,
                              y1: 6,
                              x2: 18,
                              y2: 18,
                            },
                            children: [],
                          },
                        ],
                      },
                    ],
                  },
                ],
              },
              {
                type: "element",
                tagName: "div",
                properties: { className: ["mermaid-content"] },
                children: [],
              },
            ],
          },
        ],
      },
    ],
  } as Root)
}

Mermaid.css = style
Mermaid.afterDOMLoaded = script

export default (() => Mermaid) satisfies QuartzComponentConstructor
