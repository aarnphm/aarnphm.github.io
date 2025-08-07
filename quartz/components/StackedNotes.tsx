import { QuartzComponent, QuartzComponentConstructor } from "./types"
import { h } from "preact"
// @ts-ignore
import script from "./scripts/matuschak.inline"
import style from "./styles/matuschak.scss"
import { svgOptions } from "./svg"

export default (() => {
  const StackedNotes: QuartzComponent = () => {
    return (
      <div class="stacked-buttons">
        <span
          id="stacked-note-toggle"
          title="Toggle stacked notes"
          aria-label="Toggle stacked notes"
          role="switch"
          // @ts-ignore
          type="button"
          aria-checked="false"
        >
          <div class="view-toggle-slide" />
          <div class="view-toggle-switch">
            <svg
              class="single-view-icon"
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <circle cx="12" cy="12" r="6" />
            </svg>
            <svg
              class="stacked-view-icon"
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <circle cx="14" cy="12" r="6" opacity="1" />
              <circle cx="10" cy="12" r="6" opacity="0.5" />
            </svg>
          </div>
        </span>
        {h(
          "span",
          {
            type: "button",
            class: "stacked-copy",
            tabindex: -1,
            "aria-hidden": "true",
            "aria-label": "copy stacked notes URL",
            title: "copy stacked notes URL",
          },
          h("svg", { ...svgOptions, viewbox: "0 -8 24 24", width: 16, height: 16 }, [
            h("use", { href: "#github-copy" }),
          ]),
        )}
      </div>
    )
  }

  StackedNotes.css = style
  StackedNotes.afterDOMLoaded = script

  return StackedNotes
}) satisfies QuartzComponentConstructor
