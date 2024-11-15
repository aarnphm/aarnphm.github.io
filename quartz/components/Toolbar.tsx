import { QuartzComponent, QuartzComponentConstructor } from "./types"
// @ts-ignore
import script from "./scripts/toolbar.inline.ts"
import style from "./styles/toolbar.scss"

export default (() => {
  const Toolbar: QuartzComponent = () => {
    return (
      <div class="toolbar">
        <div class="toolbar-content">
          <button class="toolbar-item pen-button" aria-label="Toggle toolbar" aria-expanded="false">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="1em"
              height="1em"
              fill="var(--darkgray)"
              viewBox="0 0 256 256"
              class="pen-icon"
            >
              <path d="M227.31,73.37,182.63,28.68a16,16,0,0,0-22.63,0L36.69,152A15.86,15.86,0,0,0,32,163.31V208a16,16,0,0,0,16,16H92.69A15.86,15.86,0,0,0,104,219.31L227.31,96a16,16,0,0,0,0-22.63ZM92.69,208H48V163.31l88-88L180.69,120ZM192,108.68,147.31,64l24-24L216,84.68Z"></path>
            </svg>
          </button>
          <button
            class="toolbar-item"
            id="reader-button"
            aria-label="Toggle reader mode"
            data-active="false"
          >
            <span class="tooltip">Reader mode</span>
            <svg
              class="reader-icon"
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
            </svg>
          </button>
          <button class="toolbar-item" id="pdf-button" aria-label="Export to PDF">
            <span class="tooltip">Export PDF</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <line x1="10" y1="9" x2="8" y2="9" />
            </svg>
          </button>
        </div>
      </div>
    )
  }

  Toolbar.css = style
  Toolbar.afterDOMLoaded = script

  return Toolbar
}) satisfies QuartzComponentConstructor
