import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/minimal.scss"
import { i18n } from "../i18n"
import { version } from "../../package.json"
import { classNames } from "../util/lang"
import type { JSX } from "preact"
import { Date as DateComponent, getDate } from "./Date"

interface Options {
  links: Record<string, string>
}

const iconMapping: Record<string, JSX.Element> = {
  github: (
    <svg
      viewBox="64 64 896 896"
      focusable="false"
      data-icon="github"
      width="1em"
      height="1em"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M511.6 76.3C264.3 76.2 64 276.4 64 523.5 64 718.9 189.3 885 363.8 946c23.5 5.9 19.9-10.8 19.9-22.2v-77.5c-135.7 15.9-141.2-73.9-150.3-88.9C215 726 171.5 718 184.5 703c30.9-15.9 62.4 4 98.9 57.9 26.4 39.1 77.9 32.5 104 26 5.7-23.5 17.9-44.5 34.7-60.8-140.6-25.2-199.2-111-199.2-213 0-49.5 16.3-95 48.3-131.7-20.4-60.5 1.9-112.3 4.9-120 58.1-5.2 118.5 41.6 123.2 45.3 33-8.9 70.7-13.6 112.9-13.6 42.4 0 80.2 4.9 113.5 13.9 11.3-8.6 67.3-48.8 121.3-43.9 2.9 7.7 24.7 58.3 5.5 118 32.4 36.8 48.9 82.7 48.9 132.3 0 102.2-59 188.1-200 212.9a127.5 127.5 0 0138.1 91v112.5c.8 9 0 17.9 15 17.9 177.1-59.7 304.6-227 304.6-424.1 0-247.2-200.4-447.3-447.5-447.3z"></path>
    </svg>
  ),
  twitter: (
    <svg
      viewBox="64 64 896 896"
      focusable="false"
      data-icon="twitter"
      width="1em"
      height="1em"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M928 254.3c-30.6 13.2-63.9 22.7-98.2 26.4a170.1 170.1 0 0075-94 336.64 336.64 0 01-108.2 41.2A170.1 170.1 0 00672 174c-94.5 0-170.5 76.6-170.5 170.6 0 13.2 1.6 26.4 4.2 39.1-141.5-7.4-267.7-75-351.6-178.5a169.32 169.32 0 00-23.2 86.1c0 59.2 30.1 111.4 76 142.1a172 172 0 01-77.1-21.7v2.1c0 82.9 58.6 151.6 136.7 167.4a180.6 180.6 0 01-44.9 5.8c-11.1 0-21.6-1.1-32.2-2.6C211 652 273.9 701.1 348.8 702.7c-58.6 45.9-132 72.9-211.7 72.9-14.3 0-27.5-.5-41.2-2.1C171.5 822 261.2 850 357.8 850 671.4 850 843 590.2 843 364.7c0-7.4 0-14.8-.5-22.2 33.2-24.3 62.3-54.4 85.5-88.2z"></path>
    </svg>
  ),
}

const getIcon = (name: string) => iconMapping[name] ?? <span>{name}</span>

export default ((opts?: Options) => {
  const Footer: QuartzComponent = ({ displayClass, cfg, fileData }: QuartzComponentProps) => {
    const year = new Date().getFullYear()
    const links = opts?.links ?? []

    if (fileData.frontmatter?.poem) {
      return (
        <footer class={classNames(displayClass, "poetry-footer")}>
          <DateComponent date={getDate(cfg, fileData)!} locale={cfg.locale} />
        </footer>
      )
    }

    return (
      <>
        <footer class={classNames(displayClass, "minimal-footer")}>
          <div id="icons">
            <ul>
              <li>
                <a href={"/"} target="_self">
                  <span role="img" class="min-icon" aria-label="home">
                    <svg
                      viewBox="64 64 896 896"
                      focusable="false"
                      data-icon="home"
                      width="1em"
                      height="1em"
                      fill="currentColor"
                      aria-hidden="true"
                    >
                      <path d="M946.5 505L560.1 118.8l-25.9-25.9a31.5 31.5 0 00-44.4 0L77.5 505a63.9 63.9 0 00-18.8 46c.4 35.2 29.7 63.3 64.9 63.3h42.5V940h691.8V614.3h43.4c17.1 0 33.2-6.7 45.3-18.8a63.6 63.6 0 0018.7-45.3c0-17-6.7-33.1-18.8-45.2zM568 868H456V664h112v204zm217.9-325.7V868H632V640c0-22.1-17.9-40-40-40H432c-22.1 0-40 17.9-40 40v228H238.1V542.3h-96l370-369.7 23.1 23.1L882 542.3h-96.1z"></path>
                    </svg>
                  </span>
                </a>
              </li>
              {Object.entries(links).map(([text, link]) => {
                const label = text.toLowerCase()
                return (
                  <li>
                    <a href={link} target="_blank">
                      <span role="img" class="min-icon" aria-label={label}>
                        {getIcon(label)}
                      </span>
                    </a>
                  </li>
                )
              })}
            </ul>
          </div>
          <p class="info">
            {i18n(cfg.locale).components.footer.createdWith}{" "}
            <a href="https://quartz.jzhao.xyz/" target="_blank" rel="noopener noreferrer">
              Quartz v{version}
            </a>{" "}
            © {year}
          </p>
        </footer>
      </>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
