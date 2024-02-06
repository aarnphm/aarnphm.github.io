import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import MetaConstructor from "./Meta"

import landingStyle from "./styles/landing.scss"
//@ts-ignore
import landingScript from "./scripts/landing.inline"
//@ts-ignore
import darkModeScript from "./scripts/darkmode.inline"
//@ts-ignore
import keybindScript from "./scripts/keybind.inline"
import { byDateAndAlphabetical } from "./PageList"
import { i18n } from "../i18n"
import { FullSlug, SimpleSlug, resolveRelative } from "../util/path"
import { Data } from "vfile"
import { getDate, formatDate } from "./Date"

export const HyperAlias = {
  books: "/books",
  mailbox: "/posts/",
  projects: "/dump/projects",
  uses: "/uses",
  advices: "/dump/quotes",
  affecter: "/influence",
  scents: "/dump/Scents",
}
export const SocialAlias = {
  github: "https://github.com/aarnphm",
  twitter: "https://x.com/aarnphm_",
  curius: "/curius",
  contact: "mailto:contact@aarnphm.xyz",
}
export const KeybindAlias = {
  "cmd+k": "search",
  "cmd+g": "graph",
  "cmd+o": "toggle dark mode",
  "cmd+\\": "homepage",
}

type AliasLinkProp = {
  name: string
  url: string
  isInternal?: boolean
  newTab?: boolean
  enablePopover?: boolean
}

const AliasLink = (props: AliasLinkProp) => {
  const opts = { isInternal: false, newTab: false, enablePopover: true, ...props }
  const className = ["landing-links"]
  if (opts.isInternal && opts.enablePopover) className.push("internal")
  return (
    <a href={opts.url} target={opts.newTab ? "_blank" : "_self"} className={className.join(" ")}>
      {opts.name}
    </a>
  )
}

const Limits = 12

const NotesConstructor = (() => {
  function Notes({ allFiles, fileData, cfg }: QuartzComponentProps) {
    const pages = allFiles
      .filter((f: Data) => {
        return (
          !["university", "tags", "index", "influence", "uses", "curius", "music", "quotes"].some(
            (it) => (f.slug as FullSlug).includes(it),
          ) && !f.frontmatter?.noindex
        )
      })
      .sort(byDateAndAlphabetical(cfg))
    const remaining = Math.max(0, pages.length - Limits)
    return (
      <>
        <h2>récentes:</h2>
        <div class="notes-container">
          <div>
            <ul class="landing-notes">
              {pages.slice(0, Limits).map((page) => {
                const title = page.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title
                const date = page.dates?.modified

                return (
                  <li>
                    <a href={resolveRelative(fileData.slug!, page.slug!)} class="min-links">
                      <div class="landing-meta">
                        <span class="landing-mspan">
                          {formatDate(getDate(cfg, page)!, cfg.locale)}
                        </span>
                        <u>{title}</u>
                      </div>
                    </a>
                  </li>
                )
              })}
            </ul>
            {remaining > 0 && (
              <p>
                <u>
                  <a
                    href={resolveRelative(fileData.slug!, "dump/" as SimpleSlug)}
                    class="min-links"
                  >
                    {i18n(cfg.locale).components.recentNotes.seeRemainingMore({ remaining })}
                  </a>
                </u>
              </p>
            )}
          </div>
          <div class="spacer"></div>
        </div>
      </>
    )
  }
  return Notes
}) satisfies QuartzComponentConstructor

const ContentConstructor = (() => {
  const Notes = NotesConstructor()

  function Content(componentData: QuartzComponentProps) {
    return (
      <div class="content-container">
        <h1>My name is Aaron.</h1>
        <p>
          Beige and <span class="rose">rosé</span> are my two favorite colours.{" "}
          <a href={"/dump/Chaos"} target="_self" class="internal landing-links">
            Chaos
          </a>{" "}
          constructs the id and form the ego. I enjoy treating my friends with{" "}
          <a href={"/dump/Dishes"} target="_self" class="internal landing-links">
            cooking
          </a>
          . I spend a lot of time{" "}
          <a href={"/dump/writing"} target="_self" class="internal landing-links">
            writing
          </a>{" "}
          and{" "}
          <a href={"/books"} target="_self" class="internal landing-links">
            reading
          </a>{" "}
          when I'm not coding. I'm pretty bullish on investing into self and fullfil one's desire in
          life.
        </p>
        <p>
          Currently, I'm building{" "}
          <a
            href="https://bentoml.com"
            target="_blank"
            rel="noopener noreferrer"
            class="landing-links"
          >
            serving infrastructure
          </a>{" "}
          and explore our interaction with large language models.
        </p>
        <hr />
        <Notes {...componentData} />
        <hr />
        <div class="hyperlinks">
          <h2>garden:</h2>
          <div id="garden">
            {Object.entries(HyperAlias).map(([name, url], index, array) => (
              <>
                <AliasLink
                  key={name}
                  name={name}
                  url={url}
                  isInternal
                  enablePopover={name !== "tunes"}
                />
              </>
            ))}
          </div>
          <h2>socials:</h2>
          <div id="socials">
            {Object.entries(SocialAlias).map(([name, url], index, array) => (
              <>
                <AliasLink key={name} name={name} url={url} newTab={name !== "curius"} />
                {index === array.length - 1 ? "" : <span>{"  "}</span>}
              </>
            ))}
          </div>
        </div>
        <hr />
        <ul class="keybinds">
          {Object.entries(KeybindAlias).map(([key, value], index, array) => (
            <li>
              <a id="landing-keybind" data-keybind={key.replaceAll("+", "--")}>
                {key}
              </a>
              : {value}
            </li>
          ))}
        </ul>
      </div>
    )
  }
  return Content
}) satisfies QuartzComponentConstructor

export default (() => {
  const Meta = MetaConstructor()
  const Content = ContentConstructor()
  function LandingComponent(componentData: QuartzComponentProps) {
    return (
      <div class="popover-hint">
        <div class="landing">
          <Meta {...componentData} />
          <Content {...componentData} />
        </div>
      </div>
    )
  }
  LandingComponent.css = landingStyle
  LandingComponent.beforeDOMLoaded = darkModeScript
  LandingComponent.afterDOMLoaded = landingScript + keybindScript

  return LandingComponent
}) satisfies QuartzComponentConstructor
