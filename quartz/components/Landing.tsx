import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import MetaConstructor from "./Meta"

import style from "./styles/landing.scss"
import { byDateAndAlphabetical } from "./PageList"
import { i18n } from "../i18n"
import { FullSlug, SimpleSlug, resolveRelative } from "../util/path"
import { Data } from "vfile"
import { getDate, formatDate } from "./Date"
import DarkmodeConstructor from "./Darkmode"
import SpacerConstructor from "./Spacer"
import KeybindConstructor from "./Keybind"
import HeaderConstructor from "./Header"
import { classNames } from "../util/lang"

export const HyperAlias = {
  books: "/books",
  mailbox: "/posts/",
  projects: "/thoughts/projects",
  uses: "/uses",
  advices: "/quotes",
  affecter: "/influence",
  scents: "/thoughts/Scents",
}
export const SocialAlias = {
  github: "https://github.com/aarnphm",
  twitter: "https://x.com/aarnphm_",
  curius: "/curius",
  contact: "mailto:contact@aarnphm.xyz",
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

const notesLimit = 12

const NotesConstructor = (() => {
  const Spacer = SpacerConstructor()

  function Notes({ allFiles, fileData, displayClass, cfg }: QuartzComponentProps) {
    const pages = allFiles
      .filter((f: Data) => {
        return (
          !["university", "tags", "index", "influence", "uses", "curius", "music", "quotes"].some(
            (it) => (f.slug as FullSlug).includes(it),
          ) && !f.frontmatter?.noindex
        )
      })
      .sort(byDateAndAlphabetical(cfg))
    const remaining = Math.max(0, pages.length - notesLimit)
    const classes = ["min-links", "internal"].join(" ")
    return (
      <>
        <h2>récentes:</h2>
        <div class="notes-container">
          <div>
            <ul class="landing-notes">
              {pages.slice(0, notesLimit).map((page) => {
                const title = page.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title

                return (
                  <li>
                    <a href={resolveRelative(fileData.slug!, page.slug!)} class={classes}>
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
                    href={resolveRelative(fileData.slug!, "thoughts/" as SimpleSlug)}
                    class={classes}
                  >
                    {i18n(cfg.locale).components.recentNotes.seeRemainingMore({ remaining })}
                  </a>
                </u>
              </p>
            )}
          </div>
          <Spacer {...({ allFiles, fileData, displayClass, cfg } as QuartzComponentProps)} />
        </div>
      </>
    )
  }
  return Notes
}) satisfies QuartzComponentConstructor

const ContentConstructor = (() => {
  const Header = HeaderConstructor()
  const Notes = NotesConstructor()
  const Darkmode = DarkmodeConstructor()
  const Keybind = KeybindConstructor()

  function Content(componentData: QuartzComponentProps) {
    return (
      <div class="content-container">
        <Header {...componentData}>
          <h1>My name is Aaron.</h1>
          <Keybind {...componentData} />
          <Darkmode {...componentData} />
        </Header>
        <p>
          Beige and <span class="rose">rosé</span> are my two favorite colours.{" "}
          <a href={"/thoughts/Chaos"} target="_self" class="internal landing-links">
            Chaos
          </a>{" "}
          constructs the id and form the ego. I enjoy treating my friends with{" "}
          <a href={"/thoughts/Dishes"} target="_self" class="internal landing-links">
            cooking
          </a>
          . I spend a lot of time{" "}
          <a href={"/thoughts/writing"} target="_self" class="internal landing-links">
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
          <h2>jardin:</h2>
          <div class="clickable-container">
            {Object.entries(HyperAlias).map(([name, url]) => (
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
          <hr />
          <h2>sociaux:</h2>
          <div class="clickable-container">
            {Object.entries(SocialAlias).map(([name, url], index, array) => (
              <>
                <AliasLink key={name} name={name} url={url} newTab={name !== "curius"} />
                {index === array.length - 1 ? "" : <span>{"  "}</span>}
              </>
            ))}
          </div>
        </div>
      </div>
    )
  }
  return Content
}) satisfies QuartzComponentConstructor

export default (() => {
  const Meta = MetaConstructor({ enableDarkMode: false })
  const Content = ContentConstructor()
  const Spacer = SpacerConstructor()

  function LandingComponent(componentData: QuartzComponentProps) {
    return (
      <div class="popover-hint">
        <div class={classNames(componentData.displayClass, "landing")}>
          <Meta {...componentData} />
          <Content {...componentData} />
        </div>
      </div>
    )
  }

  LandingComponent.css = style

  return LandingComponent
}) satisfies QuartzComponentConstructor
