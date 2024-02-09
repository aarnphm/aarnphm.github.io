import { QuartzComponentConstructor, QuartzComponentProps } from "./types"
import MetaConstructor from "./Meta"

import style from "./styles/landing.scss"
import { byDateAndAlphabetical } from "./PageList"
import { i18n } from "../i18n"
import { FullSlug, SimpleSlug, _stripSlashes, resolveRelative } from "../util/path"
import { Data } from "vfile"
import { getDate, formatDate } from "./Date"
import DarkmodeConstructor from "./Darkmode"
import SpacerConstructor from "./Spacer"
import KeybindConstructor from "./Keybind"
import HeaderConstructor from "./Header"
import { classNames } from "../util/lang"
import { GlobalConfiguration } from "../cfg"

export const HyperAlias = {
  livres: "/books",
  "boîte aux lettres": "/posts/",
  projets: "/projects",
  uses: "/uses",
  advices: "/quotes",
  affecter: "/influence",
  parfum: "/thoughts/Scents",
}

export const ContentAlias = {
  Chaos: "/thoughts/Chaos",
  cooking: "/thoughts/Dishes",
  writing: "/thoughts/writing",
  reading: "/books",
  "open-source projects": "/projects",
  agency: "/thoughts/Agency",
  desire: "/thoughts/desire",
  hypertext: "/thoughts/Hypertext",
  "large language models": "/thoughts/LLMs",
  "digital garden": "/thoughts/Digital-garden",
}

const combined = (...objects: any[]) => objects.flatMap(Object.values).map((a) => _stripSlashes(a))

export const LandingLinks = combined(HyperAlias, ContentAlias)

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

type ContentType = keyof typeof ContentAlias

const getContentAlias = (name: string) => {
  return <AliasLink name={name} url={ContentAlias[name as ContentType] ?? "/"} isInternal />
}

const notesLimit = 20

const NotesConstructor = (() => {
  const Spacer = SpacerConstructor()

  function Notes({ allFiles, fileData, displayClass, cfg }: QuartzComponentProps) {
    const pages = allFiles
      .filter((f: Data) => {
        return (
          ![
            "university",
            "tags",
            "index",
            "influence",
            "uses",
            "curius",
            "music",
            "quotes",
            ...cfg.ignorePatterns,
          ].some((it) => (f.slug as FullSlug).includes(it)) &&
          !f.frontmatter?.noindex &&
          !f.frontmatter?.construction
        )
      })
      .sort(byDateAndAlphabetical(cfg))
    const remaining = Math.max(0, allFiles.length - notesLimit)
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
                    {i18n(cfg.locale).components.recentNotes.seeRemainingMore({
                      remaining,
                    })}
                  </a>
                </u>
              </p>
            )}
          </div>
          <Spacer {...({ allFiles, fileData, displayClass, cfg: cfg } as QuartzComponentProps)} />
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
          <h1 lang={"en"}>My name is Aaron.</h1>
          <Keybind {...componentData} />
          <Darkmode {...componentData} />
        </Header>
        <p lang="en">
          Beige and <span class="rose">rosé</span> are my two favorite colours.{" "}
          {getContentAlias("Chaos")} constructs the id and form the ego. I enjoy treating my friends
          with {getContentAlias("cooking")}. I spend a lot of time {getContentAlias("writing")}
          {", "}
          {getContentAlias("reading")}
          {", "} and maintaining {getContentAlias("open-source projects")}. I'm pretty bullish on
          high {getContentAlias("agency")} and fullfil one's {getContentAlias("desire")} in life.
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
          and explore our interaction with {getContentAlias("large language models")}.
        </p>
        <p>
          You are currently at the <em>index</em> of my {getContentAlias("hypertext")}{" "}
          {getContentAlias("digital garden")}. As far as a "about" page goes, feel free to explore
          around. Please don't hesitate to reach out if you have any questions or just want to chat.
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
          <h2>média:</h2>
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
      <div id="quartz-root" class="page">
        <div id="quartz-body">
          <div class="center">
            <article class="popover-hint">
              <div class={classNames(componentData.displayClass, "landing")}>
                <Meta {...componentData} />
                <Content {...componentData} />
              </div>
            </article>
          </div>
        </div>
      </div>
    )
  }

  LandingComponent.css = style

  return LandingComponent
}) satisfies QuartzComponentConstructor
