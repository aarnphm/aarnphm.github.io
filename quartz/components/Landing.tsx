import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import MetaComponent from "./Meta"
import style from "./styles/landing.scss"
import { byDateAndAlphabetical } from "./PageList"
import { i18n } from "../i18n"
import { FullSlug, SimpleSlug, resolveRelative } from "../util/path"
import { Data } from "vfile"
import { getDate, formatDate } from "./Date"
import SpacerComponent from "./Spacer"
import HeaderComponent from "./Header"
import ContentComponent from "./pages/Content"
import BodyComponent from "./Body"
import { classNames } from "../util/lang"
import { JSX } from "preact"
import { GlobalConfiguration } from "../cfg"
import { QuartzPluginData } from "../plugins/vfile"

export const HyperAlias = {
  livres: "/books",
  "boîte aux lettres": "/posts/",
  projets: "/projects",
  uses: "/uses",
  advices: "/quotes",
  affecter: "/influence",
  parfum: "/thoughts/Scents",
  tunes: "/music",
  "atelier with friends": "/thoughts/atelier-with-friends",
}

export const SocialAlias = {
  github: "https://github.com/aarnphm",
  twitter: "https://x.com/aarnphm_",
  substack: "https://livingalonealone.com",
  curius: "/curius",
  contact: "mailto:contact@aarnphm.xyz",
  "site source": "https://github.com/aarnphm/sites",
}

type AliasLinkProp = {
  name?: string
  url?: string
  isInternal?: boolean
  newTab?: boolean | ((name: string) => boolean)
  enablePopover?: boolean
}

const AliasLink = (props: AliasLinkProp) => {
  const opts = { isInternal: false, newTab: false, enablePopover: true, ...props }
  const className = ["landing-links"]
  if (opts.isInternal && opts.enablePopover) className.push("internal")
  return (
    <a
      href={opts.url}
      target={opts.newTab ? "_blank" : "_self"}
      rel="noopener noreferrer"
      className={className.join(" ")}
    >
      {opts.name}
    </a>
  )
}

function byModifiedAndAlphabetical(
  cfg: GlobalConfiguration,
): (f1: QuartzPluginData, f2: QuartzPluginData) => number {
  return (f1, f2) => {
    if (f1.dates && f2.dates) {
      // sort descending
      return f2.dates?.modified.getTime() - f1.dates?.modified.getTime()
    } else if (f1.dates && !f2.dates) {
      // prioritize files with dates
      return -1
    } else if (!f1.dates && f2.dates) {
      return 1
    }

    // otherwise, sort lexographically by title
    const f1Title = f1.frontmatter?.title.toLowerCase() ?? ""
    const f2Title = f2.frontmatter?.title.toLowerCase() ?? ""
    return f1Title.localeCompare(f2Title)
  }
}

const NotesComponent = ((opts?: {
  slug: SimpleSlug
  numLimits?: number
  header?: string
  sortBy?: "modified" | "alphabetical"
}) => {
  const Spacer = SpacerComponent()

  const sortCaller = opts?.sortBy === "modified" ? byModifiedAndAlphabetical : byDateAndAlphabetical

  const Notes: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { allFiles, fileData, cfg } = componentData
    const pages = allFiles
      .filter((f: Data) => {
        if (f.slug!.startsWith(opts!.slug)) {
          return (
            !["university", "tags", "index", ...cfg.ignorePatterns].some((it) =>
              (f.slug as FullSlug).includes(it),
            ) &&
            !f.frontmatter?.noindex &&
            !f.frontmatter?.construction
          )
        }
        return false
      })
      .sort(sortCaller(cfg))

    const remaining = Math.max(0, pages.length - opts!.numLimits!)
    const classes = ["min-links", "internal"].join(" ")
    return (
      <div id="note-item">
        <h2>{opts!.header}.</h2>
        <div class="notes-container">
          <div class="recent-links">
            <ul class="landing-notes">
              {pages.slice(0, opts!.numLimits).map((page) => {
                const title = page.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title

                return (
                  <li>
                    <a href={resolveRelative(fileData.slug!, page.slug!)} class={classes}>
                      <div class="landing-meta">
                        <span class="landing-mspan">
                          {formatDate(getDate(cfg, page)!, cfg.locale)}
                        </span>
                        <span class="landing-mtitle">{title}</span>
                      </div>
                    </a>
                  </li>
                )
              })}
            </ul>
            {remaining > 0 && (
              <p>
                <em>
                  <a href={resolveRelative(fileData.slug!, opts!.slug)} class={classes}>
                    {i18n(cfg.locale).components.recentNotes.seeRemainingMore({
                      remaining,
                    })}
                  </a>
                </em>
              </p>
            )}
          </div>
          <Spacer {...componentData} />
        </div>
      </div>
    )
  }
  return Notes
}) satisfies QuartzComponentConstructor

const ClickableContainer = (props: {
  title: string
  links: Record<string, string>
  cfg: AliasLinkProp
}) => {
  const { title, links, cfg } = props
  let newTab: boolean | undefined

  return (
    <>
      <h2>{title}:</h2>
      <div class="clickable-container">
        {Object.entries(links).map(([name, url]) => {
          if (typeof cfg.newTab === "function") {
            newTab = cfg.newTab(name)
          } else {
            newTab = cfg.newTab
          }
          return <AliasLink key={name} {...cfg} name={name} url={url} newTab={newTab} />
        })}
      </div>
    </>
  )
}

const HyperlinksComponent = ((props?: { children: JSX.Element[] }) => {
  const { children } = props ?? { children: [] }

  const Hyperlink: QuartzComponent = () => <div class="hyperlinks">{children}</div>
  return Hyperlink
}) satisfies QuartzComponentConstructor

const ElementComponent = (() => {
  const Content = ContentComponent()
  const RecentNotes = NotesComponent({
    header: "récentes",
    slug: "thoughts/" as SimpleSlug,
    numLimits: 6,
    sortBy: "modified",
  })
  const RecentPosts = NotesComponent({
    header: "écriture",
    slug: "posts/" as SimpleSlug,
    numLimits: 6,
    sortBy: "alphabetical",
  })
  const Hyperlink = HyperlinksComponent({
    children: [
      ClickableContainer({
        title: "jardin",
        links: HyperAlias,
        cfg: { isInternal: true, newTab: false },
      }),
      ClickableContainer({
        title: "média",
        links: SocialAlias,
        cfg: { isInternal: false, newTab: (name) => name !== "curius" },
      }),
    ],
  })

  const Element: QuartzComponent = (componentData: QuartzComponentProps) => {
    return (
      <div class="content-container">
        <Content {...componentData} />
        <div class="notes-outer">
          <RecentNotes {...componentData} />
          <RecentPosts {...componentData} />
        </div>
        <Hyperlink {...componentData} />
      </div>
    )
  }

  return Element
}) satisfies QuartzComponentConstructor

export default (() => {
  const Meta = MetaComponent()
  const Element = ElementComponent()
  const Body = BodyComponent()
  const Header = HeaderComponent()

  const LandingComponent: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { displayClass } = componentData
    return (
      <div id="quartz-root" class="page">
        {/* @ts-ignore */}
        <Body {...componentData}>
          <div class="center">
            <div class="page-header">
              <Header {...componentData}>
                <h1 class="article-title" style="margin-top: 0">
                  Bonjour, je suis Aaron.
                </h1>
                <Meta {...componentData} />
              </Header>
            </div>
            <div class={classNames(displayClass, "landing")}>
              <Element {...componentData} />
            </div>
          </div>
        </Body>
      </div>
    )
  }

  LandingComponent.css = style

  return LandingComponent
}) satisfies QuartzComponentConstructor
