import { QuartzEmitterPlugin } from "../types"
import { QuartzComponent, QuartzComponentProps } from "../../components/types"
import { Navigation, DesktopOnly, Spacer } from "../../components"
import BodyConstructor from "../../components/Body"
import { write } from "./helpers"
import { FullPageLayout } from "../../cfg"
import { FilePath, pathToRoot } from "../../util/path"
import { pageResources, renderPage } from "../../components/renderPage"
import DepGraph from "../../depgraph"
import { sharedPageComponents } from "../../../quartz.layout"
import { classNames } from "../../util/lang"
import { i18n } from "../../i18n"
//@ts-ignore
import curiusScript from "../../components/scripts/curius.inline"
//@ts-ignore
import curiusFriendScript from "../../components/scripts/curius-friends.inline"

const CuriusContent: QuartzComponent = (props: QuartzComponentProps) => {
  const { displayClass } = props
  const Footer = Navigation({ prev: "/mechinterp", next: "/books" })
  return (
    <>
      <CuriusHeader {...props} />
      <div class={classNames(displayClass, "curius", "popover-hint")} id="curius">
        <div class="curius-page-container">
          <div id="curius-fetching-text"></div>
          <div id="curius-fragments"></div>
          <div class="highlight-modal" id="highlight-modal">
            <ul id="highlight-modal-list"></ul>
          </div>
        </div>
      </div>
      <Footer {...props} />
    </>
  )
}
CuriusContent.afterDOMLoaded = curiusScript

const CuriusFriends: QuartzComponent = (props: QuartzComponentProps) => {
  const { displayClass } = props
  return (
    <div class={classNames(displayClass, "curius-friends")}>
      <h4 style={["font-size: initial", "margin-top: unset", "margin-bottom: 0.5rem"].join(";")}>
        mes amis.
      </h4>
      <ul class="overflow section-ul" id="friends-list" style="margin-top: unset"></ul>
      <div id="see-more-friends">
        Void{" "}
        <span id="more" style="text-decoration: none !important">
          de plus
        </span>
        <svg
          fill="currentColor"
          preserveAspectRatio="xMidYMid meet"
          height="1rem"
          width="1rem"
          viewBox="0 -10 40 40"
        >
          <g>
            <path d="m31 12.5l1.5 1.6-12.5 13.4-12.5-13.4 1.5-1.6 11 11.7z"></path>
          </g>
        </svg>
      </div>
    </div>
  )
}
CuriusFriends.afterDOMLoaded = curiusFriendScript

const CuriusHeader: QuartzComponent = ({ cfg, displayClass }: QuartzComponentProps) => {
  const searchPlaceholder = i18n(cfg.locale).components.search.searchBarPlaceholder
  return (
    <div class={classNames(displayClass, "curius-header")}>
      <div class="curius-search">
        <input
          id="curius-bar"
          type="text"
          aria-label={searchPlaceholder}
          placeholder={searchPlaceholder}
        />
        <div id="curius-search-container"></div>
      </div>
      <div class="curius-title">
        <em>
          Voir de plus{" "}
          <a href="https://curius.app/aaron-pham" target="_blank">
            curius.app/aaron-pham
          </a>
        </em>
        <svg
          id="curius-refetch"
          aria-labelledby="refetch"
          data-tooltip="refresh"
          data-id="refetch"
          height="12"
          type="button"
          viewBox="0 0 24 24"
          width="12"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M17.65 6.35c-1.63-1.63-3.94-2.57-6.48-2.31-3.67.37-6.69 3.35-7.1 7.02C3.52 15.91 7.27 20 12 20c3.19 0 5.93-1.87 7.21-4.56.32-.67-.16-1.44-.9-1.44-.37 0-.72.2-.88.53-1.13 2.43-3.84 3.97-6.8 3.31-2.22-.49-4.01-2.3-4.48-4.52C5.31 9.44 8.26 6 12 6c1.66 0 3.14.69 4.22 1.78l-1.51 1.51c-.63.63-.19 1.71.7 1.71H19c.55 0 1-.45 1-1V6.41c0-.89-1.08-1.34-1.71-.71z"></path>
        </svg>
      </div>
    </div>
  )
}

const CuriusTrail: QuartzComponent = (props: QuartzComponentProps) => {
  const { cfg, displayClass } = props
  return (
    <div class={classNames(displayClass, "curius-trail")} data-limits={3} data-locale={cfg.locale}>
      <ul class="section-ul" id="trail-list"></ul>
    </div>
  )
}

export const CuriusPage: QuartzEmitterPlugin = () => {
  const opts: FullPageLayout = {
    ...sharedPageComponents,
    beforeBody: [],
    left: [CuriusFriends],
    right: [DesktopOnly(CuriusTrail)],
    pageBody: CuriusContent,
    afterBody: [],
    header: [],
    footer: Spacer(),
  }

  const { head, header, beforeBody, pageBody, left, right, afterBody, footer: Footer } = opts
  const Body = BodyConstructor()

  return {
    name: "CuriusPage",
    getQuartzComponents() {
      return [
        head,
        ...header,
        Body,
        ...beforeBody,
        pageBody,
        ...afterBody,
        ...left,
        ...right,
        Footer,
      ]
    },
    async getDependencyGraph(_ctx, _content, _resources) {
      return new DepGraph<FilePath>()
    },
    async emit(ctx, content, resources): Promise<FilePath[]> {
      const cfg = ctx.cfg.configuration
      const fps: Promise<FilePath>[] = []

      for (const [tree, file] of content) {
        const slug = file.data.slug!
        if (slug === "curius") {
          const externalResources = pageResources(pathToRoot(slug), resources)
          const componentData: QuartzComponentProps = {
            ctx,
            fileData: file.data,
            externalResources,
            cfg,
            children: [],
            tree,
            allFiles: [],
          }
          const fp = write({
            ctx,
            content: renderPage(cfg, slug, componentData, opts, externalResources),
            slug: slug,
            ext: ".html",
          })
          fps.push(fp)
          break
        }
      }
      return await Promise.all(fps)
    },
  }
}
