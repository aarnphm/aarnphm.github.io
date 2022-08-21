import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"
// @ts-ignore
import script from "./scripts/comments.inline"
import commentsStyle from "./styles/comments.scss"

type Options = {
  provider: "giscus"
  options: {
    repo: `${string}/${string}`
    repoId: string
    category: string
    categoryId: string
    mapping?: "url" | "title" | "og:title" | "specific" | "number" | "pathname"
    strict?: boolean
    reactionsEnabled?: boolean
    inputPosition?: "top" | "bottom"
    lang?: string
  }
}

function boolToStringBool(b: boolean): string {
  return b ? "1" : "0"
}

export default ((opts: Options) => {
  const Comments: QuartzComponent = ({
    ctx,
    displayClass,
    cfg,
    fileData,
  }: QuartzComponentProps) => {
    const enableComments = fileData.frontmatter?.comments ?? true
    const isPost = fileData.slug!.startsWith("posts/")
    const isUniversity = fileData.slug!.includes("university")
    if (!enableComments || isPost || isUniversity || ctx.argv.serve) {
      return <></>
    }
    return (
      <div
        class={classNames(displayClass, "giscus")}
        data-repo={opts.options.repo}
        data-repo-id={opts.options.repoId}
        data-category={opts.options.category}
        data-category-id={opts.options.categoryId}
        data-mapping={opts.options.mapping ?? "url"}
        data-strict={boolToStringBool(opts.options.strict ?? true)}
        data-reactions-enabled={boolToStringBool(opts.options.reactionsEnabled ?? true)}
        data-input-position={opts.options.inputPosition ?? "bottom"}
        data-theme={`https://${cfg.baseUrl ?? "example.com"}/index.css`}
        data-lang={opts.options.lang ?? cfg.locale.split("-")[0]}
      ></div>
    )
  }

  Comments.afterDOMLoaded = script
  Comments.css = commentsStyle

  return Comments
}) satisfies QuartzComponentConstructor<Options>
