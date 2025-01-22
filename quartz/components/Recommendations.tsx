import { LCG } from "../util/helpers"
import { classNames } from "../util/lang"
import { FilePath, resolveRelative, slugifyFilePath } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

interface Options {
  topChoices: number
}

const defaultOptions: Options = {
  topChoices: 7,
}

export default ((userOpts?: Options) => {
  const opts = { ...defaultOptions, ...userOpts }

  const Recommendations: QuartzComponent = ({
    fileData,
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    let p = fileData.slug as string
    if (fileData.filePath) p = fileData.filePath
    const seed =
      slugifyFilePath(p as FilePath)
        .split("")
        .reduce((acc, char) => acc + char.charCodeAt(0), 0) ?? 0
    const rng = new LCG(seed)

    const distributions = allFiles.filter(
      (f) => f.slug !== fileData.slug && !f.slug!.includes("university"),
    )
    const recs = rng.shuffle(distributions).slice(0, opts.topChoices)

    return (
      <section data-recs={true} class={classNames(displayClass, "recommendations", "main-col")}>
        <h2 id="label" lang="fr">
          Vous pourriez aimer ce qui suit
        </h2>
        <menu class="overflow">
          {recs.map((file) => (
            <li>
              <a
                href={resolveRelative(fileData.slug!, file.slug!)}
                data-recommendation={file.slug!}
              >
                {file.frontmatter?.title}
              </a>
            </li>
          ))}
        </menu>
      </section>
    )
  }

  Recommendations.css = `
.recommendations {
  margin-bottom: 1.9rem;

  & > .overflow {
    padding-inline-start: 12px;
    list-style: square;
    margin-block: 0;
  }
}`

  return Recommendations
}) satisfies QuartzComponentConstructor
