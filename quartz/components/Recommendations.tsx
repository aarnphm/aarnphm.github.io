import { LCG } from "../util/helpers"
import { classNames } from "../util/lang"
import { resolveRelative, slugifyFilePath } from "../util/path"
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
    const seed =
      slugifyFilePath(fileData.filePath!)
        .split("")
        .reduce((acc, char) => acc + char.charCodeAt(0), 0) ?? 0
    const rng = new LCG(seed)

    const distributions = allFiles.filter((f) => f.slug !== fileData.slug)
    const recs = rng.shuffle(distributions).slice(0, opts.topChoices)

    return (
      <section data-recs={true} class={classNames(displayClass, "recommendations")}>
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
  grid-area: grid-center-one;

  & > .overflow {
    padding-inline-start: 1rem;
  }
}`

  return Recommendations
}) satisfies QuartzComponentConstructor
