import { pathToRoot } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const PageTitle: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const baseDir = pathToRoot(fileData.slug!)
  return (
    <a class="page-title" href={baseDir} aria-label="home" title="Return home">
      <img src="/static/icon.webp" alt="profile pic" />
    </a>
  )
}

PageTitle.css = `
.page-title {
  transform: scale(0.8);
}

.page-title img {
  border-radius: 999px;
  display: inline-block;
  height: 1.5rem;
  width: 1.5rem;
  vertical-align: middle;
  margin: 0;
}
`

export default (() => PageTitle) satisfies QuartzComponentConstructor
