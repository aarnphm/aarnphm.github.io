import { pathToRoot } from "../util/path"
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"

export default (() => {
  const PageTitle: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
    const baseDir = pathToRoot(fileData.slug!)
    return (
      <a class="page-title" href={baseDir} aria-label="home" title="Return home">
        <img src="/static/icon.webp" alt="profile" />
      </a>
    )
  }

  PageTitle.css = `
.page-title {
  transform: scale(0.8);
}

.page-title img {
  border-radius: var(--radius-full);
  display: inline-block;
  height: 1.5rem;
  width: 1.5rem;
  vertical-align: middle;
  margin: 0;
}
`

  return PageTitle
}) satisfies QuartzComponentConstructor
