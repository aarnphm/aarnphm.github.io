import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { classNames } from '../util/lang'
import style from './styles/license.scss'

function frontmatterString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

function isStringRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export default (() => {
  const License: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const frontmatter = isStringRecord(fileData.frontmatter) ? fileData.frontmatter : undefined
    const license = frontmatterString(frontmatter?.license)
    if (!license) return null

    const licenseUrl =
      frontmatterString(frontmatter?.license_url) ?? frontmatterString(frontmatter?.licenseUrl)

    return (
      <section data-license class={classNames(displayClass, 'license-meta', 'main-col')}>
        <h2 id="license-label">license</h2>
        <p>
          {licenseUrl ? (
            <a href={licenseUrl} target="_blank" rel="noopener noreferrer">
              {license}
            </a>
          ) : (
            <span>{license}</span>
          )}
        </p>
      </section>
    )
  }

  License.css = style
  return License
}) satisfies QuartzComponentConstructor
