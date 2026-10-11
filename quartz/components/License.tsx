import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { CC_ICON_PATHS, CC_ICON_VIEWBOX, type CcModule } from '../util/cc-icons'
import { classNames } from '../util/lang'
import { isRecord } from '../util/type-guards'
import style from './styles/license.scss'

// creativecommons.org/licenses/by-nc-sa/4.0/ → the CC mark plus one icon per module
const ccLicenseRe = /creativecommons\.org\/licenses\/([a-z-]+)\/(\d(?:\.\d)?)/
function ccMarks(url: string | undefined): { modules: CcModule[]; code: string } | null {
  const match = url ? ccLicenseRe.exec(url) : null
  if (!match) return null
  const modules = match[1].split('-') as CcModule[]
  if (modules.some(module => !CC_ICON_PATHS[module])) return null
  return { modules: ['cc', ...modules], code: `CC ${match[1].toUpperCase()} ${match[2]}` }
}

function frontmatterString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

export default (() => {
  const License: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const frontmatter = isRecord(fileData.frontmatter) ? fileData.frontmatter : undefined
    const license = frontmatterString(frontmatter?.license)
    if (!license) return null

    const licenseUrl =
      frontmatterString(frontmatter?.license_url) ?? frontmatterString(frontmatter?.licenseUrl)
    const cc = ccMarks(licenseUrl)

    return (
      <section data-license class={classNames(displayClass, 'license-meta', 'main-col')}>
        <h2 id="license-label">license</h2>
        <p>
          {licenseUrl ? (
            <a href={licenseUrl} target="_blank" rel="noopener noreferrer" title={license}>
              {cc ? (
                <>
                  {cc.modules.map(module => (
                    <svg
                      class="cc-icon"
                      viewBox={CC_ICON_VIEWBOX}
                      aria-hidden="true"
                      focusable="false"
                    >
                      <path fill="currentColor" d={CC_ICON_PATHS[module]} />
                    </svg>
                  ))}
                  <span>{cc.code}</span>
                </>
              ) : (
                <span>{license}</span>
              )}
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
