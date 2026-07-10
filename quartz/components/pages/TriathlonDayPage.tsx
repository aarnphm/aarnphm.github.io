import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { ATHLETE } from '../../plugins/stores/analytics'
import { emptyPayload } from '../../plugins/stores/strava'
import { htmlToJsx } from '../../util/jsx'
import { classNames } from '../../util/lang'
import { joinSegments, pathToRoot } from '../../util/path'
import { triathlonDateFromSlug, triathlonMonthSlug } from '../../util/triathlon-date-route'
// @ts-ignore
import script from '../scripts/triathlon.inline'
import style from '../styles/triathlon.scss'
import { triathlonDayCard, triathlonDayExtras, triathlonDayProps } from '../triathlon-day-card'
import { TriathlonSubnav } from './triathlon-panels'

export default (() => {
  const TriathlonDayPage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const date = triathlonDateFromSlug(fileData.slug ?? '')
    if (!date) return null
    const payload = fileData.stravaPayload ?? emptyPayload()
    const root = pathToRoot(fileData.slug!)
    const monthSlug = triathlonMonthSlug(date.slice(0, 4), date.slice(5, 7))
    const extras = {
      ...triathlonDayExtras(fileData, date),
      expanded: true,
      ...(monthSlug ? { dateHref: joinSegments(root, monthSlug) } : {}),
    }
    const detailPath = joinSegments(root, 'static/strava-detail.json')
    const card = triathlonDayCard(date, payload.totalCount > 0 ? payload : null, extras, {
      zones: payload.zones,
      curveRef: payload.powerCurveRef,
      curveYearRef: payload.powerCurveYearRef,
      curveYear: payload.powerCurveYear,
      ftp: ATHLETE.ftp,
      goalFtp: ATHLETE.goalFTP,
      vt1: null,
    })

    return (
      <article
        class={classNames(
          displayClass,
          'triathlon',
          'tri-subpage',
          'tri-day-page',
          'all-col',
          'popover-hint',
        )}
        data-tri-view="day"
      >
        <TriathlonSubnav active="on" root={root} />
        <div
          class="tri-day-embed"
          {...triathlonDayProps(extras, date)}
          data-detail-path={detailPath}
        >
          {htmlToJsx(fileData.filePath!, card)}
        </div>
      </article>
    )
  }

  TriathlonDayPage.css = style
  TriathlonDayPage.afterDOMLoaded = script

  return TriathlonDayPage
}) satisfies QuartzComponentConstructor
