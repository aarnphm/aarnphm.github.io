import type { CardModel, Fonts, FormatSpec, Palette } from './spec'
import { captureLayer, type CardLayer } from '../press-kit-layers'
import { runsFromText } from '../press-kit-text'
import { drawRule, label, monoFont } from './chrome'
import { ASCENT, draw, fit, measure } from './typography'

export function drawCalendar(
  ctx: CanvasRenderingContext2D,
  fonts: Fonts,
  model: CardModel,
  spec: FormatSpec,
  palette: Palette,
  layers?: CardLayer[],
) {
  const calendar = model.calendar
  if (!calendar) return
  const [top, right, bottom, left] = spec.pad
  const width = spec.width - left - right
  const title = model.customTitle
    ? fit(
        ctx,
        fonts,
        model.title,
        { voice: 'header', leading: 1.1 },
        width,
        spec.meta * 4,
        [spec.meta * 1.3, spec.meta * 2],
        2,
      )
    : undefined
  const headingHeight = title ? title.height + spec.gap : 0
  const available = spec.height - top - bottom - headingHeight
  const count = Math.max(1, calendar.races.length)
  const rowHeight = Math.min(available / count, spec.meta * 7)
  const total = rowHeight * count + headingHeight
  let y = top + (spec.height - top - bottom - total) / 2
  if (title)
    y += draw(ctx, title, left, y, palette.ink, undefined, 'left', 0, layers, 'titre') + spec.gap
  if (!calendar.races.length) {
    const empty = fit(
      ctx,
      fonts,
      runsFromText(calendar.empty),
      { voice: 'body', leading: 1.2 },
      width,
      rowHeight,
      [spec.meta, spec.meta * 1.5],
    )
    draw(
      ctx,
      empty,
      left,
      y + (rowHeight - empty.height) / 2,
      palette.muted,
      undefined,
      'left',
      0,
      layers,
      'calendrier',
    )
    return
  }
  const dateWidth = spec.meta * 3.7
  const gap = spec.meta
  const mainX = left + dateWidth + gap
  const mainWidth = width - dateWidth - gap
  const nameSize = Math.min(spec.meta * 1.45, rowHeight * 0.24)
  const metaSize = Math.min(spec.meta * 0.76, rowHeight * 0.115)
  const chipSize = metaSize * 0.85
  const chipHeight = chipSize * 1.85
  const mono = monoFont(fonts, metaSize)
  const chipFont = monoFont(fonts, chipSize)
  const daySize = Math.min(spec.meta * 2.05, rowHeight * 0.33)
  for (const [index, race] of calendar.races.entries()) {
    const rowY = y + index * rowHeight
    const statusWidth = race.status ? measure(ctx, mono, race.status) + gap : 0
    const name = fit(
      ctx,
      fonts,
      runsFromText(race.title),
      { voice: 'header', leading: 1.08 },
      mainWidth - statusWidth,
      rowHeight * 0.45,
      [nameSize * 0.78, nameSize],
      2,
    )
    const scheduleHeight = race.schedule.length ? chipHeight + metaSize * 0.65 : 0
    const contentHeight = name.height + metaSize * 1.8 + scheduleHeight
    const contentY = rowY + (rowHeight - contentHeight) / 2
    captureLayer(
      ctx,
      layers,
      `calendar/${race.id}`,
      race.title,
      { x: left, y: rowY, width, height: rowHeight },
      context => {
        drawRule(context, left, rowY, width, palette.rule)
        const dateHeight = metaSize * 2.5 + daySize
        const dateY = rowY + (rowHeight - dateHeight) / 2
        if (race.pending) {
          const pending = fit(
            context,
            fonts,
            runsFromText(race.pending),
            { voice: 'body', leading: 1.2 },
            dateWidth,
            rowHeight * 0.7,
            [metaSize, metaSize],
            3,
          )
          draw(context, pending, left, rowY + (rowHeight - pending.height) / 2, palette.muted)
        } else {
          label(
            context,
            mono,
            race.month.toUpperCase(),
            left,
            dateY + metaSize,
            dateWidth,
            palette.muted,
          )
          const day = fit(
            context,
            fonts,
            runsFromText(race.day),
            { voice: 'header', leading: 1 },
            dateWidth,
            daySize * 1.2,
            [daySize * 0.5, daySize],
            1,
          )
          draw(context, day, left, dateY + metaSize * 1.3, palette.ink)
          label(context, mono, race.weekday, left, dateY + dateHeight, dateWidth, palette.muted)
        }
        draw(context, name, mainX, contentY, palette.ink)
        if (race.status)
          label(
            context,
            mono,
            race.status,
            mainX,
            contentY + nameSize * ASCENT,
            mainWidth,
            palette.muted,
            'right',
          )
        const baseline = contentY + name.height + metaSize * 1.35
        const mark = metaSize * 0.42
        const formatWidth = measure(context, mono, race.format)
        const formatX = left + width - formatWidth
        label(
          context,
          mono,
          race.location,
          mainX,
          baseline,
          Math.max(0, mainWidth - formatWidth - gap * 1.4),
          palette.muted,
        )
        context.fillStyle = race.color || palette.ink
        context.fillRect(formatX - mark - metaSize * 0.5, baseline - mark * 1.5, mark, mark)
        label(context, mono, race.format, formatX, baseline, formatWidth, palette.muted)
        let chipX = mainX
        const chipY = baseline + metaSize * 0.65
        for (const day of race.schedule) {
          const chipWidth = measure(context, chipFont, day.label) + chipSize * 1.2
          if (chipX + chipWidth > left + width) break
          context.fillStyle = day.race ? race.color || palette.ink : palette.bg
          context.fillRect(chipX, chipY, chipWidth, chipHeight)
          if (!day.race) {
            context.strokeStyle = palette.rule
            context.lineWidth = 1
            context.strokeRect(chipX + 0.5, chipY + 0.5, chipWidth - 1, chipHeight - 1)
          }
          label(
            context,
            chipFont,
            day.label,
            chipX + chipSize * 0.6,
            chipY + (chipHeight - chipSize) / 2 + chipSize * ASCENT,
            chipWidth - chipSize * 1.2,
            day.race ? race.onColor || palette.bg : palette.muted,
          )
          chipX += chipWidth + chipSize * 0.4
        }
        if (index === calendar.races.length - 1)
          drawRule(context, left, rowY + rowHeight - 1, width, palette.rule)
      },
    )
  }
}
