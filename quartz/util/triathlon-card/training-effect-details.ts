import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { dlabel } from './metric-text'
import {
  trainingEffectGroup,
  trainingEffectNote,
  trainingEffectScore,
} from './training-effect-data'

export const calculatedTrainingEffectEvidenceText = (
  presentation: TriathlonPresentation,
  d: StravaActivityDetail,
): string | null => {
  const effect = d.calculatedTrainingEffect
  if (!effect) return null
  const french = presentation.locale === 'fr'
  const locale = french ? 'fr-CA' : 'en-US'
  const number = (value: number, digits = 1): string =>
    value.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits })
  const aerobic = effect.evidence.aerobic
  const aerobicText = french
    ? `Aérobie ${number(effect.aerobic)} à partir ${
        aerobic.source === 'relative-effort'
          ? `de l'effort relatif ${number(aerobic.load)}`
          : `de la charge d'exercice ${number(aerobic.load)}`
      }.`
    : `Aerobic ${number(effect.aerobic)} uses ${
        aerobic.source === 'relative-effort'
          ? `relative effort ${number(aerobic.load)}`
          : `exercise load ${number(aerobic.load)}`
      }.`
  const anaerobic = effect.evidence.anaerobic
  let anaerobicText: string
  if (anaerobic.source === 'power') {
    anaerobicText = french
      ? `Anaérobie ${number(effect.anaerobic)} à partir de ${anaerobic.effortCount.toLocaleString(locale)} efforts brefs au-dessus de l'eCP ${number(anaerobic.criticalPowerWatts)} W et de l'eW′ ${number(anaerobic.wPrimeKilojoules)} kJ. L'estimation répartit ces efforts sur ${dlabel(d.movingTimeS)} de déplacement.`
      : `Anaerobic ${number(effect.anaerobic)} uses ${anaerobic.effortCount.toLocaleString(locale)} short power efforts above eCP ${number(anaerobic.criticalPowerWatts)} W and eW′ ${number(anaerobic.wPrimeKilojoules)} kJ. The estimate spreads those efforts across ${dlabel(d.movingTimeS)} of moving time.`
  } else if (anaerobic.source === 'pace') {
    anaerobicText = french
      ? `Anaérobie ${number(effect.anaerobic)} à partir de ${dlabel(anaerobic.weightedSeconds)} d'efforts brefs pondérés par l'allure.`
      : `Anaerobic ${number(effect.anaerobic)} uses ${dlabel(anaerobic.weightedSeconds)} of weighted short pace efforts.`
  } else {
    anaerobicText = french
      ? `Anaérobie ${number(effect.anaerobic)} à partir de ${dlabel(anaerobic.seconds)} dans les zones de fréquence cardiaque supérieures.`
      : `Anaerobic ${number(effect.anaerobic)} uses ${dlabel(anaerobic.seconds)} in the upper heart-rate zones.`
  }
  return `${french ? 'Estimation calculée' : 'Calculated estimate'}. ${aerobicText} ${anaerobicText}`
}

export const buildSaunaHeatTrainingLoad = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const sauna = d.sauna
  if (d.sport !== 'sauna' || !sauna) return null
  const recorded = sauna.heatTrainingLoad
  if (recorded == null || !Number.isFinite(recorded)) return null
  const score = Math.min(10, Math.max(0, recorded))
  const locale = f.presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  const titleKey = 'heat training load'
  const label = triText(f.presentation.locale, titleKey)
  const wrap = f.el('section', 'tri-zone tri-sauna-htl', undefined, {
    'aria-label': label,
    'data-i18n-aria-label': titleKey,
    'data-sauna-htl-source': sauna.source,
  })
  const head = f.el('div', 'tri-sauna-htl-head')
  const title = f.el('div', 'tri-zone-title')
  f.add(
    title,
    f.el('span', 'tri-sauna-htl-title-full', label, { 'data-i18n': titleKey }),
    f.el('span', 'tri-sauna-htl-title-short', 'HTL'),
  )
  f.add(
    head,
    title,
    f.el(
      'span',
      'tri-sauna-htl-score',
      `${score.toLocaleString(locale, { maximumFractionDigits: 1 })} / 10`,
    ),
  )
  const meter = f.el('div', 'tri-sauna-htl-meter', undefined, {
    role: 'meter',
    'aria-label': label,
    'data-i18n-aria-label': titleKey,
    'aria-valuemin': '0',
    'aria-valuemax': '10',
    'aria-valuenow': String(score),
  })
  for (let index = 0; index < 10; index++) {
    const segment = f.el('span', 'tri-sauna-htl-segment', undefined, { 'aria-hidden': 'true' })
    f.add(
      segment,
      f.el('span', 'tri-sauna-htl-fill', undefined, {
        style: `width:${(Math.min(1, Math.max(0, score - index)) * 100).toFixed(1)}%`,
      }),
    )
    f.add(meter, segment)
  }
  f.add(wrap, head, meter)
  return wrap
}

export const buildTrainingEffectDetails = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const garmin = d.garmin
  const hasGarminDetails =
    garmin?.aerobicTrainingEffect != null ||
    garmin?.anaerobicTrainingEffect != null ||
    garmin?.aerobicTrainingEffectMessage != null ||
    garmin?.anaerobicTrainingEffectMessage != null
  const calculated = hasGarminDetails ? null : d.calculatedTrainingEffect
  if (!hasGarminDetails && !calculated) return null
  const locale = f.presentation.locale === 'fr' ? 'fr-CA' : 'en-US'
  const effects = [
    {
      label: 'aerobic',
      score: garmin?.aerobicTrainingEffect ?? calculated?.aerobic ?? null,
      note: trainingEffectNote(
        'aerobic',
        garmin?.aerobicTrainingEffect ?? calculated?.aerobic ?? null,
        hasGarminDetails ? (garmin?.aerobicTrainingEffectMessage ?? null) : null,
      ),
      group: trainingEffectGroup(
        'aerobic',
        hasGarminDetails ? (garmin?.aerobicTrainingEffectMessage ?? null) : null,
        hasGarminDetails ? (garmin?.trainingEffectLabel ?? null) : null,
      ),
    },
    {
      label: 'anaerobic',
      score: garmin?.anaerobicTrainingEffect ?? calculated?.anaerobic ?? null,
      note: trainingEffectNote(
        'anaerobic',
        garmin?.anaerobicTrainingEffect ?? calculated?.anaerobic ?? null,
        hasGarminDetails ? (garmin?.anaerobicTrainingEffectMessage ?? null) : null,
      ),
      group: trainingEffectGroup(
        'anaerobic',
        hasGarminDetails ? (garmin?.anaerobicTrainingEffectMessage ?? null) : null,
        hasGarminDetails ? (garmin?.trainingEffectLabel ?? null) : null,
      ),
    },
  ].filter(effect => effect.score != null || effect.note != null)
  if (effects.length === 0) return null
  const titleKey = 'training effect'
  const calculatedEvidence = calculated
    ? calculatedTrainingEffectEvidenceText(f.presentation, d)
    : null
  const wrap = f.el('section', 'tri-zone tri-training-effect', undefined, {
    'aria-label': triText(f.presentation.locale, titleKey),
    'data-i18n-aria-label': titleKey,
    'data-training-effect-source': calculated ? 'calculated' : 'garmin',
  })
  f.add(
    wrap,
    f.el('div', 'tri-zone-title', triText(f.presentation.locale, titleKey), {
      'data-i18n': titleKey,
      ...(calculatedEvidence
        ? { 'data-gloss': '', 'data-gloss-def': calculatedEvidence, tabindex: '0' }
        : {}),
    }),
  )
  const list = f.el('div', 'tri-training-effect-list')
  for (const effect of effects) {
    const score = trainingEffectScore(effect.score)
    const item = f.el('div', 'tri-training-effect-item', undefined, {
      'data-training-effect-group': effect.group,
    })
    const meter = f.el(
      'div',
      'tri-training-effect-meter',
      undefined,
      score == null
        ? { 'aria-hidden': 'true' }
        : {
            role: 'meter',
            'aria-label': effect.label,
            'data-i18n-aria-label': effect.label,
            'aria-valuemin': '0',
            'aria-valuemax': '5',
            'aria-valuenow': `${score}`,
          },
    )
    f.add(
      meter,
      f.el('span', 'tri-training-effect-meter-fill', undefined, {
        style: `--tri-training-effect-progress:${((score ?? 0) * 20).toFixed(1)}%`,
      }),
    )
    f.add(
      item,
      f.el('span', 'tri-training-effect-label', effect.label, { 'data-i18n': effect.label }),
      meter,
      f.el(
        'span',
        'tri-training-effect-score',
        effect.score?.toLocaleString(locale, {
          minimumFractionDigits: 1,
          maximumFractionDigits: 1,
        }) ?? '—',
      ),
    )
    if (effect.note) f.add(item, f.el('p', 'tri-training-effect-note', effect.note))
    f.add(list, item)
  }
  f.add(wrap, list)
  return wrap
}
