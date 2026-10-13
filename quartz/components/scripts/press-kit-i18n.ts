import type { ActivityCardData } from '../../util/press-kit-activity'
import type { Locale } from '../../util/triathlon-presentation'
import { triLabel } from '../../util/triathlon-i18n'
import { TRI_LOCALE_KEY } from '../triathlon/runtime/preferences'

export function cardLocale(): Locale {
  try {
    const locale = localStorage.getItem(TRI_LOCALE_KEY)
    if (locale === 'en' || locale === 'fr') return locale
  } catch {}
  return document.documentElement.lang.toLowerCase().startsWith('fr') ? 'fr' : 'en'
}

export function activityMetricLabel(label: string, locale = cardLocale()): string {
  return triLabel(locale, label.replace(/(?:\s*·\s*(?:trace|estimated?|estimation))+$/i, ''))
}

// The share panel keeps every metric label to one line; the full label stays in the toggle's
// accessible name and title.
const SHORT_METRIC_LABELS: Record<string, string> = {
  'average ambient temperature': 'avg ambient temp',
  'ambient temperature': 'ambient temp',
  'Feels Like Elevation': 'feels-like elev.',
  'headwind when into it / crosswind': 'headwind / crosswind',
  'yaw and wind coverage': 'yaw / wind cover',
  'performance condition': 'perf. condition',
  'vertical oscillation': 'vert. oscillation',
  'ground contact balance': 'GCT balance',
  'step speed loss percent': 'step speed loss %',
  'heart rate cost over terrain': 'hr cost / terrain',
  "effet d'entraînement": 'effet entraîn.',
  'indice de variabilité': 'indice variab.',
  "température de l'appareil": 'temp. appareil',
  'température ambiante': 'temp. ambiante',
  'température ambiante moyenne': 'temp. ambiante moy.',
  'couverture nuageuse moyenne': 'nuages moy.',
  'part de vent de face': 'part vent de face',
  'plus longue période de vent de face': 'vent de face max',
  'vitesse du vent ambiant': 'vit. vent ambiant',
  'plage de vitesse du vent': 'plage vit. vent',
  'vent de face quand il est contre / vent latéral': 'vent face / latéral',
  "rapport de vitesse de l'air apparente": 'rapport air apparent',
  'lacet et couverture du vent': 'lacet / couv. vent',
  'condition de performance': 'condition perf.',
  'indice de contrainte thermique': 'contrainte therm.',
  'température centrale': 'temp. centrale',
  'température cutanée': 'temp. cutanée',
  'oscillation verticale': 'oscill. verticale',
  'temps de contact au sol': 'contact au sol',
  'équilibre du temps de contact au sol': 'équil. contact sol',
  'perte de vitesse au sol': 'perte vit. sol',
  'pourcentage de perte de vitesse au sol': '% perte vit. sol',
  "facteur de charge d'impact": "charge d'impact",
  'coût cardiaque et relief': 'coût FC / relief',
}

export const shortMetricLabel = (label: string): string => SHORT_METRIC_LABELS[label] ?? label

export function localizeActivity(
  activity: ActivityCardData,
  locale = cardLocale(),
): ActivityCardData {
  return {
    ...activity,
    stats: activity.stats.map(stat => ({
      ...stat,
      label: activityMetricLabel(stat.label, locale),
    })),
    charts: activity.charts.map(chart => ({
      ...chart,
      title: triLabel(locale, chart.title),
      points: chart.points.map(point => ({ ...point, label: triLabel(locale, point.label) })),
    })),
  }
}
