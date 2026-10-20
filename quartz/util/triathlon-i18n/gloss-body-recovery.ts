import type { Gloss } from './types'

export const en: Record<string, Gloss> = {
  lactate: {
    term: 'lactate threshold',
    def: 'LT2 is the highest effort you can hold for about one hour. Running LT2 is the hill-adjusted pace at your lactate-threshold heart rate (LTHR). A straight line through steady 8-minute parts of runs from the last 56 days gives this pace. LTHR comes from Garmin, or from the declared value when Garmin has none. Garmin also gives a running threshold pace, but Garmin calculates it from its VO2max estimate. The panel uses the Garmin pace only when it is within 5% of the heart-rate pace. Cycling LT2 is critical power from the last six weeks of power data. Swimming LT2 is the 90th percentile of session pace. These estimates do not measure blood lactate or ventilatory thresholds.',
  },
  trend: {
    term: 'pace trend',
    def: 'Pace trend follows fitness, not the mix of easy and hard sessions. For running, each session becomes the hill-adjusted pace at one reference heart rate. For cycling, each ride with a power meter becomes the power at one reference heart rate. A line through the steady 8-minute parts of every session gives the change per heartbeat, and the reference is the median heart rate of those parts. Swimming uses the session pace. A Kalman filter fits a level and a rate of change through the sessions. It gives less weight to a session that is far from the trend, and an open dot marks that session. The line is the level for the last eight weeks. The shaded area is the 80% range of the level, and it becomes wider over the two-week forecast. The dotted outline is the forecast from two weeks ago. "Band held" counts the days when the level was inside the range forecast two weeks before.',
  },
  weight: {
    term: 'body weight',
    def: 'Body weight comes from your daily weight records. It is used in recovery charts and in energy estimates that depend on weight.',
  },
  wtrend: {
    term: 'weight trend',
    def: 'Weight trend is the weekly rate of change in your logged weight. A negative value means your weight is decreasing.',
  },
  wgoal: {
    term: 'weight goal',
    def: 'Weight goal comes from Garmin Connect. The difference is your current weight minus your goal. The estimated date uses your current weekly trend and appears only when the trend is moving toward the goal.',
  },
  bodyfat: {
    term: 'body fat',
    def: 'Body fat is the percentage reported by the Garmin Index scale. Use the trend instead of one reading because hydration can change the result by about 1 percentage point.',
  },
  dexa: {
    term: 'DEXA body composition',
    def: 'DEXA is a lab scan that measures total mass and separates fat mass from lean mass. It also measures bone mineral content. Use it as the main body composition reading. The scale is useful for following daily changes.',
  },
  bmi: {
    term: 'BMI',
    def: 'BMI is your weight in kilograms divided by your height in metres squared. Muscle can raise BMI, so read it together with body fat.',
  },
  ffmi: {
    term: 'FFMI',
    def: 'Fat-free mass index is your fat-free mass in kilograms divided by your height in metres squared. DEXA supplies the main reading. Daily estimates use weight and body fat from the Garmin Index scale.',
  },
  bmr: {
    term: 'BMR (Katch McArdle)',
    def: 'BMR estimates how many calories your body uses each day at rest. This page uses the Katch McArdle formula, which starts from lean mass. The estimate changes when the Garmin Index scale reports a new body fat value.',
  },
  effort: {
    term: 'relative effort',
    def: "Strava's Relative Effort score uses heart rate or your own effort rating. This chart adds the scores from each calendar week. The shaded range is based on the three previous full weeks.",
  },
  hrv: {
    term: 'HRV',
    def: 'HRV is the change in time between heartbeats, measured in milliseconds. This chart compares your 7 day average with a 28 day personal baseline. A value more than one standard deviation below your baseline is far below your usual range and can be a sign of poor recovery.',
  },
  rhr: {
    term: 'resting heart rate',
    def: 'Resting heart rate is your lowest overnight heart rate. A rise of at least 5 bpm, or more than one standard deviation above your 28 day baseline, can be an early sign of fatigue or illness.',
  },
  tempdev: {
    term: 'temperature deviation',
    def: 'Temperature deviation is the change in skin temperature from your personal baseline. An increase of at least 0.5 °C can be a sign that your immune system is responding to something. It may appear 24 to 48 hours before symptoms.',
  },
  ambienttemp: {
    term: 'ambient workout temperature',
    def: 'Ambient temperature is the overlap-weighted Apple WeatherKit estimate sampled at the route coordinate nearest each hourly midpoint. Missing weather remains a gap. Strava device temperature is reported separately.',
  },
  heatstrain: {
    term: 'CORE Heat Strain Index',
    def: 'CORE combines core and skin temperature into a real-time Heat Strain Index. Values at or above 3.0 count toward heat exposure on this page.',
  },
  heatdose: {
    term: 'heat exposure dose',
    def: 'CORE Heat Strain Index at or above 3.0 supplies the primary dose. A GPS run or ride above 22 °C supplies the fallback dose when CORE data is absent. Sixty hot workout minutes equal one dose. Recorded sauna HTL adds HTL / 10, with a passive cap of 8 HTL per day and a combined daily cap of one dose. Missing sauna HTL earns no inferred credit. The proxy targets 14 doses, holds for three days without heat, then decays by 2.5% per day.',
  },
  heatacclimation: {
    term: 'heat acclimatisation proxy',
    def: 'This Garden proxy combines CORE heat strain, ambient workout exposure from WeatherKit or Strava, and manually recorded sauna HTL. Sauna duration, room temperature, humidity, and cooldown remain separate from measured HSI. The HTL conversion is a Garden heuristic, not the CORE Heat Adaptation Score or a physiological measurement.',
  },
  sleepdebt: {
    term: 'sleep debt',
    def: 'Sleep debt adds up the time you slept below 7 hours during the past 14 nights. This chart uses 7 hours as its target. Athletes often need 8 to 10 hours.',
  },
}

export const fr: Record<string, Gloss> = {
  lactate: {
    term: 'seuil lactique',
    def: "Le LT2 est l'effort le plus élevé que tu peux tenir pendant environ une heure. En course à pied, le LT2 est l'allure ajustée aux côtes à ta fréquence cardiaque au seuil lactique (FCSL). Une droite tracée à travers les parties stables de 8 minutes des courses des 56 derniers jours donne cette allure. La FCSL vient de Garmin, ou de la valeur déclarée quand Garmin n'en a pas. Garmin donne aussi une allure au seuil en course à pied, mais Garmin la calcule à partir de son estimation de VO2max. Le panneau utilise l'allure Garmin seulement quand elle est à moins de 5 % de l'allure tirée de la fréquence cardiaque. À vélo, le LT2 est la puissance critique des six dernières semaines de données de puissance. En natation, le LT2 est le 90e centile de l'allure des séances. Ces estimations ne mesurent ni le lactate sanguin ni les seuils ventilatoires.",
  },
  trend: {
    term: "tendance d'allure",
    def: "La tendance d'allure suit la condition physique, pas le mélange de séances faciles et difficiles. En course à pied, chaque séance devient l'allure ajustée aux côtes à une fréquence cardiaque de référence. À vélo, chaque sortie avec un capteur de puissance devient la puissance à une fréquence cardiaque de référence. Une droite à travers les parties stables de 8 minutes de chaque séance donne le changement par battement, et la référence est la fréquence cardiaque médiane de ces parties. En natation, la tendance utilise l'allure de séance. Un filtre de Kalman ajuste un niveau et un rythme de changement à travers les séances. Il donne moins de poids à une séance éloignée de la tendance, et un point vide marque cette séance. La ligne est le niveau des huit dernières semaines. La zone ombrée est la plage à 80 % du niveau, et elle s'élargit sur les deux semaines de prévision. Le contour pointillé est la prévision d'il y a deux semaines. « Bande tenue » compte les jours où le niveau était dans la plage prévue deux semaines plus tôt.",
  },
  weight: {
    term: 'poids',
    def: "Le poids vient de tes mesures quotidiennes. Il est utilisé dans les graphiques de récupération et dans les estimations d'énergie qui dépendent du poids.",
  },
  wtrend: {
    term: 'tendance de poids',
    def: 'La tendance de poids est le taux de changement hebdomadaire de ton poids enregistré. Une valeur négative signifie que ton poids baisse.',
  },
  wgoal: {
    term: 'objectif de poids',
    def: "L'objectif de poids vient de Garmin Connect. La différence est ton poids actuel moins ton objectif. La date estimée utilise ta tendance hebdomadaire et apparaît seulement lorsque tu te rapproches de l'objectif.",
  },
  bodyfat: {
    term: 'masse grasse',
    def: "La masse grasse est le pourcentage indiqué par la balance Garmin Index. Utilise la tendance au lieu d'une seule mesure, car l'hydratation peut changer le résultat d'environ 1 point de pourcentage.",
  },
  dexa: {
    term: 'composition corporelle DEXA',
    def: 'Le DEXA est un scan en laboratoire qui mesure la masse totale et sépare la masse grasse de la masse maigre. Il mesure aussi le contenu minéral osseux. Utilise le DEXA comme mesure principale de composition corporelle. La balance sert à suivre les changements quotidiens.',
  },
  bmi: {
    term: 'IMC',
    def: "L'IMC est ton poids en kilogrammes divisé par ta taille en mètres au carré. Les muscles peuvent augmenter l'IMC, donc lis cette valeur avec la masse grasse.",
  },
  ffmi: {
    term: 'FFMI (indice de masse maigre)',
    def: "L'indice de masse maigre est ta masse maigre en kilogrammes divisée par ta taille en mètres au carré. Le DEXA fournit la mesure principale. Les estimations quotidiennes utilisent le poids et la masse grasse de la balance Garmin Index.",
  },
  bmr: {
    term: 'métabolisme de base (Katch McArdle)',
    def: "Le métabolisme de base estime les calories que ton corps utilise chaque jour au repos. Cette page utilise la formule de Katch McArdle, qui part de la masse maigre. L'estimation change lorsque la balance Garmin Index indique une nouvelle valeur de masse grasse.",
  },
  effort: {
    term: 'effort relatif',
    def: "Le score d'effort relatif de Strava utilise la fréquence cardiaque ou ta propre note d'effort. Ce graphique additionne les scores de chaque semaine civile. La zone ombrée repose sur les trois semaines complètes précédentes.",
  },
  hrv: {
    term: 'VFC',
    def: "La VFC est la variation du temps entre les battements. Elle est mesurée en millisecondes. Ce graphique compare ta moyenne sur 7 jours avec ta référence personnelle sur 28 jours. Une valeur située à plus d'un écart type sous ta référence peut signaler une mauvaise récupération.",
  },
  rhr: {
    term: 'fréquence cardiaque au repos',
    def: "La fréquence cardiaque au repos est ta fréquence la plus basse pendant la nuit. Une hausse d'au moins 5 bpm, ou de plus d'un écart type au-dessus de ta référence sur 28 jours, peut être un signe précoce de fatigue ou de maladie.",
  },
  tempdev: {
    term: 'écart de température',
    def: "L'écart de température est la différence entre ta température cutanée et ta référence personnelle. Une hausse d'au moins 0,5 °C peut signaler une réponse du système immunitaire. Elle peut apparaître 24 à 48 heures avant les symptômes.",
  },
  ambienttemp: {
    term: "température ambiante d'une séance",
    def: "La température ambiante est l'estimation Apple WeatherKit pondérée selon le chevauchement et relevée au point du parcours le plus proche du milieu de chaque heure. Une donnée météo manquante reste vide. La température de l'appareil Strava est indiquée séparément.",
  },
  heatstrain: {
    term: 'indice de contrainte thermique CORE',
    def: "CORE combine la température centrale et la température cutanée pour calculer un indice de contrainte thermique en temps réel. Les valeurs d'au moins 3,0 comptent dans l'exposition à la chaleur sur cette page.",
  },
  heatdose: {
    term: "dose d'exposition à la chaleur",
    def: "L'indice de contrainte thermique CORE d'au moins 3,0 fournit la dose principale. Une course ou une sortie à vélo avec GPS au-dessus de 22 °C fournit la dose de repli lorsque les données CORE sont absentes. Soixante minutes chaudes d’exercice représentent une dose. La HTL de sauna consignée ajoute HTL / 10, avec un plafond passif de 8 HTL par jour et un plafond combiné d’une dose quotidienne. Une HTL absente ne donne aucun crédit estimé. L'indice vise 14 doses, reste stable pendant trois jours sans chaleur, puis baisse de 2,5 % par jour.",
  },
  heatacclimation: {
    term: "indice d'acclimatation à la chaleur",
    def: 'Cet indice Garden combine la contrainte thermique CORE, l’exposition ambiante WeatherKit ou Strava et la HTL de sauna consignée. La durée, la température du sauna, l’humidité et le refroidissement restent distincts du HSI mesuré. La conversion HTL est une approximation Garden, et non le score d’adaptation CORE ou une mesure physiologique.',
  },
  sleepdebt: {
    term: 'dette de sommeil',
    def: 'La dette de sommeil additionne le temps dormi sous 7 heures pendant les 14 dernières nuits. Ce graphique utilise une cible de 7 heures. Les athlètes ont souvent besoin de 8 à 10 heures.',
  },
}
