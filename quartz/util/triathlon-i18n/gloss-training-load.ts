import type { Gloss } from './types'

export const en: Record<string, Gloss> = {
  tss: {
    term: 'training stress score (TSS)',
    def: 'TSS is the daily sum of session stress. Each session uses intensity factor squared, multiplied by duration in hours and 100. Intensity is estimated from threshold-adjusted speed.',
  },
  cp: {
    term: 'critical power (CP)',
    def: 'Critical power is the sustainable-power asymptote fitted from maximal 3, 7, and 12 minute cycling efforts. This site estimates it from complete device-power windows and reports the fit provenance.',
  },
  wprime: {
    term: 'W′',
    def: 'W′ is the finite work capacity above critical power. It is the slope of the fitted power-duration model and is reported in kilojoules.',
  },
  'torque effectiveness': {
    term: 'torque effectiveness (TE)',
    def: 'Torque effectiveness compares the positive torque that drives the crank with negative torque that resists it during each revolution. A value of 100% means no negative torque was recorded. Interpret left and right trends alongside power and cadence; the metric has no universal target.',
  },
  'pedal smoothness': {
    term: 'pedal smoothness (PS)',
    def: 'Pedal smoothness is average power divided by peak power across one crank cycle. A higher percentage means power was applied more evenly through the revolution. It describes the shape of power delivery; total power and efficiency are separate measurements.',
  },
  'power phase': {
    term: 'power phase (PP)',
    def: 'Power phase is the crank-angle interval where each leg produces driving force. 0° is the 12 o’clock position and 180° is the 6 o’clock position; start and end mark the interval boundaries. The graph can cross 360°→0° because the crank cycle is circular.',
  },
  ctl: {
    term: 'fitness (CTL)',
    def: 'Fitness is your average daily training load over the past 42 days. Recent days count more. It rises when you train consistently and falls when you train less.',
  },
  atl: {
    term: 'fatigue (ATL)',
    def: 'Fatigue is your average daily training load over the past 7 days. Recent days count more. A high value means you have done more training lately.',
  },
  tsb: {
    term: 'form (TSB)',
    def: 'Form is fitness minus fatigue. A positive value means your recent load is below your longer term load, so you should be fresher. A negative value means your recent load is higher.',
  },
  acwr: {
    term: 'ACWR',
    def: 'ACWR divides your training load from the past 7 days by your load from the past 28 days. A value from 0.8 to 1.3 is the target range. A value above 1.5 means your recent load rose sharply.',
  },
  ramp: {
    term: 'ramp',
    def: 'Ramp is the change in fitness from one week to the next. A positive value means your training load is building. A large jump means your load increased quickly.',
  },
  monotony: {
    term: 'monotony',
    def: "Monotony measures how similar your daily training loads were during the week. A higher value means the days had similar loads. A value above about 2, together with a high weekly load, is a warning sign in Foster's method.",
  },
  strain: {
    term: 'strain',
    def: 'Strain is your weekly training load multiplied by monotony. It is high when the week had a high load and little change between days.',
  },
  load: {
    term: 'load',
    def: 'Load estimates how hard each workout was from its pace and duration. About 100 points means one hour at threshold effort. The activity records each sensor value separately. This score uses only pace and duration.',
  },
  score: {
    term: 'readiness',
    def: 'Readiness is a score from 0 to 100. Fitness compared with the race demand makes up 45%. The distance you have covered in training for each leg makes up 55%.',
  },
  binding: {
    term: 'binding leg',
    def: 'The binding leg is the sport holding your readiness score down the most. It combines how much of the race distance you have covered with how recently you trained that sport. Focus on this sport first.',
  },
  predtime: {
    term: 'predicted time',
    def: 'Predicted time is the estimated finish time for all three legs and both transitions. The pace model predicts each leg. Until the model loads, the estimate adjusts your threshold pace for the race distance. Each transition adds 5 minutes.',
  },
  conf: {
    term: 'confidence',
    def: 'Confidence shows how much recent data supports an estimate. Firm means there are enough recent efforts. Low means there are only a few efforts. Stale means the latest effort is more than 45 days old. Prior means there is no personal data, so the estimate uses a general starting value.',
  },
  threshold: {
    term: 'threshold pace',
    def: 'Threshold pace is your estimated pace for an effort lasting about one hour. For running, it is the hill-adjusted pace at your lactate-threshold heart rate, from steady parts of runs in the last 56 days. For swimming and cycling speed, it uses your faster sessions. The pace model uses it as a starting point.',
  },
}

export const fr: Record<string, Gloss> = {
  tss: {
    term: "score de stress d'entraînement (TSS)",
    def: "Le TSS est la somme quotidienne du stress des séances. Chaque séance utilise le facteur d'intensité au carré, multiplié par la durée en heures et par 100. L'intensité est estimée à partir de la vitesse ajustée au seuil.",
  },
  cp: {
    term: 'puissance critique (CP)',
    def: "La puissance critique est l'asymptote de puissance soutenable ajustée aux efforts maximaux de 3, 7 et 12 minutes. Ce site l'estime à partir de fenêtres complètes de puissance mesurée et indique leur provenance.",
  },
  wprime: {
    term: 'W′',
    def: "W′ est la capacité de travail finie au-dessus de la puissance critique. Elle correspond à la pente du modèle puissance-durée et s'exprime en kilojoules.",
  },
  'torque effectiveness': {
    term: 'efficacité du couple (TE)',
    def: "L'efficacité du couple compare le couple positif qui entraîne la manivelle au couple négatif qui s'y oppose pendant chaque tour. Une valeur de 100 % signifie qu'aucun couple négatif n'a été enregistré. Interprète les tendances gauche et droite avec la puissance et la cadence; cette mesure n'a pas de cible universelle.",
  },
  'pedal smoothness': {
    term: 'fluidité du pédalage (PS)',
    def: "La fluidité du pédalage est la puissance moyenne divisée par la puissance maximale sur un tour de manivelle. Une valeur plus élevée signifie que la puissance est répartie plus uniformément sur le tour. Elle décrit la forme de l'application de la puissance; la puissance totale et le rendement sont des mesures distinctes.",
  },
  'power phase': {
    term: 'phase de puissance (PP)',
    def: "La phase de puissance est l'intervalle angulaire pendant lequel chaque jambe produit une force motrice. 0° correspond à 12 h et 180° à 6 h ; le début et la fin délimitent l'intervalle. Le graphique peut franchir 360°→0°, car le cycle de manivelle est circulaire.",
  },
  ctl: {
    term: 'condition (CTL)',
    def: "La condition représente ta charge d'entraînement quotidienne moyenne sur les 42 derniers jours. Les jours récents comptent davantage. Elle monte quand tu t'entraînes régulièrement et baisse quand tu t'entraînes moins.",
  },
  atl: {
    term: 'fatigue (ATL)',
    def: "La fatigue représente ta charge d'entraînement quotidienne moyenne sur les 7 derniers jours. Les jours récents comptent davantage. Une valeur haute signifie que tu t'es plus entraîné récemment.",
  },
  tsb: {
    term: 'forme (TSB)',
    def: 'La forme est la condition moins la fatigue. Une valeur positive signifie que ta charge récente est sous ta charge à long terme, donc tu devrais être plus frais. Une valeur négative signifie que ta charge récente est plus haute.',
  },
  acwr: {
    term: 'ACWR',
    def: "L'ACWR divise ta charge des 7 derniers jours par ta charge des 28 derniers jours. La plage cible va de 0,8 à 1,3. Une valeur au-dessus de 1,5 signifie que ta charge récente a monté rapidement.",
  },
  ramp: {
    term: 'progression',
    def: "La progression est le changement de condition d'une semaine à l'autre. Une valeur positive signifie que ta charge augmente. Un grand saut signifie que ta charge a augmenté rapidement.",
  },
  monotony: {
    term: 'monotonie',
    def: "La monotonie mesure à quel point tes charges quotidiennes se ressemblent pendant la semaine. Une valeur haute signifie que les journées ont des charges proches. Une valeur au-dessus d'environ 2, avec une charge hebdomadaire haute, est un signal d'alerte dans la méthode de Foster.",
  },
  strain: {
    term: 'contrainte',
    def: 'La contrainte est la charge hebdomadaire multipliée par la monotonie. Elle est haute lorsque la semaine a une charge haute et peu de variation entre les jours.',
  },
  load: {
    term: 'charge',
    def: "La charge estime la difficulté de chaque séance à partir de l'allure et de la durée. Environ 100 points représentent une heure au seuil. L'activité enregistre aussi chaque valeur de capteur. Cette note utilise seulement l'allure et la durée.",
  },
  score: {
    term: 'préparation',
    def: "La préparation est une note de 0 à 100. La condition comparée aux exigences de la course représente 45 %. La distance couverte à l'entraînement pour chaque segment représente 55 %.",
  },
  binding: {
    term: 'segment limitant',
    def: 'Le segment limitant est la discipline qui réduit le plus ta note de préparation. Il combine la part de la distance de course déjà couverte avec la date de ton dernier entraînement dans cette discipline. Travaille ce segment en premier.',
  },
  predtime: {
    term: 'temps estimé',
    def: "Le temps estimé est le temps d'arrivée pour les trois segments et les deux transitions. Le modèle d'allure prédit chaque segment. Pendant son chargement, l'estimation ajuste ton allure seuil à la distance de course. Chaque transition ajoute 5 minutes.",
  },
  conf: {
    term: 'fiabilité',
    def: 'La fiabilité indique la quantité de données récentes derrière une estimation. "firm" signifie que les efforts récents sont assez nombreux. "low" signifie que les efforts sont peu nombreux. "stale" signifie que le dernier effort date de plus de 45 jours. "prior" signifie que les données personnelles manquent, donc le calcul utilise une valeur de départ générale.',
  },
  threshold: {
    term: 'allure seuil',
    def: "L'allure seuil estime ton allure pour un effort d'environ une heure. En course à pied, c'est l'allure ajustée aux côtes à ta fréquence cardiaque au seuil lactique, tirée des parties stables des courses des 56 derniers jours. En natation et pour la vitesse à vélo, elle utilise tes séances les plus rapides. Le modèle d'allure l'utilise comme point de départ.",
  },
}
