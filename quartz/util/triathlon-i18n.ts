export type Locale = 'en' | 'fr'

interface Gloss {
  term: string
  def: string
}

interface TriDict {
  ui: Record<string, string>
  gloss: Record<string, Gloss>
}

const TRI_LOCALE_KEY = 'tri-locale'
let locale: Locale = 'en'

export const triLocale = (): Locale => locale
export const setTriLocale = (v: Locale): void => {
  locale = v
}

const triLocaleTag = (): string => (locale === 'fr' ? 'fr-CA' : 'en-US')

export const triNumber = (
  value: number,
  minimumFractionDigits = 0,
  maximumFractionDigits = minimumFractionDigits,
): string => value.toLocaleString(triLocaleTag(), { minimumFractionDigits, maximumFractionDigits })

const triDate = (iso: string): Date | null => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
  if (!match) return null
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const date = new Date(Date.UTC(year, month - 1, day))
  return date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
    ? date
    : null
}

const triDateText = (iso: string, options: Intl.DateTimeFormatOptions): string => {
  const date = triDate(iso)
  return date ? date.toLocaleDateString(triLocaleTag(), { ...options, timeZone: 'UTC' }) : iso
}

export const triShortDate = (iso: string): string =>
  triDateText(iso, { month: 'short', day: 'numeric' })

export const triLongDate = (iso: string): string =>
  triDateText(iso, { year: 'numeric', month: 'short', day: 'numeric' })

export const triMonth = (iso: string): string => triDateText(iso, { month: 'short' })

export const triMonthYear = (iso: string): string =>
  triDateText(iso, { year: 'numeric', month: 'long' })

export const triWeekdayNarrow = (day: number): string =>
  new Date(Date.UTC(2024, 0, 7 + Math.min(6, Math.max(0, day))))
    .toLocaleDateString(triLocaleTag(), { weekday: 'narrow', timeZone: 'UTC' })
    .toUpperCase()

export type SwimActivityTextPoint = {
  elapsed: string
  cumulativeDistanceM: number
  windowStartDistanceM?: number
}

const swimTextNumber = (value: number, maximumFractionDigits = 0): string =>
  triNumber(value, 0, maximumFractionDigits)

export const swimActivityDistanceText = (distanceM: number): string =>
  `${swimTextNumber(distanceM)} m`

export const swimActivityHeaderValue = (
  kind: 'pace' | 'stroke',
  value: number,
  pace: string,
): string => (kind === 'pace' ? pace : swimTextNumber(value, 1))

export const swimActivityPointText = (point: SwimActivityTextPoint): string => {
  const end = swimTextNumber(point.cumulativeDistanceM)
  const distance =
    point.windowStartDistanceM == null
      ? `${end} m`
      : `${swimTextNumber(point.windowStartDistanceM)}–${end} m`
  return `${distance} · ${point.elapsed} ${locale === 'fr' ? 'écoulé' : 'elapsed'}`
}

export const swimActivityDisplayValue = (
  kind: 'pace' | 'stroke',
  value: number,
  pace: string,
): string =>
  kind === 'pace'
    ? `${pace} /100m`
    : `${swimTextNumber(value, 1)} ${locale === 'fr' ? 'coups/min' : 'str/min'}`

export const swimActivityValueText = (
  kind: 'pace' | 'stroke',
  point: SwimActivityTextPoint,
  value: number,
  pace: string,
): string => {
  const end = swimTextNumber(point.cumulativeDistanceM)
  const position =
    point.windowStartDistanceM == null
      ? locale === 'fr'
        ? `${end} mètres, temps écoulé ${point.elapsed}`
        : `${end} metres, ${point.elapsed} elapsed`
      : locale === 'fr'
        ? `bloc de ${swimTextNumber(point.cumulativeDistanceM - point.windowStartDistanceM)} mètres, de ${swimTextNumber(point.windowStartDistanceM)} à ${end} mètres, temps écoulé ${point.elapsed}`
        : `${swimTextNumber(point.cumulativeDistanceM - point.windowStartDistanceM)} metre block from ${swimTextNumber(point.windowStartDistanceM)} to ${end} metres, ${point.elapsed} elapsed`
  if (kind === 'pace')
    return locale === 'fr'
      ? `${position}, allure de nage ${pace} par 100 mètres`
      : `${position}, swim pace ${pace} per 100 metres`
  const rate = swimTextNumber(value, 1)
  return locale === 'fr'
    ? `${position}, fréquence de nage ${rate} coups par minute`
    : `${position}, stroke rate ${rate} strokes per minute`
}

export const swimActivityComparisonText = (
  kind: 'pace' | 'stroke',
  delta: number | null,
  priorCount: number | null,
): string => {
  if (delta == null || priorCount == null) return locale === 'fr' ? 'moy. activité' : 'activity avg'
  if (Math.abs(delta) < 0.05)
    return locale === 'fr'
      ? `identique aux ${priorCount} précédentes`
      : `same as prior ${priorCount}`
  const magnitude = swimTextNumber(Math.abs(delta), 1)
  if (kind === 'pace')
    return locale === 'fr'
      ? `${magnitude} s ${delta < 0 ? 'plus rapide' : 'plus lente'} que les ${priorCount} précédentes`
      : `${magnitude}s ${delta < 0 ? 'faster' : 'slower'} vs prior ${priorCount}`
  const signed = `${delta > 0 ? '+' : '−'}${magnitude}`
  return locale === 'fr'
    ? `${signed} coups/min vs ${priorCount} précédentes`
    : `${signed} str/min vs prior ${priorCount}`
}

export const detectLocale = (): Locale => {
  try {
    return navigator.language.toLowerCase().startsWith('fr') ? 'fr' : 'en'
  } catch {
    return 'en'
  }
}

export const initTriLocale = (): void => {
  try {
    const saved = localStorage.getItem(TRI_LOCALE_KEY)
    locale = saved === 'fr' || saved === 'en' ? saved : detectLocale()
  } catch {
    locale = 'en'
  }
}

export const applyTriLocale = (next: Locale): void => {
  if (next === locale) return
  locale = next
  try {
    localStorage.setItem(TRI_LOCALE_KEY, next)
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new CustomEvent('tri:locale'))
}

const en: TriDict = {
  ui: {
    fitness: 'fitness',
    fatigue: 'fatigue',
    form: 'form',
    efficiency: 'efficiency',
    decoupling: 'decoupling',
    'hrv baseline': 'hrv baseline',
    monotony: 'monotony',
    strain: 'strain',
    'fitness age': 'fitness age',
    base: 'base',
    'ACSM estimate': 'ACSM estimate',
    rhr: 'rhr',
    sprint: 'sprint',
    threshold: 'threshold',
    endurance: 'endurance',
    climb: 'climb',
    'stride length': 'stride length',
    'estimated stride length': 'estimated stride length',
    cadence: 'cadence',
    'vertical oscillation': 'vertical oscillation',
    'stroke rate': 'stroke rate',
    tempo: 'tempo',
    anaerobic: 'anaerobic',
    VO2max: 'VO2max',
    neuromuscular: 'neuromuscular',
    'warm up': 'warm up',
    'fat burning': 'fat burning',
    vigorous: 'vigorous',
    maximal: 'maximal',
    water: 'water',
    peak: 'peak',
    latest: 'latest',
    lab: 'lab',
    goal: 'goal',
    fat: 'fat',
    debt: 'debt',
    bone: 'bone',
    muscle: 'muscle',
    bmi: 'bmi',
    baseline: 'baseline',
    ramp: 'ramp',
    'this wk': 'this wk',
    'active wk': 'active wk',
    avg: 'avg',
    'wtd avg': 'wtd avg',
    'training impulse': 'training impulse',
    'vs last': 'vs last',
    'training load · injury risk': 'training load · injury risk',
    'weekly load': 'weekly load',
    'race readiness': 'race readiness',
    'pace trend + forecast': 'pace trend + forecast',
    'things to improve': 'things to improve',
    'body weight': 'body weight',
    'relative effort': 'relative effort',
    'ambient heat · acclimatisation': 'ambient heat · acclimatisation',
    'no outdoor temperature data': 'no outdoor temperature data',
    'heat days': 'heat days',
    '14d': '14d',
    'activity temperature': 'activity temperature',
    'acclimatisation proxy': 'acclimatisation proxy',
    'heat exposure': 'heat exposure',
    'ambient workout temperature and heat acclimatisation proxy over time':
      'ambient workout temperature and heat acclimatisation proxy over time',
    'weather coverage': 'weather coverage',
    confidence: 'confidence',
    moderate: 'moderate',
    low: 'low',
    none: 'none',
    exposure: 'exposure',
    exposures: 'exposures',
    day: 'day',
    days: 'days',
    'decay after': 'decay after',
    'hot min': 'hot min',
    proxy: 'proxy',
    'recovery · hrv · rhr': 'recovery · hrv · rhr',
    'sleep · debt': 'sleep · debt',
    'body composition': 'body composition',
    'body composition by region': 'body composition by region',
    'lab test date': 'lab test date',
    wk: 'wk',
    BMR: 'BMR',
    FFM: 'FFM',
    essential: 'essential',
    athlete: 'athlete',
    obese: 'obese',
    Metabolic: 'Metabolic',
    Ventilation: 'Ventilation',
    Target: 'Target',
    Min: 'Min',
    Max: 'Max',
    Avg: 'Avg',
    HR: 'HR',
    'Warm-Up': 'Warm-Up',
    Test: 'Test',
    'Cool-Down': 'Cool-Down',
    'vo2max · fitness age': 'vo2max · fitness age',
    'vo2 test profile': 'vo2 test profile',
    'ftp hypothesis': 'ftp hypothesis',
    abilities: 'abilities',
    'cardiovascular health': 'cardiovascular health',
    'fitness · fatigue · form': 'fitness · fatigue · form',
    'form · ramp': 'form · ramp',
    'heart rate zones': 'heart rate zones',
    'power zones': 'power zones',
    '25W power distribution': '25W power distribution',
    'power curve': 'power curve',
    'this ride': 'this ride',
    '6-week best': '6-week best',
    'comparison range': 'comparison range',
    selection: 'selection',
    '6 weeks': '6 weeks',
    'all of': 'all of',
    lengths: 'lengths',
    '100 m': '100 m',
    'swim chart aggregation': 'swim chart aggregation',
    'swim activity analysis': 'swim activity analysis',
    'pace /100m': 'pace /100m',
    'stroke rate str/min': 'stroke rate str/min',
    speed: 'speed',
    pace: 'pace',
    power: 'power',
    'heart rate': 'heart rate',
    elevation: 'elevation',
    time: 'time',
    'avg hr': 'avg hr',
    'monotony / monotony —': 'monotony / monotony —',
    'strain / strain —': 'strain / strain —',
    'building base — ACWR needs ~4 weeks': 'building base — ACWR needs ~4 weeks',
    'not enough data': 'not enough data',
    today: 'today',
    'projected load': 'projected load',
    'assumed future daily load': 'assumed future daily load',
    'no activity': 'no activity',
    'no weeks': 'no weeks',
    'above range': 'above range',
    'in range': 'in range',
    'below range': 'below range',
    now: 'now',
    faster: 'faster',
    slower: 'slower',
    flat: 'flat',
    weakest: 'weakest',
    'no weight logged': 'no weight logged',
    'no effort logged': 'no effort logged',
    'no recovery data': 'no recovery data',
    'no sleep logged': 'no sleep logged',
    'no dexa scan logged': 'no dexa scan logged',
    '% fat': '% fat',
    lean: 'lean',
    arms: 'arms',
    legs: 'legs',
    trunk: 'trunk',
    bmd: 'bmd',
    'no power or hr data yet': 'no power or hr data yet',
    'This estimate comes from running VO2max.': 'This estimate comes from running VO2max.',
    'A lower resting heart rate is better.': 'A lower resting heart rate is better.',
    'The 7 day average is compared with the 28 day baseline.':
      'The 7 day average is compared with the 28 day baseline.',
    'This is pace or power per heartbeat.': 'This is pace or power per heartbeat.',
    'This needs at least 20 minutes with heart rate and pace or power data.':
      'This needs at least 20 minutes with heart rate and pace or power data.',
    'Under 5% means steady output.': 'Under 5% means steady output.',
    'From 5% to 10% means some late fade.': 'From 5% to 10% means some late fade.',
    'Over 10% means high late fade.': 'Over 10% means high late fade.',
    'no vo2 test logged': 'no vo2 test logged',
    'vt1 · aerobic threshold': 'vt1 · aerobic threshold',
    'no vo2-derived ftp estimate': 'no vo2-derived ftp estimate',
    'efficiency estimate': 'efficiency estimate',
    'total running vo2max': 'total running vo2max',
    'estimated cycling vo2max': 'estimated cycling vo2max',
    'vo2 used at threshold': 'vo2 used at threshold',
    'energy used per second': 'energy used per second',
    'maximum aerobic power': 'maximum aerobic power',
    'value from vo2 report': 'value from vo2 report',
    'running vo2max': 'running vo2max',
    'measured during treadmill test': 'measured during treadmill test',
    'running to cycling adjustment': 'running to cycling adjustment',
    'reduces running vo2max for cycling': 'reduces running vo2max for cycling',
    'vo2max used at threshold': 'vo2max used at threshold',
    'estimated because the treadmill test did not find the second threshold':
      'estimated because the treadmill test did not find the second threshold',
    'cycling efficiency': 'cycling efficiency',
    'share of energy turned into bike power': 'share of energy turned into bike power',
    reset: 'reset',
    'no heart data yet': 'no heart data yet',
    'map unavailable': 'map unavailable',
    'go back': 'back',
    'metrics & terms': 'metrics & terms',
    activities: 'activities',
    'filter activities': 'filter activities',
    'sort activities': 'sort activities',
    'sort by distance, cadence, pace': 'sort by distance, cadence, pace',
    'no matches': 'no matches',
    'filter routes': 'filter routes',
    'no routes': 'no routes',
    'loading…': 'loading…',
    'no plan': 'no plan',
    'no detail': 'no detail',
    'no activities': 'no activities',
    'no data': 'no data',
    'go to page · toggle units...': 'go to page · toggle units...',
    'command palette': 'command palette',
    command: 'command',
    'no commands': 'no commands',
    'imperial → metric': 'imperial → metric',
    'metric → imperial': 'metric → imperial',
    'distance · pace · weight · composition': 'distance · pace · weight · composition',
    'overview · bars': 'overview · bars',
    tools: 'tools',
    'gear · pace · fuel · calculator': 'gear · pace · fuel · calculator',
    analytics: 'analytics',
    'charts · search': 'charts · search',
    maps: 'maps',
    training: 'training',
    feed: 'feed',
    on: 'on',
    'all activities · list': 'all activities · list',
    'weight unit': 'weight unit',
    home: 'home',
    running: 'running',
    swim: 'swim',
    bike: 'bike',
    run: 'run',
    walk: 'walk',
    wearables: 'wearables',
    fuel: 'fuel',
    mandarins: 'mandarins',
    apple: 'apple',
    banana: 'banana',
    gear: 'gear',
    calculator: 'calculator',
    heat: 'heat',
    hr: 'hr',
    map: 'map',
    'triathlon calculator': 'triathlon calculator',
    average: 'average',
    projected: 'projected',
    projection: 'projection',
    'dashed line is projected from bike power': 'dashed line is projected from bike power',
    'vs current': 'vs current',
    finish: 'finish',
    'avg power': 'avg power',
    'est power': 'est power',
    'max power': 'max power',
    'max speed': 'max speed',
    energy: 'energy',
    'max hr': 'max hr',
    wind: 'wind',
    gust: 'gust',
    fueling: 'fueling',
    recovery: 'recovery',
    consumed: 'consumed',
    fluid: 'fluid',
    target: 'target',
    sweat: 'sweat',
    sleep: 'sleep',
    slept: 'slept',
    hrv: 'hrv',
    'resting hr': 'resting hr',
    'day burn': 'day burn',
    'day active': 'day active',
    rest: 'rest',
    strength: 'strength',
    freestyle: 'freestyle',
    breast: 'breast',
    back: 'back',
    fly: 'fly',
    mixed: 'mixed',
    kick: 'kick',
    race: 'race',
    'Loading PDF': 'Loading PDF',
    'fuel plan': 'fuel plan',
    'inspired by rauno': 'inspired by rauno',
    Close: 'Close',
    olympic: 'olympic',
    'Copy embed link': 'Copy embed link',
    copy: 'copy',
    copied: 'copied',
    'go to page · toggle units…': 'go to page · toggle units…',
    routes: 'routes',
    'sleep score': 'sleep score',
    readiness: 'readiness',
    'sleep stages': 'sleep stages',
    deep: 'deep',
    light: 'light',
    rem: 'rem',
    awake: 'awake',
    'no detail for this night': 'no detail for this night',
    'rock bottom — no sleep recorded': 'rock bottom — no sleep recorded',
    bedtime: 'bedtime',
    'wake-up': 'wake-up',
    latency: 'latency',
    'lowest hr': 'lowest hr',
    breath: 'breath',
    'resting heart rate': 'resting heart rate',
    'deep sleep': 'deep sleep',
    'rem sleep': 'rem sleep',
    restfulness: 'restfulness',
    timing: 'timing',
    'total sleep': 'total sleep',
    'activity balance': 'activity balance',
    'body temperature': 'body temperature',
    'hrv balance': 'hrv balance',
    'previous day activity': 'previous day activity',
    'previous night': 'previous night',
    'recovery index': 'recovery index',
    'sleep balance': 'sleep balance',
    'sleep regularity': 'sleep regularity',
    age: 'age',
    feet: 'feet',
    metres: 'metres',
    'projected finish range, including both transitions':
      'The projected finish has an 80% chance of falling in this range. It includes both transitions.',
    'custom date missing': 'custom date missing',
    'radar sprint bike definition':
      'Sprint uses your best 5 second bike power divided by body weight. A higher value means more power for your weight during a short effort.',
    'radar sprint run definition':
      'Sprint uses your fastest recorded 30 second running speed. It shows your top speed during a short effort.',
    'radar sprint swim definition':
      'Sprint uses the fastest average speed from one recorded swim. Pool data is recorded by length, so this page does not estimate a shorter peak from within the swim.',
    'radar threshold bike definition':
      'Threshold uses FTP divided by body weight. FTP estimates the bike power you can hold for about one hour.',
    'radar threshold run definition':
      'Threshold uses the fastest running speed you can hold for a sustained effort after adjusting for hills.',
    'radar threshold swim definition':
      'Threshold uses critical swim speed. It estimates the pace you can hold during a long, steady swim from your recorded sustained efforts.',
    'radar endurance definition':
      'Endurance uses the 42 day training load for this sport. The score compares that load with the target share of your total training.',
    'radar pace swim definition':
      'Pace uses the fastest valid average pace from one swim, based on active time. Fewer seconds per 100 metres give a higher score.',
    'radar climb run definition':
      'Climb uses the vertical {unit} gained per hour of running. It counts moving time.',
    'radar climb bike definition':
      'Climb uses the vertical {unit} gained per hour of cycling. It counts moving time.',
    'radar cadence bike definition':
      'Cadence compares your average pedal rate with 90 revolutions per minute. The score falls when your rate is above or below 90.',
    'radar cadence run definition':
      'Cadence compares your average step rate with 180 steps per minute. The score falls when your rate is above or below 180.',
    'radar stroke rate swim definition':
      'Stroke rate is the average from recorded swims that include both stroke counts and timing. Each rate is the stroke count divided by the time during which strokes were recorded. The target is 30 strokes per minute.',
    'radar recovery definition':
      'Recovery uses your average Oura Readiness score over the past 14 days. If Readiness is unavailable, it uses HRV.',
    'radar stride run definition':
      'Stride length is your 42 day average distance per running step. Native Apple Watch samples take priority. When those samples are unavailable, the value is estimated from speed and cadence. The score shows where the average falls within your recorded range.',
    'radar oscillation run definition':
      'Vertical oscillation is your 42 day average vertical movement per running step from Apple Watch. The score is inverted within your recorded range, so less movement plots farther from the centre. Pace, terrain, height, and running style affect the measurement.',
    'radar unit wkg definition':
      '$\\mathrm{W/kg}$ means watts per kilogram of body weight. A rider producing 270 W at 90 kg has 3.0 W/kg.',
    'radar unit ctl definition':
      'CTL is your average daily training load over 42 days. Recent days count more.',
    'radar unit fth definition':
      '$\\mathrm{ft/h}$ means vertical feet climbed per hour. This page converts metres to feet using $1\\,\\mathrm{m}=3.281\\,\\mathrm{ft}$.',
    'radar unit mh definition':
      '$\\mathrm{m/h}$ means vertical metres climbed per hour. This page divides elevation gain by moving uphill time and scales it to one hour.',
    'radar unit mspeed definition':
      '$\\mathrm{m/s}$ means metres travelled per second. Multiply the value by 3.6 to convert it to kilometres per hour.',
    'radar unit s100m definition':
      's/100 m means the seconds needed to swim 100 metres. A smaller value means a faster pace.',
    'radar unit rpm definition':
      'rpm means revolutions per minute. It measures how fast you turn the pedals.',
    'radar unit spm definition': 'spm means steps per minute. It measures your running cadence.',
    'radar unit strmin definition':
      'str/min means strokes per minute. The rate is the stroke count divided by the time during which strokes were recorded.',
    'radar unit readiness definition':
      "Readiness is Oura's daily recovery score from 0 to 100. It uses your sleep and HRV. It also uses resting heart rate and recent activity.",
    'radar unit ms definition':
      'ms means milliseconds. HRV measures the change in time between heartbeats. A value above your usual range can mean better recovery.',
    'radar unit stride definition':
      'Stride length is stored in metres per step. Imperial mode converts it to feet using 1 m = 3.281 ft.',
    'radar unit oscillation definition':
      'Vertical oscillation is stored in centimetres. Imperial mode converts it to inches using 1 in = 2.54 cm.',
    'radar unit default definition':
      'The raw value is the original measurement used to calculate this score from 0 to 100.',
  },
  gloss: {
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
      def: 'Threshold pace is your estimated pace for an effort lasting about one hour. It uses your faster sessions and adjusts running pace for hills. The pace model uses it as a starting point.',
    },
    trend: {
      term: 'pace trend',
      def: 'Pace trend shows whether your threshold pace is getting faster or slower. It uses your recent sessions. The shaded area shows the range of likely future values.',
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
      def: 'Ambient temperature is the duration-weighted WeatherKit estimate during the activity at the route centre. Strava device temperature fills gaps. It is separate from Oura skin-temperature deviation.',
    },
    heatdose: {
      term: 'heat exposure dose',
      def: 'A GPS run or ride above 22 °C contributes up to one daily dose. Sixty hot minutes equal one dose. The proxy targets 14 doses, holds for three days without heat, then decays by 2.5% per day.',
    },
    heatacclimation: {
      term: 'heat acclimatisation proxy',
      def: 'This percentage tracks recent outdoor heat exposure. It does not measure physiological acclimatisation because core temperature, humidity, solar load, clothing, sweat response, hydration, and passive heat exposure are unavailable.',
    },
    sleepdebt: {
      term: 'sleep debt',
      def: 'Sleep debt adds up the time you slept below 7 hours during the past 14 nights. This chart uses 7 hours as its target. Athletes often need 8 to 10 hours.',
    },
    overreaching: {
      term: 'overreaching',
      def: 'Overreaching is flagged when HRV is low while training load rises quickly. Here, low HRV means at least one standard deviation below your baseline. A fast load increase means a high ACWR or a weekly ramp above 10%.',
    },
    oreadiness: {
      term: 'Oura Readiness',
      def: 'Oura Readiness is a daily score from 0 to 100. A score of 85 or higher is optimal. A score from 70 to 84 is good. A score below 70 calls for attention. Several days below 70 can point to accumulated strain.',
    },
    vo2max: {
      term: 'VO₂max',
      def: 'VO₂max is the maximum amount of oxygen your body can use during hard exercise. It is measured in millilitres per kilogram per minute. The cycling estimate starts with FTP. This page treats FTP as 95% of your best 20 minute power, then estimates maximum aerobic power and VO₂max.',
    },
    ftp: {
      term: 'FTP hypothesis',
      def: 'FTP is the cycling power you may be able to hold for about one hour. This estimate comes from a treadmill VO₂max test, so it crosses from running to cycling and has low confidence. Use your heart rate zones until you complete a bike power test.',
    },
    fitage: {
      term: 'fitness age',
      def: 'Fitness age is the age with the same median VO₂max in the male FRIEND reference data. This chart limits the result to ages 20 to 80. A lower fitness age means your VO₂max is above the median for your calendar age.',
    },
    vam: {
      term: 'VAM',
      def: 'VAM is the vertical distance you climb in one hour. This page calculates it from elevation gain and moving time. It shows metres per hour or feet per hour based on your unit setting.',
    },
    radar: {
      term: 'abilities',
      def: 'Each sport has six scores from 0 to 100. Sprint and threshold use power or speed. Endurance uses 42 days of training load. Run uses stride length, cadence, and vertical oscillation. Bike uses climbing rate and cadence. Swim uses pace and stroke rate. Bike and swim retain recovery. The dashed line shows the projected score 28 days from now.',
    },
    ef: {
      term: 'efficiency factor',
      def: 'Efficiency factor compares your pace or power with your heart rate. On the bike, it divides normalized power by average heart rate. Normalized power gives more weight to hard efforts. On the run, the calculation divides speed adjusted for hills by average heart rate. A higher value at the same effort means you are producing more output per heartbeat.',
    },
    decouple: {
      term: 'decoupling',
      def: 'Decoupling compares the pace or power you produced per heartbeat in the first and second half of a workout. A lower value is better. A value under 5% means you stayed steady. A value over 10% means you faded in the second half.',
    },
    legdist: {
      term: 'total distance',
      def: 'Total distance is the distance covered in this sport during the selected season. Swim distance is shown in metres. Bike and run distance use the kilometre or mile setting.',
    },
    legcount: {
      term: 'sessions',
      def: 'This is the number of workouts logged for this sport during the selected season.',
    },
    legtime: { term: 'total time', def: 'Total time adds the moving time from these workouts.' },
    herodist: {
      term: 'season total',
      def: 'Season total adds the distance from all three sports during the selected season. It uses the kilometre or mile setting.',
    },
  },
}

const fr: TriDict = {
  ui: {
    fitness: 'condition',
    fatigue: 'fatigue',
    form: 'forme',
    efficiency: 'efficience',
    decoupling: 'découplage',
    'hrv baseline': 'référence vfc',
    monotony: 'monotonie',
    strain: 'contrainte',
    'fitness age': 'âge de forme',
    base: 'base',
    'ACSM estimate': 'estimation ACSM',
    rhr: 'fc',
    sprint: 'sprint',
    threshold: 'seuil',
    endurance: 'endurance',
    climb: 'grimpe',
    'stride length': 'longueur de foulée',
    'estimated stride length': 'longueur de foulée estimée',
    cadence: 'cadence',
    'vertical oscillation': 'oscillation verticale',
    'stroke rate': 'fréquence de nage',
    tempo: 'tempo',
    anaerobic: 'anaérobie',
    VO2max: 'VO2max',
    neuromuscular: 'neuromusculaire',
    'warm up': 'échauffement',
    'fat burning': 'combustion graisses',
    vigorous: 'intense',
    maximal: 'maximal',
    water: 'eau',
    peak: 'pic',
    latest: 'dernier',
    lab: 'labo',
    goal: 'objectif',
    fat: 'gras',
    debt: 'dette',
    bone: 'os',
    muscle: 'muscle',
    bmi: 'IMC',
    baseline: 'référence',
    ramp: 'progression',
    'this wk': 'cette sem',
    'active wk': 'sem actives',
    avg: 'moy',
    'wtd avg': 'moy pond',
    'training impulse': 'charge séance',
    'vs last': 'vs préc',
    'training load · injury risk': 'charge · risque de blessure',
    'weekly load': 'charge hebdo',
    'race readiness': 'préparation course',
    'pace trend + forecast': 'tendance allure + prévision',
    'things to improve': 'à améliorer',
    'body weight': 'poids',
    'relative effort': 'effort relatif',
    'ambient heat · acclimatisation': 'chaleur ambiante · acclimatation',
    'no outdoor temperature data': 'aucune température extérieure',
    'heat days': 'jours de chaleur',
    '14d': '14 j',
    'activity temperature': 'température par activité',
    'acclimatisation proxy': "indice d'acclimatation",
    'heat exposure': 'exposition à la chaleur',
    'ambient workout temperature and heat acclimatisation proxy over time':
      "température ambiante des séances et indice d'acclimatation au fil du temps",
    'weather coverage': 'couverture météo',
    confidence: 'fiabilité',
    moderate: 'modérée',
    low: 'faible',
    none: 'aucune',
    exposure: 'exposition',
    exposures: 'expositions',
    day: 'jour',
    days: 'jours',
    'decay after': 'baisse après',
    'hot min': 'min chaudes',
    proxy: 'indice',
    'recovery · hrv · rhr': 'récupération · vfc · fc',
    'sleep · debt': 'sommeil · dette',
    'body composition': 'composition corporelle',
    'body composition by region': 'composition corporelle par région',
    'lab test date': 'date des tests de laboratoire',
    wk: 'sem',
    BMR: 'MB',
    FFM: 'MM',
    essential: 'essentiel',
    athlete: 'athlète',
    obese: 'obésité',
    Metabolic: 'Métabolique',
    Ventilation: 'Ventilation',
    Target: 'Objectif',
    Min: 'Min',
    Max: 'Max',
    Avg: 'Moy',
    HR: 'FC',
    'Warm-Up': 'Échauffement',
    Test: 'Test',
    'Cool-Down': 'Retour au calme',
    'vo2max · fitness age': 'vo2max · âge de forme',
    'vo2 test profile': 'profil test vo2',
    'ftp hypothesis': 'hypothèse ftp',
    abilities: 'aptitudes',
    'cardiovascular health': 'santé cardiovasculaire',
    'fitness · fatigue · form': 'condition · fatigue · forme',
    'form · ramp': 'forme · ramp',
    'heart rate zones': 'zones de fc',
    'power zones': 'zones de puissance',
    '25W power distribution': 'répartition puissance 25W',
    'power curve': 'courbe de puissance',
    'this ride': 'cette sortie',
    '6-week best': 'meilleur sur 6 semaines',
    'comparison range': 'période de comparaison',
    selection: 'sélection',
    '6 weeks': '6 semaines',
    'all of': "toute l'année",
    lengths: 'longueurs',
    '100 m': '100 m',
    'swim chart aggregation': 'agrégation des graphiques de natation',
    'swim activity analysis': "analyse de l'activité de natation",
    'pace /100m': 'allure /100 m',
    'stroke rate str/min': 'fréquence de nage coups/min',
    speed: 'vitesse',
    pace: 'allure',
    power: 'puissance',
    'heart rate': 'fc',
    elevation: 'altitude',
    time: 'temps',
    'avg hr': 'fc moy',
    'monotony / monotony —': 'monotonie / monotonie —',
    'strain / strain —': 'contrainte / contrainte —',
    'building base — ACWR needs ~4 weeks': "constitution de la base — l'ACWR nécessite ~4 semaines",
    'not enough data': 'données insuffisantes',
    today: 'auj.',
    'projected load': 'charge projetée',
    'assumed future daily load': 'charge quotidienne future supposée',
    'no activity': 'aucune activité',
    'no weeks': 'aucune semaine',
    'above range': 'au-dessus de la plage',
    'in range': 'dans la plage',
    'below range': 'sous la plage',
    now: 'maint.',
    faster: 'plus rapide',
    slower: 'plus lent',
    flat: 'stable',
    weakest: 'point faible',
    'no weight logged': 'aucun poids',
    'no effort logged': 'aucun effort',
    'no recovery data': 'aucune donnée récup.',
    'no sleep logged': 'aucun sommeil',
    'no dexa scan logged': 'aucun scan dexa',
    '% fat': '% gras',
    lean: 'maigre',
    arms: 'bras',
    legs: 'jambes',
    trunk: 'tronc',
    bmd: 'dmo',
    'no power or hr data yet': 'pas encore de données puissance ou fc',
    'This estimate comes from running VO2max.': 'Cette estimation vient de la VO₂max en course.',
    'A lower resting heart rate is better.':
      'Une fréquence cardiaque au repos plus basse est meilleure.',
    'The 7 day average is compared with the 28 day baseline.':
      'La moyenne sur 7 jours est comparée à la référence sur 28 jours.',
    'This is pace or power per heartbeat.':
      "Il s'agit de l'allure ou de la puissance par battement.",
    'This needs at least 20 minutes with heart rate and pace or power data.':
      "Il faut au moins 20 minutes avec la fréquence cardiaque et l'allure ou la puissance.",
    'Under 5% means steady output.': 'Une valeur sous 5 % signifie un effort stable.',
    'From 5% to 10% means some late fade.':
      'Une valeur de 5 % à 10 % signifie une légère baisse en fin de séance.',
    'Over 10% means high late fade.':
      'Une valeur au-dessus de 10 % signifie une forte baisse en fin de séance.',
    'no vo2 test logged': 'aucun test vo2',
    'vt1 · aerobic threshold': 'vt1 · seuil aérobie',
    'no vo2-derived ftp estimate': "pas d'estimation ftp via vo2",
    'efficiency estimate': "estimation par l'efficacité",
    'total running vo2max': 'vo2max totale en course',
    'estimated cycling vo2max': 'vo2max estimée à vélo',
    'vo2 used at threshold': 'vo2 utilisée au seuil',
    'energy used per second': 'énergie utilisée par seconde',
    'maximum aerobic power': 'puissance aérobie maximale',
    'value from vo2 report': 'valeur du rapport vo2',
    'running vo2max': 'vo2max course',
    'measured during treadmill test': 'mesurée pendant le test sur tapis',
    'running to cycling adjustment': 'ajustement de la course au vélo',
    'reduces running vo2max for cycling': 'réduit la vo2max en course pour le vélo',
    'vo2max used at threshold': 'vo2max utilisée au seuil',
    'estimated because the treadmill test did not find the second threshold':
      "estimée car le test sur tapis n'a pas trouvé le second seuil",
    'cycling efficiency': 'efficacité à vélo',
    'share of energy turned into bike power': "part de l'énergie transformée en puissance à vélo",
    reset: 'réinit.',
    'no heart data yet': 'pas encore de données cardiaques',
    'map unavailable': 'carte indisponible',
    'go back': 'retour',
    'metrics & terms': 'mesures et termes',
    activities: 'activités',
    'filter activities': 'filtrer les activités',
    'sort activities': 'trier les activités',
    'sort by distance, cadence, pace': 'trier par distance, cadence, pace',
    'no matches': 'aucun résultat',
    'filter routes': 'filtrer les parcours',
    'no routes': 'aucun parcours',
    'loading…': 'chargement…',
    'no plan': 'aucun plan',
    'no detail': 'aucun détail',
    'no activities': 'aucune activité',
    'no data': 'aucune donnée',
    'go to page · toggle units...': "aller à une page · changer d'unités...",
    'command palette': 'palette de commandes',
    command: 'commande',
    'no commands': 'aucune commande',
    'imperial → metric': 'impérial → métrique',
    'metric → imperial': 'métrique → impérial',
    'distance · pace · weight · composition': 'distance · allure · poids · composition',
    'overview · bars': "vue d'ensemble · barres",
    tools: 'outils',
    'gear · pace · fuel · calculator': 'matériel · allure · nutrition · calculateur',
    analytics: 'analyses',
    'charts · search': 'graphiques · recherche',
    maps: 'cartes',
    training: 'entraînement',
    feed: 'flux',
    on: 'journal',
    'all activities · list': 'toutes les activités · liste',
    'weight unit': 'unité de poids',
    home: 'accueil',
    running: 'course à pied',
    swim: 'natation',
    bike: 'vélo',
    run: 'course',
    walk: 'marche',
    wearables: 'capteurs',
    fuel: 'nutrition',
    mandarins: 'mandarines',
    apple: 'pomme',
    banana: 'banane',
    gear: 'matériel',
    calculator: 'calculateur',
    heat: 'densité',
    hr: 'fc',
    map: 'carte',
    'triathlon calculator': 'calculateur triathlon',
    average: 'moyenne',
    projected: 'projetée',
    projection: 'projection',
    'dashed line is projected from bike power':
      'la ligne pointillée est une projection basée sur la puissance à vélo',
    'vs current': 'vs actuel',
    finish: 'arrivée',
    'avg power': 'puiss moy',
    'est power': 'puiss est',
    'max power': 'puiss max',
    'max speed': 'vitesse max',
    energy: 'énergie',
    'max hr': 'fc max',
    wind: 'vent',
    gust: 'rafale',
    fueling: 'nutrition',
    recovery: 'récupération',
    consumed: 'ingéré',
    fluid: 'hydratation',
    target: 'objectif',
    sweat: 'sudation',
    sleep: 'sommeil',
    slept: 'dormi',
    hrv: 'vfc',
    'resting hr': 'fc repos',
    'day burn': 'dépense jour',
    'day active': 'actif jour',
    rest: 'repos',
    strength: 'renforcement',
    freestyle: 'crawl',
    breast: 'brasse',
    back: 'dos',
    fly: 'papillon',
    mixed: 'mixte',
    kick: 'jambes',
    race: 'course',
    'Loading PDF': 'Chargement du PDF',
    'fuel plan': 'plan nutrition',
    'inspired by rauno': 'inspiré de rauno',
    Close: 'Fermer',
    olympic: 'olympique',
    'Copy embed link': "copier le lien d'intégration",
    copy: 'copier',
    copied: 'copié',
    'go to page · toggle units…': "aller à une page · changer d'unités…",
    routes: 'parcours',
    'sleep score': 'score de sommeil',
    readiness: 'préparation',
    'sleep stages': 'phases de sommeil',
    deep: 'profond',
    light: 'léger',
    rem: 'paradoxal',
    awake: 'éveil',
    'no detail for this night': 'aucun détail pour cette nuit',
    'rock bottom — no sleep recorded': 'nuit blanche — aucun sommeil enregistré',
    bedtime: 'coucher',
    'wake-up': 'réveil',
    latency: 'latence',
    'lowest hr': 'fc min',
    breath: 'respiration',
    'resting heart rate': 'fréquence cardiaque au repos',
    'deep sleep': 'sommeil profond',
    'rem sleep': 'sommeil paradoxal',
    restfulness: 'tranquillité',
    timing: 'horaire',
    'total sleep': 'sommeil total',
    'activity balance': "équilibre d'activité",
    'body temperature': 'température corporelle',
    'hrv balance': 'équilibre vfc',
    'previous day activity': 'activité de la veille',
    'previous night': 'nuit précédente',
    'recovery index': 'indice de récupération',
    'sleep balance': 'équilibre de sommeil',
    'sleep regularity': 'régularité du sommeil',
    age: 'âge',
    feet: 'pieds',
    metres: 'mètres',
    'projected finish range, including both transitions':
      "Le temps d'arrivée a 80 % de chances de se trouver dans cette plage. Les deux transitions sont incluses.",
    'custom date missing': 'date perso manquante',
    'radar sprint bike definition':
      'Le sprint utilise ta meilleure puissance à vélo sur 5 secondes, divisée par ton poids. Une valeur plus haute signifie plus de puissance par kilogramme pendant un effort court.',
    'radar sprint run definition':
      'Le sprint utilise ta vitesse de course la plus rapide sur 30 secondes. Il montre ta vitesse maximale pendant un effort court.',
    'radar sprint swim definition':
      "Le sprint utilise la vitesse moyenne la plus rapide d'une séance de natation. Les données de piscine sont enregistrées par longueur, donc cette page n'estime pas un pic plus court au sein de la séance.",
    'radar threshold bike definition':
      'Le seuil utilise la FTP divisée par ton poids. La FTP estime la puissance à vélo que tu peux tenir pendant environ une heure.',
    'radar threshold run definition':
      'Le seuil utilise la vitesse de course la plus rapide que tu peux tenir pendant un effort soutenu, après un ajustement pour les côtes.',
    'radar threshold swim definition':
      "Le seuil utilise la vitesse critique de natation. Elle estime l'allure que tu peux tenir pendant une longue nage régulière à partir de tes efforts soutenus.",
    'radar endurance definition':
      "L'endurance utilise la charge d'entraînement de ce sport sur 42 jours. La note compare cette charge avec la part cible de ton entraînement total.",
    'radar pace swim definition':
      "L'allure utilise la meilleure allure moyenne valide d'une séance de natation, selon le temps actif. Moins de secondes par 100 mètres donne une note plus haute.",
    'radar climb run definition':
      'La grimpe utilise les {unit} de dénivelé gagnés par heure de course. Elle compte le temps en mouvement.',
    'radar climb bike definition':
      'La grimpe utilise les {unit} de dénivelé gagnés par heure de vélo. Elle compte le temps en mouvement.',
    'radar cadence bike definition':
      'La cadence compare ta fréquence moyenne de pédalage avec 90 tours par minute. La note baisse lorsque ta fréquence est au-dessus ou en dessous de 90.',
    'radar cadence run definition':
      'La cadence compare ta fréquence moyenne de pas avec 180 pas par minute. La note baisse lorsque ta fréquence est au-dessus ou en dessous de 180.',
    'radar stroke rate swim definition':
      'La fréquence de nage est la moyenne des fréquences calculées pour les séances qui contiennent un nombre de coups et une durée. Chaque fréquence divise le nombre de coups par le temps pendant lequel ils ont été enregistrés. La cible est de 30 coups par minute.',
    'radar recovery definition':
      'La récupération utilise ton score Oura moyen des 14 derniers jours. Si le score Oura manque, elle utilise la VFC.',
    'radar stride run definition':
      "La longueur de foulée est ta distance moyenne par pas sur 42 jours. Les mesures natives de l'Apple Watch sont prioritaires. Lorsqu'elles manquent, la valeur est estimée avec la vitesse et la cadence. La note indique où cette moyenne se situe dans ta plage enregistrée.",
    'radar oscillation run definition':
      "L'oscillation verticale est ton mouvement vertical moyen par pas sur 42 jours, mesuré par l'Apple Watch. La note est inversée dans ta plage enregistrée, donc moins de mouvement s'affiche plus loin du centre. L'allure, le terrain, la taille et le style de course influencent la mesure.",
    'radar unit wkg definition':
      '$\\mathrm{W/kg}$ signifie watts par kilogramme de poids. Un cycliste qui produit 270 W à 90 kg a 3,0 W/kg.',
    'radar unit ctl definition':
      "La CTL est ta charge d'entraînement quotidienne moyenne sur 42 jours. Les jours récents comptent davantage.",
    'radar unit fth definition':
      '$\\mathrm{ft/h}$ signifie pieds de dénivelé par heure. Cette page convertit les mètres en pieds avec $1\\,\\mathrm{m}=3.281\\,\\mathrm{ft}$.',
    'radar unit mh definition':
      '$\\mathrm{m/h}$ signifie mètres de dénivelé par heure. Cette page divise le dénivelé par le temps de montée et ramène le résultat à une heure.',
    'radar unit mspeed definition':
      '$\\mathrm{m/s}$ signifie mètres parcourus par seconde. Multiplie la valeur par 3,6 pour la convertir en kilomètres par heure.',
    'radar unit s100m definition':
      's/100 m signifie le nombre de secondes nécessaires pour nager 100 mètres. Une valeur plus basse signifie une allure plus rapide.',
    'radar unit rpm definition':
      'rpm signifie tours par minute. Cette unité mesure la vitesse de pédalage.',
    'radar unit spm definition':
      'spm signifie pas par minute. Cette unité mesure ta cadence de course.',
    'radar unit strmin definition':
      'str/min signifie coups par minute. La fréquence divise le nombre de coups par le temps pendant lequel ils ont été enregistrés.',
    'radar unit readiness definition':
      "La préparation est le score quotidien de récupération d'Oura, de 0 à 100. Elle utilise ton sommeil et ta VFC. Elle utilise aussi ta fréquence cardiaque au repos et ton activité récente.",
    'radar unit ms definition':
      'ms signifie millisecondes. La VFC mesure la variation du temps entre les battements. Une valeur au-dessus de ta plage habituelle peut indiquer une meilleure récupération.',
    'radar unit stride definition':
      'La longueur de foulée est stockée en mètres par pas. Le mode impérial la convertit en pieds avec 1 m = 3,281 ft.',
    'radar unit oscillation definition':
      "L'oscillation verticale est stockée en centimètres. Le mode impérial la convertit en pouces avec 1 in = 2,54 cm.",
    'radar unit default definition':
      "La valeur brute est la mesure d'origine utilisée pour calculer cette note de 0 à 100.",
  },
  gloss: {
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
      def: "L'allure seuil estime ton allure pour un effort d'environ une heure. Elle utilise tes séances les plus rapides et ajuste l'allure de course pour les côtes. Le modèle d'allure l'utilise comme point de départ.",
    },
    trend: {
      term: "tendance d'allure",
      def: 'La tendance indique si ton allure seuil devient plus rapide ou plus lente. Elle utilise tes séances récentes. La zone ombrée montre la plage des valeurs futures probables.',
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
      def: "La température ambiante est l'estimation WeatherKit pondérée par la durée de la séance au centre du parcours. La température de l'appareil Strava remplit les données manquantes. Elle est distincte de l'écart de température cutanée Oura.",
    },
    heatdose: {
      term: "dose d'exposition à la chaleur",
      def: "Une course ou une sortie à vélo avec GPS au-dessus de 22 °C contribue au plus une dose quotidienne. Soixante minutes chaudes représentent une dose. L'indice vise 14 doses, reste stable pendant trois jours sans chaleur, puis baisse de 2,5 % par jour.",
    },
    heatacclimation: {
      term: "indice d'acclimatation à la chaleur",
      def: "Ce pourcentage suit l'exposition récente à la chaleur extérieure. Il ne mesure pas l'acclimatation physiologique, car la température corporelle, l'humidité, le rayonnement solaire, les vêtements, la transpiration, l'hydratation et l'exposition passive ne sont pas disponibles.",
    },
    sleepdebt: {
      term: 'dette de sommeil',
      def: 'La dette de sommeil additionne le temps dormi sous 7 heures pendant les 14 dernières nuits. Ce graphique utilise une cible de 7 heures. Les athlètes ont souvent besoin de 8 à 10 heures.',
    },
    overreaching: {
      term: 'surmenage',
      def: 'Le surmenage est signalé lorsque la VFC est basse pendant que la charge augmente rapidement. Ici, une VFC basse signifie au moins un écart type sous ta référence. Une hausse rapide signifie un ACWR élevé ou une progression hebdomadaire au-dessus de 10 %.',
    },
    oreadiness: {
      term: 'préparation Oura',
      def: 'La préparation Oura est une note quotidienne de 0 à 100. Une note de 85 ou plus est optimale. Une note de 70 à 84 est bonne. Une note sous 70 demande ton attention. Plusieurs jours sous 70 peuvent signaler une contrainte accumulée.',
    },
    vo2max: {
      term: 'VO₂max',
      def: "La VO₂max est la quantité maximale d'oxygène que ton corps peut utiliser pendant un effort intense. Elle est mesurée en millilitres par kilogramme par minute. L'estimation à vélo part de la FTP. Cette page estime la FTP à 95 % de ta meilleure puissance sur 20 minutes, puis calcule la puissance aérobie maximale et la VO₂max.",
    },
    ftp: {
      term: 'hypothèse FTP',
      def: "La FTP est la puissance à vélo que tu pourrais tenir pendant environ une heure. Cette estimation vient d'un test de VO₂max sur tapis, donc elle passe de la course au vélo et reste peu fiable. Utilise tes zones de fréquence cardiaque jusqu'à ton prochain test de puissance à vélo.",
    },
    fitage: {
      term: 'âge physiologique',
      def: "L'âge physiologique est l'âge qui a la même VO₂max médiane dans les données de référence FRIEND pour les hommes. Ce graphique limite le résultat de 20 à 80 ans. Un âge physiologique plus bas signifie que ta VO₂max dépasse la médiane de ton âge réel.",
    },
    vam: {
      term: 'VAM',
      def: "La VAM est le dénivelé grimpé en une heure. Cette page la calcule avec le dénivelé et le temps en mouvement. Elle affiche des mètres par heure ou des pieds par heure selon ton choix d'unités.",
    },
    radar: {
      term: 'aptitudes',
      def: "Chaque sport a six notes de 0 à 100. Le sprint et le seuil utilisent la puissance ou la vitesse. L'endurance utilise 42 jours de charge d'entraînement. La course utilise la longueur de foulée, la cadence et l'oscillation verticale. Le vélo utilise la vitesse de grimpe et la cadence. La natation utilise l'allure et la fréquence de nage. Le vélo et la natation conservent la récupération. La ligne pointillée montre la note prévue dans 28 jours.",
    },
    ef: {
      term: "facteur d'efficacité",
      def: "Le facteur d'efficacité compare ton allure ou ta puissance avec ta fréquence cardiaque. À vélo, il divise la puissance normalisée par la fréquence cardiaque moyenne. La puissance normalisée donne plus de poids aux efforts intenses. En course, le calcul divise la vitesse ajustée pour les côtes par la fréquence cardiaque moyenne. Une valeur plus haute au même effort signifie plus de rendement par battement.",
    },
    decouple: {
      term: 'découplage',
      def: "Le découplage compare l'allure ou la puissance produite par battement entre les deux moitiés de la séance. Une valeur plus basse est meilleure. Une valeur sous 5 % signifie que l'effort est resté stable. Une valeur au-dessus de 10 % signifie une baisse pendant la seconde moitié.",
    },
    legdist: {
      term: 'distance totale',
      def: 'La distance totale est la distance parcourue dans ce sport pendant la saison choisie. La natation est affichée en mètres. Le vélo et la course utilisent le réglage en kilomètres ou en milles.',
    },
    legcount: {
      term: 'séances',
      def: 'Ce nombre est le total des séances enregistrées pour ce sport pendant la saison choisie.',
    },
    legtime: {
      term: 'temps total',
      def: 'Le temps total additionne le temps en mouvement de ces séances.',
    },
    herodist: {
      term: 'total saison',
      def: 'Le total saison additionne la distance des trois sports pendant la saison choisie. Il utilise le réglage en kilomètres ou en milles.',
    },
  },
}

const TRI_I18N: Record<Locale, TriDict> = { en, fr }

export const tl = (s: string): string => TRI_I18N[locale].ui[s] ?? s

type Vo2SourceMethod = 'garmin' | 'apple' | 'bike' | 'run' | 'hrratio' | 'lab' | 'none'

type Vo2BikeSourceText = {
  ftpW: number
  ftpSource: 'athlete' | 'strava' | 'derived'
  mapW: number
  weightKg: number
}

export const vo2SourceText = (method: Vo2SourceMethod, bike: Vo2BikeSourceText | null): string => {
  if (method === 'garmin')
    return locale === 'fr'
      ? "Cette valeur vient de Garmin Connect ou d'une saisie manuelle."
      : 'This value comes from Garmin Connect or a manual entry.'
  if (method === 'apple')
    return locale === 'fr'
      ? "Cette mesure vient de l'Apple Watch."
      : 'This is an Apple Watch measurement.'
  if (method === 'run')
    return locale === 'fr'
      ? 'Cette estimation utilise la vitesse de course et la fréquence cardiaque.'
      : 'This estimate uses running speed and heart rate.'
  if (method === 'hrratio')
    return locale === 'fr'
      ? 'Cette estimation utilise les fréquences cardiaques maximale et au repos.'
      : 'This estimate uses maximum and resting heart rate.'
  if (method === 'lab')
    return locale === 'fr'
      ? "Cette valeur vient d'un test d'effort progressif."
      : 'This value comes from a graded exercise test.'
  if (method === 'bike' && bike != null) {
    const weight = bike.weightKg.toLocaleString(locale === 'fr' ? 'fr-CA' : 'en-US', {
      maximumFractionDigits: 1,
    })
    const source =
      bike.ftpSource === 'athlete'
        ? locale === 'fr'
          ? 'athlète'
          : 'athlete'
        : bike.ftpSource === 'strava'
          ? 'Strava'
          : locale === 'fr'
            ? 'estimée'
            : 'estimated'
    return locale === 'fr'
      ? `FTP ${bike.ftpW} W (${source}). La puissance aérobie maximale estimée est de ${bike.mapW} W. Le poids est de ${weight} kg.`
      : `FTP ${bike.ftpW} W (${source}). Estimated maximum aerobic power is ${bike.mapW} W. Body weight is ${weight} kg.`
  }
  return locale === 'fr'
    ? 'Il manque les données de puissance ou de fréquence cardiaque.'
    : 'There is no power or heart rate data.'
}

export const trendUnavailableText = (
  sampleSize: number | null,
  daysSinceLastEffort: number | null,
): string => {
  if (sampleSize === 0)
    return locale === 'fr' ? "Aucun effort n'a été enregistré." : 'No efforts were recorded.'
  if (daysSinceLastEffort === 0)
    return locale === 'fr'
      ? "Le dernier effort date d'aujourd'hui."
      : 'The latest effort was today.'
  if (daysSinceLastEffort != null) {
    const unit =
      locale === 'fr'
        ? daysSinceLastEffort === 1
          ? 'jour'
          : 'jours'
        : daysSinceLastEffort === 1
          ? 'day'
          : 'days'
    return locale === 'fr'
      ? `Le dernier effort remonte à ${daysSinceLastEffort} ${unit}.`
      : `The latest effort was ${daysSinceLastEffort} ${unit} ago.`
  }
  return locale === 'fr' ? 'Données insuffisantes.' : 'Not enough data.'
}

export const powerCurveReferenceLabel = (year: number | null): string =>
  year == null ? tl('6-week best') : locale === 'fr' ? `meilleur de ${year}` : `${year} best`

export const glossFor = (key: string): Gloss | undefined =>
  TRI_I18N[locale].gloss[key] ?? en.gloss[key]

export const glossKeys = (): string[] => Object.keys(en.gloss)
