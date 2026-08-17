export interface FtpHypothesisParams {
  crossModalDiscountPct: number
  thresholdPct: number
  grossEfficiencyPct: number
}

export interface FtpHypothesisCalculation extends FtpHypothesisParams {
  vo2max: number
  massKg: number
  absoluteVo2: number
  cyclingVo2max: number
  thresholdVo2: number
  metabolicWatts: number
  efficiencyFtp: number
  acsmMapWatts: number
  acsmFtp: number
  ftp: number
  low: number
  high: number
  wattsPerKg: number
}

export const FTP_HYPOTHESIS_DEFAULTS: FtpHypothesisParams = {
  crossModalDiscountPct: 8,
  thresholdPct: 85,
  grossEfficiencyPct: 21,
}

const ACSM_BASE = 7
const MAP_FTP_RATIO = 0.75
const VO2_KJ_PER_L = 20.9
const ACSM_KGM_PER_WATT = 6.12
const ACSM_VO2_PER_KGM = 1.8
const ERROR_W = 25

const round = (value: number, digits: number): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

const round5 = (value: number): number => Math.round(value / 5) * 5
const round10 = (value: number): number => Math.round(value / 10) * 10

export const calculateFtpHypothesis = (
  vo2max: number,
  massKg: number,
  params: FtpHypothesisParams = FTP_HYPOTHESIS_DEFAULTS,
): FtpHypothesisCalculation | null => {
  if (!(vo2max > 0) || !(massKg > 0)) return null
  const discount = params.crossModalDiscountPct / 100
  const threshold = params.thresholdPct / 100
  const efficiency = params.grossEfficiencyPct / 100
  const absoluteVo2 = (vo2max * massKg) / 1000
  const cyclingVo2max = absoluteVo2 * (1 - discount)
  const thresholdVo2 = cyclingVo2max * threshold
  const metabolicWatts = (thresholdVo2 * VO2_KJ_PER_L * 1000) / 60
  const efficiencyFtp = metabolicWatts * efficiency
  const cyclingVo2Rel = vo2max * (1 - discount)
  const acsmMapWatts =
    (Math.max(0, cyclingVo2Rel - ACSM_BASE) * massKg) / ACSM_VO2_PER_KGM / ACSM_KGM_PER_WATT
  const acsmFtp = acsmMapWatts * MAP_FTP_RATIO
  const ftpMean = (efficiencyFtp + acsmFtp) / 2
  const ftp = round10(ftpMean)
  return {
    vo2max: round(vo2max, 1),
    massKg: round(massKg, 1),
    crossModalDiscountPct: params.crossModalDiscountPct,
    thresholdPct: params.thresholdPct,
    grossEfficiencyPct: params.grossEfficiencyPct,
    absoluteVo2: round(absoluteVo2, 2),
    cyclingVo2max: round(cyclingVo2max, 2),
    thresholdVo2: round(thresholdVo2, 2),
    metabolicWatts: round(metabolicWatts, 0),
    efficiencyFtp: round(efficiencyFtp, 0),
    acsmMapWatts: round(acsmMapWatts, 0),
    acsmFtp: round(acsmFtp, 0),
    ftp,
    low: round5(ftpMean - ERROR_W),
    high: round5(ftpMean + ERROR_W),
    wattsPerKg: round(ftp / massKg, 2),
  }
}
