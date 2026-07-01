import {
  type PaceContext,
  type PaceDayState,
  type PaceLegSpec,
  buildFeatureVector,
  contextFromMetaRow,
  dayStateFromFeedRow,
  parsePaceFeed,
} from './pace-features'
import { isRecord, readNumber, readString } from './type-guards'

export interface PaceForecast {
  mu: number
  sigma: number
}

export interface LegForecast {
  sport: PaceLegSpec['sport']
  distanceKm: number
  mu: number
  sigma: number
  midSec: number
  fastSec: number
  slowSec: number
}

export interface FinishForecast {
  legs: LegForecast[]
  midSec: number
  fastSec: number
  slowSec: number
}

export interface WorkerLike {
  postMessage(message: unknown): void
  onmessage: ((event: MessageEvent) => void) | null
  terminate?(): void
}

export const Z80 = 1.2816

export class PaceForecaster {
  ready = false
  day: PaceDayState | null = null
  ctx: PaceContext | null = null
  private dayList: PaceDayState[] = []
  private seq = 0
  private pending = new Map<number, (p: PaceForecast | null) => void>()
  private loadResolve: ((ok: boolean) => void) | null = null

  constructor(private readonly worker: WorkerLike) {
    worker.onmessage = (event: MessageEvent): void => this.onMessage(event.data)
  }

  async init(base: string, family: string, feedUrl: string): Promise<boolean> {
    try {
      const text = await (await fetch(feedUrl)).text()
      const feed = parsePaceFeed(text)
      this.dayList = feed.days.map(dayStateFromFeedRow).filter((d): d is PaceDayState => d !== null)
      this.day = this.dayList.at(-1) ?? null
      this.ctx = feed.meta ? contextFromMetaRow(feed.meta) : null
    } catch {
      return false
    }
    if (!this.day || !this.ctx) return false
    const loaded = await new Promise<boolean>(resolve => {
      this.loadResolve = resolve
      this.worker.postMessage({ type: 'load', base, family })
    })
    this.ready = loaded
    return loaded
  }

  predict(raw: Float32Array, presence: Float32Array): Promise<PaceForecast | null> {
    if (!this.ready) return Promise.resolve(null)
    const id = ++this.seq
    return new Promise(resolve => {
      this.pending.set(id, resolve)
      this.worker.postMessage({ type: 'predict', id, raw: [...raw], presence: [...presence] })
    })
  }

  forecastLegAt(day: PaceDayState, leg: PaceLegSpec): Promise<PaceForecast | null> {
    if (!this.ctx) return Promise.resolve(null)
    const fv = buildFeatureVector(day, leg, this.ctx)
    return this.predict(fv.raw, fv.presence)
  }

  forecastLeg(leg: PaceLegSpec): Promise<PaceForecast | null> {
    return this.day ? this.forecastLegAt(this.day, leg) : Promise.resolve(null)
  }

  dayStateAgo(n: number): PaceDayState | null {
    const i = this.dayList.length - 1 - n
    return i >= 0 ? this.dayList[i] : null
  }

  async forecastFinish(
    legs: PaceLegSpec[],
    transitionSec = 0,
    z = Z80,
  ): Promise<FinishForecast | null> {
    const out: LegForecast[] = []
    let mid = transitionSec
    let timeVar = 0
    for (const leg of legs) {
      const f = await this.forecastLeg(leg)
      if (!f || f.mu <= 0) return null
      const meters = leg.distanceKm * 1000
      const midSec = meters / f.mu
      const timeSd = (meters / (f.mu * f.mu)) * f.sigma
      out.push({
        sport: leg.sport,
        distanceKm: leg.distanceKm,
        mu: f.mu,
        sigma: f.sigma,
        midSec,
        fastSec: Math.max(0, midSec - z * timeSd),
        slowSec: midSec + z * timeSd,
      })
      mid += midSec
      timeVar += timeSd * timeSd
    }
    const sd = Math.sqrt(timeVar)
    return { legs: out, midSec: mid, fastSec: Math.max(0, mid - z * sd), slowSec: mid + z * sd }
  }

  dispose(): void {
    this.worker.terminate?.()
    this.pending.clear()
    this.ready = false
  }

  private onMessage(data: unknown): void {
    if (!isRecord(data)) return
    const type = readString(data, 'type')
    if (type === 'loaded') {
      this.loadResolve?.(data.ok === true)
      this.loadResolve = null
      return
    }
    if (type === 'prediction') {
      const id = readNumber(data, 'id')
      if (id == null) return
      const resolve = this.pending.get(id)
      this.pending.delete(id)
      resolve?.(
        data.ok === true
          ? { mu: readNumber(data, 'mu') ?? 0, sigma: readNumber(data, 'sigma') ?? 0 }
          : null,
      )
    }
  }
}
