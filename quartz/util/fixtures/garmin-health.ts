import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import type { GarminHealthDay } from '../../plugins/stores/garmin-health'
import {
  garminBodyBattery,
  garminTrainingReadiness,
  garminTrainingStatus,
  garminEnduranceScore,
  garminHillScore,
  garminHealthFetchResult,
} from '../garmin-health'
import { isRecord } from '../type-guards'

const payload: unknown = JSON.parse(
  readFileSync(new URL('./garmin-health.json', import.meta.url), 'utf8'),
)
assert.ok(isRecord(payload))
export const raw = payload
export const date = '2026-09-21'
export const fetchedAt = Date.parse('2026-09-22T01:00:00Z')
export const health: GarminHealthDay = {
  source: 'garmin',
  date,
  bodyBattery: garminHealthFetchResult(garminBodyBattery(raw.bodyBattery, date), fetchedAt),
  trainingReadiness: garminHealthFetchResult(
    garminTrainingReadiness(raw.readiness, date),
    fetchedAt,
  ),
  trainingStatus: garminHealthFetchResult(
    garminTrainingStatus(raw.trainingStatus, date),
    fetchedAt,
  ),
  enduranceScore: garminHealthFetchResult(garminEnduranceScore(raw.endurance, date), fetchedAt),
  hillScore: garminHealthFetchResult(garminHillScore(raw.hill, date), fetchedAt),
}
