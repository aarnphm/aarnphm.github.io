---
date: '2026-06-19'
description: iOS + watchOS app that snapshots Apple Health into one JSON file on iCloud Drive.
id: index
modified: 2026-07-09 22:41:23 GMT-04:00
seealso:
  - '[[triathlon]]'
tags:
  - projects
title: HealthDataExporter
---

A small iOS + watchOS app that reads Apple Health and writes a single JSON file to iCloud Drive. It exports the calories, weight, VO2 max, swim splits, swim pace inputs, and stroke rate inputs I use for [[triathlon]] training.

## metrics

A 180-day window back from `now`, bucketed by the device's autoupdating calendar and timezone.

| metric         | HealthKit type                             | reduction                                       |
| -------------- | ------------------------------------------ | ----------------------------------------------- |
| active energy  | `activeEnergyBurned`                       | cumulative sum per day                          |
| basal energy   | `basalEnergyBurned`                        | cumulative sum per day                          |
| dietary energy | `dietaryEnergyConsumed`                    | cumulative sum per day                          |
| body mass      | `bodyMass`                                 | latest reading per day                          |
| VO2 max        | `vo2Max`                                   | latest reading per day                          |
| swims          | `distanceSwimming` + `swimmingStrokeCount` | one row per swim or multisport swim activity    |
| workout HR     | `workoutType` + `heartRate`                | one row per workout with its heart rate samples |

## output

Encodes `apple-health-import.json` into the ubiquity container `iCloud.xyz.aarnphm.healthexporter`, surfaced at `iCloud Drive/HealthExporter/`

```json
{
  "version": 3,
  "generatedAt": "2026-06-19T20:27:00-04:00",
  "timezone": "America/Toronto",
  "days": [
    {
      "date": "2026-06-19",
      "burnKcal": 2890,
      "activeKcal": 640,
      "intakeKcal": 2100,
      "weightKg": 89.8,
      "vo2max": 48.2
    }
  ],
  "swims": [
    {
      "id": "A45B1F35-9F51-4917-B656-C17BF2D07434",
      "date": "2026-06-19",
      "start": "2026-06-19T11:00:00Z",
      "end": "2026-06-19T12:00:00Z",
      "totalM": 1500,
      "laps": 60,
      "activeTimeS": 1800,
      "strokeCount": 960,
      "strokeTimeS": 1700,
      "strokes": { "freestyle": 1300, "breaststroke": 200 }
    }
  ],
  "workouts": [
    {
      "id": "7E0BEF46-8C0E-4E08-8E2B-0F2E0A1C9E63",
      "activity": "cycling",
      "start": "2026-07-01T01:11:00Z",
      "end": "2026-07-01T02:07:45Z",
      "durationS": 3405,
      "heartRate": [
        { "time": "2026-07-01T01:11:04Z", "bpm": 118 },
        { "time": "2026-07-01T01:11:09Z", "bpm": 122 }
      ]
    }
  ]
}
```

The exporter writes every workout row, including workouts with no heart rate samples. A swim row uses its workout UUID. A swim inside a `swimBikeRun` workout uses its workout activity UUID. Two swims on the same day remain separate.

## architecture

- `HealthAggregator` folds samples into day buckets and swim sessions. It does not import HealthKit, so `HealthAggregatorTests` runs without a device.
- `HealthKitService` handles authorization and HealthKit queries. It reads all workouts once. Two associated sample queries then fetch swim distance and stroke count data for every matching workout.
- `HealthExportWriter` encodes JSON and writes it to the iCloud container.
- `HealthExportRuntime` connects the other parts and exports again when an observer query fires.

## background

`HealthBackgroundScheduler` registers a `BGAppRefreshTask` (`xyz.aarnphm.healthexporter.export`) aimed at ~02:30 local and schedules it again every time the app enters background. This is the daily reconciliation pass. One observer covers every exported HealthKit type. Hourly background delivery uses active energy as its regular trigger, plus infrequent changes to dietary energy, body mass, VO2 max, and workouts. Heart rate and swim samples are picked up by those exports without causing their own wakeups. Automatic exports query data from today and the previous two calendar days. They merge it into the document containing 180 days and skip the iCloud write when the health data did not change. The app reloads the existing export on launch and updates it when it is more than one hour old.

iOS decides the exact background execution time. Swiping the app away in the app switcher suppresses HealthKit background launches until the app is opened again. Leaving the app suspended or allowing iOS to terminate it preserves background delivery.

The watch target carries no HealthKit access of its own. It sends an `export` command over `WCSession`, and the phone publishes the resulting day and swim counts back through application context for the watch to display.

## quirks

- `totalM`, `activeTimeS`, and `strokeCount` come from workout or workout activity statistics. These totals remain complete when HealthKit condenses old samples.
- `laps` comes from workout lap events when they are available. Associated distance samples provide the metres by style. The exporter does not invent a pool length or lap distance when HealthKit omits them.
- Stroke style comes from `HKMetadataKeySwimmingStrokeStyle` on workout lap events. The app maps it to freestyle, breaststroke, backstroke, butterfly, mixed, or kickboard.
- `strokeTimeS` is the union of every positive associated stroke count interval. The app excludes intervals whose matching lap event is kickboard. It leaves `strokeTimeS` empty when HealthKit no longer has detailed intervals.
- Pace per 100 m is `activeTimeS / totalM * 100`. Stroke rate per minute is `strokeCount / strokeTimeS * 60`.
