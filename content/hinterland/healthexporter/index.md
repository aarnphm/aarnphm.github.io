---
date: '2026-06-19'
description: iOS + watchOS app that snapshots Apple Health into one JSON file on iCloud Drive.
id: index
modified: 2026-06-22 09:24:18 GMT-04:00
seealso:
  - '[[triathlon]]'
tags:
  - projects
title: HealthDataExporter
---

A small iOS + watchOS app that reads Apple Health and writes a single JSON file to iCloud Drive, so the numbers I care about for [[triathlon]] training (calories, weight, VO2 max, swim splits) live in one machine-readable place instead of inside the Health app.

## metrics

A 180-day window back from `now`, bucketed by the device's autoupdating calendar and timezone.

| metric         | HealthKit type                             | reduction                             |
| -------------- | ------------------------------------------ | ------------------------------------- |
| active energy  | `activeEnergyBurned`                       | cumulative sum per day                |
| basal energy   | `basalEnergyBurned`                        | cumulative sum per day                |
| dietary energy | `dietaryEnergyConsumed`                    | cumulative sum per day                |
| body mass      | `bodyMass`                                 | latest reading per day                |
| VO2 max        | `vo2Max`                                   | latest reading per day                |
| swims          | `distanceSwimming` + `swimmingStrokeCount` | per-day total, laps, stroke breakdown |
| workout HR     | `workoutType` + `heartRate`                | workout-scoped heart-rate samples     |

## output

Encodes `apple-health-import.json` into the ubiquity container `iCloud.xyz.aarnphm.healthexporter`, surfaced at `iCloud Drive/HealthExporter/`

```json
{
  "version": 2,
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
      "date": "2026-06-19",
      "totalM": 1500,
      "laps": 60,
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

## architecture

- `HealthAggregator` — pure folding of samples into day buckets and swim days. No HealthKit import, so `HealthAggregatorTests` runs without a device.
- `HealthKitService` — authorization, the statistics-collection queries for cumulative daily sums, the sample queries for latest weight/VO2, swim reconstruction, and workout-scoped heart-rate streams.
- `HealthExportWriter` — JSON encode plus atomic write to the iCloud container.
- `HealthExportRuntime` — wires the three together and re-exports whenever an observer query fires.

## background

`HealthBackgroundScheduler` registers a `BGAppRefreshTask` (`xyz.aarnphm.healthexporter.export`) aimed at ~02:30 local and schedules it again every time the app enters background. This is the daily reconciliation pass. One observer covers every exported HealthKit type. Hourly background delivery uses active energy as its regular trigger, plus infrequent changes to dietary energy, body mass, VO2 max, and workouts. Heart rate and swim samples are picked up by those exports without causing their own wakeups. Automatic exports query data from today and the previous two calendar days. They merge it into the document containing 180 days and skip the iCloud write when the health data did not change. The app reloads the existing export on launch and updates it when it is more than one hour old.

iOS decides the exact background execution time. Swiping the app away in the app switcher suppresses HealthKit background launches until the app is opened again. Leaving the app suspended or allowing iOS to terminate it preserves background delivery.

The watch target carries no HealthKit access of its own. It sends an `export` command over `WCSession`, and the phone publishes the resulting day and swim counts back through application context for the watch to display.

## quirks

- Distance and stroke style arrive as separate swimming samples. The service joins them by rounded start-second; on a miss it falls back to that day's median distance, then to a 25 m default. Samples with zero meters are dropped.
- Stroke style comes out of sample metadata (`HKMetadataKeySwimmingStrokeStyle`) as an enum int, mapped to freestyle / breaststroke / backstroke / butterfly / mixed / kickboard.
