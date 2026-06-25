---
date: '2026-05-29'
description: training data and tracking
dexa:
  - ag: 1.29
    arms:
      bmc: 1
      fat: 4.7
      lean: 17.2
    bmcLbs: 7.8
    bmd: 1.383
    bmdT: 1.8
    bodyFat: 27.4
    date: '2026-06-25'
    fatLbs: 54.2
    leanLbs: 135.7
    legs:
      bmc: 3
      fat: 17.7
      lean: 54.4
    rmr: 1695
    rsmi: 9.2
    totalLbs: 197.6
    trunk:
      bmc: 2.3
      fat: 29.3
      lean: 55
    vatLbs: 1.24
id: triathlon
layout: triathlon
modified: 2026-06-25 14:09:44 GMT-04:00
seealso:
  - '[[thoughts/pdfs/triathlon.pdf|fuel plan]]'
strava: '2026-05-13'
tags:
  - life
  - self
  - evergreen
title: triathlon
triathlon: olympic
vo2max:
  - date: '2026-06-25'
    hrMax: 182
    maxKmh: 15
    percentile: 70
    value: 47.8
    ve: 117
    vt1Hr: 145
    vt1Kmh: 10.5
    zonesHr:
      - 112
      - 121
      - 142
      - 171
    zonesKmh:
      - 7
      - 8
      - 10.5
      - 14
---

```tracking
date: 2026-05-20
weight: 206
```

```tracking race=true event="B4BH 2026"
date: 2026-05-31
weight: 204
```

```tracking
date: 2026-06-05
weight: 203
```

```tracking
date: 2026-06-06
weight: 200
```

```tracking
date: 2026-06-07
weight: 198
```

```tracking
date: 2026-06-08
weight: 200
```

```tracking
date: 2026-06-09
weight: 195
```

```tracking
date: 2026-06-10
weight: 197
wind: 15kph 20NW
```

```tracking
date: 2026-06-11
weight: 195
wind: 17kph 230SW
```

<!-- training plan start
meta: supertri toronto 2026
distance: olympic
date: 2026-07-26
target: sub-3
author: gpt-5.5-pro-xhigh,claude-fable-5[1m]
-->

The plan uses four constraints from the literature and the current Runna block:

- Most minutes stay easy, with two hard swim or bike sessions per week. Endurance athletes usually improve through high-volume low-intensity work plus careful high-intensity dosing.[^seiler]
- Swim paces are set around critical swim speed, because CSS tracks the highest sustainable aerobic swim speed and gives a repeatable way to set 100 m work.[^css]
- The taper keeps intensity while cutting volume. The best-supported taper shape is about two weeks with volume down 40-60% while intensity stays sharp.[^taper]
- Runna owns the run calendar through race day. Bike and swim work wrap around those rows because the run is the binding leg.

### intensity anchors

Swim:

- Easy: 2:25-2:45/100m, relaxed breathing, finish cleaner than you started.
- CSS: 2:12-2:20/100m now, pull this toward 2:08-2:12/100m by mid-July.
- Fast: 1:55-2:08/100m on 25-50 m repeats only.
- Race: 32:30-33:00 for 1.5 km, about 2:10-2:12/100m in the pool.

Bike, using FTP 200 W until a new test proves otherwise:

- Easy: 120-150 W, HR under 145 if the day is normal.
- Tempo: 160-180 W, smooth cadence, no spikes.
- Race power: 180-195 W now, only call it 200-210 W after 60-75 minutes stops making the legs detonate.
- VO2: 220-250 W for 3-5 minute repeats.
- Race: 40 km in 80-82 minutes, 29.3-30.0 km/h.

Runna, using the shared 10 km plan:

- Estimated 10 km: 45:00-46:40, about 7:15-7:31/mi or 4:30-4:40/km.
- Easy: conversational, boring, almost suspiciously restrained.
- Intervals, tempo, drop set, race-pace half miles: one hard run stimulus. The bike that day is easy or off.
- Long run: keep it aerobic. Fuel any run over 75 minutes with 30-60 g carbs/h.
- Strength: keep the Runna session, then protect the next bike-quality day if legs feel stale.
- Race: 10 km off the bike on 07-26, first 2 km under control, then squeeze.

Recovery gates:

- If readiness is under 70 for two straight days, HRV z is below -1, RHR is 5 bpm over baseline, or TSB is below -25, replace the next hard session with easy technique or 45-60 minutes Z2.
- No low-cadence grinding while the legs are sore. Keep most race-power work at 80-90 rpm.
- Fuel every ride over 75 minutes. Use 30-60 g carbs/h for 75-150 minutes and practice 60-90 g/h for race rehearsals.[^carb]

### weekly build

| week    |          dates |        load ceiling | Runna    | swim       | bike      | key proof                                              |
| ------- | -------------: | ------------------: | -------- | ---------- | --------- | ------------------------------------------------------ |
| absorb  | 06-14 to 06-21 |             450-500 | pre-plan | 4.5-5.0 km | 4.0-5.0 h | 1.5 km continuous swim, one controlled tempo bike      |
| build 1 | 06-22 to 06-28 |             600-650 | 18.5 mi  | 5.5-6.0 km | 5.0-6.0 h | 400 m repeats, 6.5 mi long run, 2x20 min bike tempo    |
| build 2 | 06-29 to 07-05 |             680-730 | 20.0 mi  | 6.0-7.0 km | 6.5-7.5 h | tempo 2 miles, 7.5 mi long run, 40 km bike rehearsal   |
| peak    | 07-06 to 07-12 |             720-780 | 21.8 mi  | 7.0-8.0 km | 7.0-8.0 h | 1 km repeats, 9 mi progressive long run, B5 durability |
| taper 1 | 07-13 to 07-19 |             500-560 | 17.6 mi  | 5.0-5.5 km | 4.5-5.5 h | drop set, 6 mi long run, race-kit swim plus bike       |
| race    | 07-20 to 07-26 | 280-360 before race | 13.0 mi  | 2.5-3.5 km | 2.5-3.5 h | race-pace half miles, 10 km race, arrive sharp         |

The load ceiling is total weekly load. The Runna mileage is the run lane, so extra jogs do not exist. If the app moves a run, keep the weekly mileage inside the same band and keep at least 24 hours between hard run work and bike race-power work.

### calendar

| date  | Runna                                                  | swim                                                             | bike                                                    |
| ----- | ------------------------------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------- |
| 06-14 | pre-plan                                               | off or 1000 m easy technique                                     | off or 45 min easy spin                                 |
| 06-15 | pre-plan                                               | S1, 1600-1800 m                                                  | off                                                     |
| 06-16 | pre-plan                                               | off                                                              | B1, 60 min with 5x3 min at 220-240 W                    |
| 06-17 | pre-plan                                               | S2, 1800-2000 m with 3x400 m easy-CSS                            | off or 45 min easy                                      |
| 06-18 | pre-plan                                               | off                                                              | B2, 75 min with 3x10 min at 165-175 W                   |
| 06-19 | pre-plan                                               | S3, 1500-1700 m speed and drills                                 | off                                                     |
| 06-20 | pre-plan                                               | off                                                              | B3, 2:00 Z2, last 20 min at 160-170 W                   |
| 06-21 | rest day                                               | 1500 m continuous, easy                                          | 45 min recovery spin                                    |
| 06-22 | off                                                    | S1, 1800-2000 m                                                  | off                                                     |
| 06-23 | Rolling 800s, 2.6 mi                                   | off                                                              | off or 30-45 min easy spin                              |
| 06-24 | Full Body Strength Session, 55-65 min; 3.5 mi Easy Run | S2, 2200 m with 4x400 m                                          | 45 min easy                                             |
| 06-25 | Loading Up, 55-65 min                                  | off                                                              | B2, 75-90 min with 2x20 min at 175-185 W                |
| 06-26 | 4.5 mi Easy Run                                        | S3, 1800 m                                                       | off                                                     |
| 06-27 | 400 m Repeats, 4 mi; Stretch & Stability 1, 25 min     | off                                                              | 60-90 min Z2 only if the reps stay controlled           |
| 06-28 | 6.5 mi Long Run                                        | 1600-1800 m continuous or open water                             | 45-60 min recovery spin                                 |
| 06-29 | 4 mi Easy Run                                          | S1, 2000 m                                                       | off                                                     |
| 06-30 | Strength Supersets, 55-65 min                          | off                                                              | B1, 75 min with 6x3 min at 230-250 W                    |
| 07-01 | Tempo 2 Miles, 4.5 mi                                  | S2, 2400 m with 5x400 m                                          | 45 min easy                                             |
| 07-02 | Lower Body Strength Session, 55-65 min                 | off                                                              | B2, 95 min with 3x15 min at 180-190 W                   |
| 07-03 | 4 mi Easy Run                                          | S3, 1800-2000 m                                                  | off                                                     |
| 07-04 | Stretch & Stability 2, 25 min                          | off                                                              | B4, 40 km rehearsal at 175-190 W, cap spikes over 250 W |
| 07-05 | 7.5 mi Long Run                                        | 1800 m easy continuous                                           | 60 min recovery spin                                    |
| 07-06 | 3.25 mi Easy Run                                       | S1, 2200 m                                                       | off                                                     |
| 07-07 | Full Body Endurance Session, 55-65 min                 | off                                                              | B1, 80 min with 5x5 min at 220-240 W                    |
| 07-08 | 1 km Repeats, 4.5 mi                                   | S2, 2600 m with 3x600 m                                          | 45 min easy                                             |
| 07-09 | Legs & Core Endurance, 55-65 min                       | off                                                              | B2, 100 min with 3x12 min at 190-205 W                  |
| 07-10 | 5 mi Brick Run                                         | S3, 2000 m                                                       | off                                                     |
| 07-11 | Stretch & Stability 3, 25 min                          | off or 1000 m easy                                               | B5, 75-90 min at race power blocks, full fueling        |
| 07-12 | 9 mi Progressive Long Run                              | 1800-2000 m continuous, sighting every 6-8 strokes if open water | 60 min easy                                             |
| 07-13 | 3.25 mi Easy Run                                       | S1, 1800 m                                                       | off                                                     |
| 07-14 | Ticking Over, 55-65 min                                | off                                                              | B1, 60 min with 4x3 min at 225-240 W                    |
| 07-15 | Drop Set, 3.4 mi                                       | S2, 2200 m with 3x400 m                                          | 45 min easy                                             |
| 07-16 | Light(er) Work, 55-65 min                              | off                                                              | B2, 75 min with 2x15 min at 185-195 W                   |
| 07-17 | 5 mi Easy Run                                          | S3, 1600 m, stop while stroke still looks good                   | off                                                     |
| 07-18 | Stretch & Stability 4, 25 min                          | 1500 m race-kit swim, controlled                                 | B4, 30-40 km at 180-190 W, full fueling                 |
| 07-19 | 6 mi Long Run                                          | off or 1000 m easy                                               | 45 min recovery spin                                    |
| 07-20 | off                                                    | S1, 1400-1600 m                                                  | off                                                     |
| 07-21 | Hard Work Is Done, 55-65 min                           | off                                                              | B1, 50 min with 3x3 min at 220-235 W                    |
| 07-22 | Race Pace Practice Half Miles, 3.5 mi                  | S3, 1200-1500 m with 8x50 m fast                                 | 40 min easy                                             |
| 07-23 | off                                                    | off                                                              | B2, 45 min with 3x5 min at 180-190 W                    |
| 07-24 | 3.25 mi Easy Run                                       | 1000-1200 m easy, 4x50 m fast                                    | off                                                     |
| 07-25 | Stretch & Stability 5, 25 min                          | 10-15 min water feel if available                                | 20 min spin, 3x30 s openers                             |
| 07-26 | race: 10 km, 6.2 mi, 45:00-46:40 target                | race: 1.5 km, settle in the first 300 m                          | race: 40 km, cap first 10 min at 180 W, then 185-195 W  |

### session library

S1, technique plus CSS:

- 300 m easy.
- 8x50 m drill/swim by 25 m, 15-20 s rest.
- Main set starts at 6x100 m at 2:18-2:22/100m and progresses to 10x100 m at 2:12-2:18/100m.
- 4x50 m fast at 1:55-2:08/100m, full control.
- 200 m easy.

S2, endurance:

- 400 m easy.
- 3-5x400 m at 2:18-2:28/100m, 45-60 s rest.
- Progress one rep each build week before adding speed.
- 200 m easy.

S3, speed and stroke:

- 300 m easy.
- 12-20x50 m alternating fast/easy. Fast is 1:55-2:08/100m; easy is 2:30/100m or slower.
- 4x100 m pull or strong-form swim.
- 200 m easy.

B1, VO2:

- 15 min warmup.
- 5-6x3-5 min at 220-250 W, equal easy recovery.
- 10-15 min cooldown.
- End the session if power falls below target for two consecutive reps.

B2, tempo and race durability:

- 15 min warmup.
- Progress from 3x10 min at 165-175 W to 3x15 min at 180-190 W and then 3x12 min at 190-205 W.
- Keep cadence 80-90 rpm.
- Cool down until HR is boring again.

B3, long aerobic:

- 2:00-3:00 at 120-150 W.
- Add 20-30 min at 160-175 W only when readiness is normal.
- Eat 60-90 g carbs/h and drink to thirst plus sodium in heat.

B4, race rehearsal:

- 10 min easy.
- 40 km at 175-190 W, target 29-30 km/h, no power spikes over 250 W unless traffic forces it.
- Hold aero position whenever safe.
- Practice race bottle, carbs, and the first 10 minutes of restraint.

B5, race-power durability:

- 15 min warmup.
- 3x15 min at 180-195 W, 5 min easy between.
- Finish with 10 min easy. If legs are sore at minute 45, choose 185 W before 210 W

Runna R1, easy run:

- Hold conversation pace and keep cadence light.
- If HR drifts above easy, slow down before adding distance.

Runna R2, intervals or tempo:

- Warm up until stride mechanics feel normal.
- Hit the prescribed Runna work and end the session while form still exists.
- Bike quality moves at least 24 hours away unless the bike is explicitly easy.

Runna R3, long run:

- Start slower than ego wants.
- Fuel 30-60 g carbs/h once the run passes 75 minutes.
- Last 10 minutes can progress only if the next day is not a bike-quality day.

### race track

![[https://strava-embeds.com/route/3459615466775941042?style=standard&amp;fullWidth=true&amp;clubId=1459904&amp;fromEmbed=true#ns=94460b8c-1ff3-4475-b2f1-a5beeb3d47e5&amp;hostOrigin=https%3A%2F%2Fsupertri.com&amp;hostPath=%2Ftoronto-triathlon%2Folympic%2F&amp;hostTitle=Supertri+Toronto+Olympic+Triathlon]]

![[https://strava-embeds.com/route/3459617551271887794?style=standard&fullWidth=true&clubId=1459904&fromEmbed=true#ns=ba8ee3c9-ca68-41d1-a539-ca0ed0ea208f&hostOrigin=https%3A%2F%2Fsupertri.com&hostPath=%2Ftoronto-triathlon%2Folympic%2F&hostTitle=Supertri+Toronto+Olympic+Triathlon]]

![[https://strava-embeds.com/route/3459617422094043424?style=standard&amp;fullWidth=true&amp;clubId=1459904&amp;fromEmbed=true#ns=6d5683ba-67f2-4753-b660-9b1e4c7ef5f0&amp;hostOrigin=https%3A%2F%2Fsupertri.com&amp;hostPath=%2Ftoronto-triathlon%2Folympic%2F&amp;hostTitle=Supertri+Toronto+Olympic+Triathlon]]

[^seiler]: Seiler, "What is best practice for training intensity and duration distribution in endurance athletes?", 2010. https://pubmed.ncbi.nlm.nih.gov/20861519/

[^css]: Dekerle et al., "Validity and reliability of critical speed, critical stroke rate and anaerobic swimming capacity in relation to front crawl swimming performances", 2002. https://pubmed.ncbi.nlm.nih.gov/11842355/

[^taper]: Bosquet et al., "Effects of tapering on performance: a meta-analysis", 2007. https://pubmed.ncbi.nlm.nih.gov/17762369/ See also Mujika and Padilla, "Scientific bases for precompetition tapering strategies", 2003. https://pubmed.ncbi.nlm.nih.gov/12840640/

[^carb]: Burke et al., "Carbohydrates for training and competition", 2011. https://pubmed.ncbi.nlm.nih.gov/21660838/

<!-- training plan end -->
