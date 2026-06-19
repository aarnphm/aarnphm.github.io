---
date: '2026-05-29'
description: training data and tracking
id: triathlon
layout: triathlon
modified: 2026-06-19 12:47:23 GMT-04:00
seealso:
  - '[[thoughts/pdfs/triathlon.pdf|fuel plan]]'
strava: '2026-05-12'
tags:
  - life
  - self
  - evergreen
title: triathlon
triathlon: olympic
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

The plan uses three constraints from the literature:

- Most minutes stay easy, with two hard swim or bike sessions per week. Endurance athletes usually improve through high-volume low-intensity work plus careful high-intensity dosing.[^seiler]
- Swim paces are set around critical swim speed, because CSS tracks the highest sustainable aerobic swim speed and gives a repeatable way to set 100 m work.[^css]
- The taper keeps intensity while cutting volume. The best-supported taper shape is about two weeks with volume down 40-60% while intensity stays sharp.[^taper]
- Combine with Runna training plan for maximum utilization

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

Recovery gates:

- If readiness is under 70 for two straight days, HRV z is below -1, RHR is 5 bpm over baseline, or TSB is below -25, replace the next hard session with easy technique or 45-60 minutes Z2.
- No low-cadence grinding while the legs are sore. Keep most race-power work at 80-90 rpm.
- Fuel every ride over 75 minutes. Use 30-60 g carbs/h for 75-150 minutes and practice 60-90 g/h for race rehearsals.[^carb]

### weekly build

| week    |          dates |        load ceiling | swim       | bike      | key proof                                         |
| ------- | -------------: | ------------------: | ---------- | --------- | ------------------------------------------------- |
| absorb  | 06-14 to 06-21 |             450-500 | 4.5-5.0 km | 4.0-5.0 h | 1.5 km continuous swim, one controlled tempo bike |
| build 1 | 06-22 to 06-28 |             600-650 | 5.5-6.0 km | 5.5-6.5 h | 1.6-1.8 km continuous swim, 2x20 min tempo        |
| build 2 | 06-29 to 07-05 |             680-730 | 6.0-7.0 km | 6.5-7.5 h | 40 km bike rehearsal at 175-190 W                 |
| peak    | 07-06 to 07-12 |             720-780 | 7.0-8.0 km | 7.0-8.0 h | 1.8-2.0 km swim, 75-90 min race-power durability  |
| taper 1 | 07-13 to 07-19 |             500-560 | 5.0-5.5 km | 4.5-5.5 h | race-kit swim plus 30-40 km bike, full fueling    |
| race    | 07-20 to 07-26 | 280-360 before race | 2.5-3.5 km | 2.5-3.5 h | arrive sharp and fresh                            |

The load ceiling is total weekly load, including any running. Running has to fit inside the ceiling because the run is the current binding leg.

### calendar

| date  | swim                                                             | bike                                                    |
| ----- | ---------------------------------------------------------------- | ------------------------------------------------------- |
| 06-14 | off or 1000 m easy technique                                     | off or 45 min easy spin                                 |
| 06-15 | S1, 1600-1800 m                                                  | off                                                     |
| 06-16 | off                                                              | B1, 60 min with 5x3 min at 220-240 W                    |
| 06-17 | S2, 1800-2000 m with 3x400 m easy-CSS                            | off or 45 min easy                                      |
| 06-18 | off                                                              | B2, 75 min with 3x10 min at 165-175 W                   |
| 06-19 | S3, 1500-1700 m speed and drills                                 | off                                                     |
| 06-20 | off                                                              | B3, 2:00 Z2, last 20 min at 160-170 W                   |
| 06-21 | 1500 m continuous, easy                                          | 45 min recovery spin                                    |
| 06-22 | S1, 1800-2000 m                                                  | off                                                     |
| 06-23 | off                                                              | B1, 70 min with 5x4 min at 220-240 W                    |
| 06-24 | S2, 2200 m with 4x400 m                                          | 45 min easy                                             |
| 06-25 | off                                                              | B2, 90 min with 2x20 min at 175-185 W                   |
| 06-26 | S3, 1800 m                                                       | off                                                     |
| 06-27 | off                                                              | B3, 2:30 Z2, 70-80 km if roads cooperate                |
| 06-28 | 1600-1800 m continuous or open water                             | 45-60 min recovery spin                                 |
| 06-29 | S1, 2000 m                                                       | off                                                     |
| 06-30 | off                                                              | B1, 75 min with 6x3 min at 230-250 W                    |
| 07-01 | S2, 2400 m with 5x400 m                                          | 45 min easy                                             |
| 07-02 | off                                                              | B2, 95 min with 3x15 min at 180-190 W                   |
| 07-03 | S3, 1800-2000 m                                                  | off                                                     |
| 07-04 | off                                                              | B4, 40 km rehearsal at 175-190 W, cap spikes over 250 W |
| 07-05 | 1800 m easy continuous                                           | 60 min recovery spin                                    |
| 07-06 | S1, 2200 m                                                       | off                                                     |
| 07-07 | off                                                              | B1, 80 min with 5x5 min at 220-240 W                    |
| 07-08 | S2, 2600 m with 3x600 m                                          | 45 min easy                                             |
| 07-09 | off                                                              | B2, 100 min with 3x12 min at 190-205 W                  |
| 07-10 | S3, 2000 m                                                       | off                                                     |
| 07-11 | off or 1000 m easy                                               | B5, 75-90 min at race power blocks, full fueling        |
| 07-12 | 1800-2000 m continuous, sighting every 6-8 strokes if open water | 60 min easy                                             |
| 07-13 | S1, 1800 m                                                       | off                                                     |
| 07-14 | off                                                              | B1, 60 min with 4x3 min at 225-240 W                    |
| 07-15 | S2, 2200 m with 3x400 m                                          | 45 min easy                                             |
| 07-16 | off                                                              | B2, 75 min with 2x15 min at 185-195 W                   |
| 07-17 | S3, 1600 m, stop while stroke still looks good                   | off                                                     |
| 07-18 | 1500 m race-kit swim, controlled                                 | B4, 30-40 km at 180-190 W, full fueling                 |
| 07-19 | off or 1000 m easy                                               | 45 min recovery spin                                    |
| 07-20 | S1, 1400-1600 m                                                  | off                                                     |
| 07-21 | off                                                              | B1, 50 min with 3x3 min at 220-235 W                    |
| 07-22 | S3, 1200-1500 m with 8x50 m fast                                 | 40 min easy                                             |
| 07-23 | off                                                              | B2, 45 min with 3x5 min at 180-190 W                    |
| 07-24 | 1000-1200 m easy, 4x50 m fast                                    | off                                                     |
| 07-25 | 10-15 min water feel if available                                | 20 min spin, 3x30 s openers                             |
| 07-26 | race: 1.5 km, settle in the first 300 m                          | race: 40 km, cap first 10 min at 180 W, then 185-195 W  |

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

### race track

![[https://strava-embeds.com/route/3459615466775941042?style=standard&amp;fullWidth=true&amp;clubId=1459904&amp;fromEmbed=true#ns=94460b8c-1ff3-4475-b2f1-a5beeb3d47e5&amp;hostOrigin=https%3A%2F%2Fsupertri.com&amp;hostPath=%2Ftoronto-triathlon%2Folympic%2F&amp;hostTitle=Supertri+Toronto+Olympic+Triathlon]]

![[https://strava-embeds.com/route/3459617551271887794?style=standard&fullWidth=true&clubId=1459904&fromEmbed=true#ns=ba8ee3c9-ca68-41d1-a539-ca0ed0ea208f&hostOrigin=https%3A%2F%2Fsupertri.com&hostPath=%2Ftoronto-triathlon%2Folympic%2F&hostTitle=Supertri+Toronto+Olympic+Triathlon]]

![[https://strava-embeds.com/route/3459617422094043424?style=standard&amp;fullWidth=true&amp;clubId=1459904&amp;fromEmbed=true#ns=6d5683ba-67f2-4753-b660-9b1e4c7ef5f0&amp;hostOrigin=https%3A%2F%2Fsupertri.com&amp;hostPath=%2Ftoronto-triathlon%2Folympic%2F&amp;hostTitle=Supertri+Toronto+Olympic+Triathlon]]

[^seiler]: Seiler, "What is best practice for training intensity and duration distribution in endurance athletes?", 2010. https://pubmed.ncbi.nlm.nih.gov/20861519/

[^css]: Dekerle et al., "Validity and reliability of critical speed, critical stroke rate and anaerobic swimming capacity in relation to front crawl swimming performances", 2002. https://pubmed.ncbi.nlm.nih.gov/11842355/

[^taper]: Bosquet et al., "Effects of tapering on performance: a meta-analysis", 2007. https://pubmed.ncbi.nlm.nih.gov/17762369/ See also Mujika and Padilla, "Scientific bases for precompetition tapering strategies", 2003. https://pubmed.ncbi.nlm.nih.gov/12840640/

[^carb]: Burke et al., "Carbohydrates for training and competition", 2011. https://pubmed.ncbi.nlm.nih.gov/21660838/

<!-- training plan end -->
