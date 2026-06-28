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
    ffmLbs: 143.5
    fatLossTo14PctLbs: 30.8
    rmr: 1695
    rsmi: 9.2
    targetBodyFat: 14
    targetFatLbs: 23.4
    targetTotalLbs: 166.9
    tissueFat: 28.5
    tissueLean: 68.7
    totalLbs: 197.6
    trunk:
      bmc: 2.3
      fat: 29.3
      lean: 55
    vatAreaIn2: 8.64
    vatLbs: 1.24
id: triathlon
layout: triathlon
modified: 2026-06-27 17:41:09 GMT-04:00
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
    caloriesAtVt1: 850
    hrAtVo2max: 180
    hrMax: 182
    massKg: 88.9
    maxKmh: 15
    percentile: 70
    value: 47.8
    ve: 117
    vt1Hr: 145
    vt1Kmh: 10.5
    vt2Hr:
    vt2Kmh:
    zonesHr:
      - 112
      - 121
      - 142
      - 171
    zonesKcal:
      - 480
      - 580
      - 820
      - 1170
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

The plan now uses five measured constraints:

- The VO2 test on 2026-06-25 put VO2max at 47.8 ml/kg/min at 88.9 kg, vVO2max at 15.0 km/h, HR at VO2max at 180 bpm, and peak observed HR at 182 bpm. VT1 was detected at 145 bpm and 10.5 km/h. VT2 was not detected, so do not invent precise threshold work above VT1.
- The useful fat-burning lane is boring: HR 121-142 bpm, roughly 8-10.5 km/h on the treadmill test. Most aerobic run minutes live there. Easy bike days use the same HR ceiling and power stays subordinate to HR drift.
- FTP is now a 230 W hypothesis with a 210-260 W realistic band, derived from running VO2max. The math crosses treadmill to bike and invents the missing VT2, so it sets power vocabulary, not training law.
- DEXA on 2026-06-25 measured 197.6 lb total mass, 54.2 lb fat, 135.7 lb lean, 7.8 lb bone mineral content, 27.4% total-mass fat, 28.5% tissue fat, 1.24 lb VAT, and 1695 kcal/day RMR.
- Holding fat-free mass at 143.5 lb, 14% body fat implies about 166.9 lb total mass and 23.4 lb fat. That is about 30.8 lb fat loss. Race day is 31 days away, so the race block targets 4-6 lb loss, not the full cut.
- Current analytics have Olympic prediction at about 3:02 before transitions, binding leg run, CTL 69, ATL 93, and TSB -24 on 2026-06-25. The test day counts as hard. The next 48 hours buy freshness before adding more intensity.

### intensity anchors

Lab HR zones:

| lane     | HR          | speed        | use                                      |
| -------- | ----------- | ------------ | ---------------------------------------- |
| warm     | <121 bpm    | <8 km/h      | warmup, cooldown, recovery               |
| fat burn | 121-142 bpm | 8-10.5 km/h  | most easy run volume and aerobic doubles |
| VT1 cap  | 143-145 bpm | 10.5 km/h    | upper cap for ordinary endurance         |
| hard     | 146-170 bpm | 10.5-14 km/h | tempo, race-specific work                |
| VO2      | 171+ bpm    | 14-15 km/h   | short intervals only                     |

Swim:

- Easy: 2:30-2:45/100 m, nasal-calm between repeats, no thrashing.
- CSS work: 2:18-2:24/100 m. Current modeled swim threshold is about 2:21/100 m, so the job is durability near threshold, not fantasy-pacing 1.5 km at sprint effort.[^css]
- Fast: 1:55-2:08/100 m on 25-50 m repeats only.
- Race: 34:30-35:30 for 1.5 km. Faster is loot, but sub-3 mostly comes from bike-run execution now.

Bike, using FTP 230 W as a VO2-derived hypothesis until a bike test proves otherwise:

- Easy: 125-160 W, HR under 142 if the day is for fat loss, under 145 if the day is endurance.
- Tempo: 170-195 W, smooth cadence, no spikes.
- Race power: 185-205 W, with 80-90 rpm and no heroic surges over 270 W.
- VO2: 245-275 W for 3-5 minute repeats. One bike VO2 session per week is enough while Runna has hard running.
- Race: 40 km in 82-85 minutes, then run. A faster bike that detonates the first 3 km of the run is fake profit.

Run:

- Easy and fat-burning: HR 121-142 bpm, often 5:45-7:30/km. Stay there even when the watch begs for content.
- Aerobic ceiling: 145 bpm. If a normal run crosses this without intent, slow down.
- Runna intervals and tempo: the one hard run stimulus for the week. Bike that day is easy or absent.
- Long run: keep it aerobic. Fuel any run over 75 minutes with 30-60 g carbs/h.
- Race: open the 10 km off the bike at 5:05-5:10/km for 2 km, then settle toward 4:50-5:00/km if HR and legs agree. The old 45-minute run target is an A+ day, not the pacing plan.

Recovery gates:

- If readiness is under 70 for two straight days, HRV z is below -1, RHR is 5 bpm over baseline, or TSB is below -25, replace the next hard session with easy technique or 45-60 minutes Z2.
- The VO2 test is a hard session. On 06-25 and 06-26, stack recovery and easy aerobic work before chasing another interval badge.
- Fuel every ride over 75 minutes. Use 30-60 g carbs/h for 75-150 minutes and practice 60-90 g/h for race rehearsals.[^carb]

### weekly build

| week    |          dates |        load ceiling | Runna   | swim       | bike      | key proof                                          |
| ------- | -------------: | ------------------: | ------- | ---------- | --------- | -------------------------------------------------- |
| reset   | 06-25 to 06-28 |             420-500 | 15 mi   | 3.5-4.5 km | 2.5-3.5 h | absorb VO2 test, keep 400s controlled, long run ok |
| build 2 | 06-29 to 07-05 |             650-720 | 20.0 mi | 6.0-7.0 km | 6.0-7.5 h | tempo 2 miles, 7.5 mi long run, 40 km rehearsal    |
| peak    | 07-06 to 07-12 |             700-760 | 21.8 mi | 7.0-8.0 km | 7.0-8.0 h | 1 km repeats, 9 mi progressive long run, brick run |
| taper 1 | 07-13 to 07-19 |             480-540 | 17.6 mi | 5.0-5.5 km | 4.0-5.0 h | drop set, 6 mi long run, race-kit swim plus bike   |
| race    | 07-20 to 07-26 | 260-340 before race | 13.0 mi | 2.5-3.5 km | 2.0-3.0 h | race-pace half miles, 10 km race, arrive sharp     |

The load ceiling is total weekly load. The Runna mileage is the run lane, so extra jogs do not exist. If the app moves a run, keep the weekly mileage inside the same band and keep at least 24 hours between hard run work and bike race-power work. Most minutes stay easy, then taper volume drops while short intensity stays alive.[^seiler][^taper]

### calendar

| date  | Runna                                                  | swim                                                             | bike                                                        |
| ----- | ------------------------------------------------------ | ---------------------------------------------------------------- | ----------------------------------------------------------- |
| 06-25 | VO2 and DEXA day; Loading Up only if already scheduled | off                                                              | no bike quality; 20-30 min spin only if legs need movement  |
| 06-26 | 4.5 mi Easy Run, HR 121-142                            | S3, 1600-1800 m, no strain                                       | off                                                         |
| 06-27 | 400 m Repeats, 4 mi; Stretch & Stability 1, 25 min     | off                                                              | 45-60 min Z2 only if reps stay controlled                   |
| 06-28 | 6.5 mi Long Run, HR mostly under 145                   | 1500-1800 m continuous or open water                             | 30-45 min recovery spin                                     |
| 06-29 | 4 mi Easy Run                                          | S1, 2000 m                                                       | off                                                         |
| 06-30 | Strength Supersets, 55-65 min                          | off                                                              | B1, 70-75 min with 5x3 min at 245-265 W                     |
| 07-01 | Tempo 2 Miles, 4.5 mi                                  | S2, 2200-2400 m with 4-5x400 m                                   | 30-45 min easy                                              |
| 07-02 | Lower Body Strength Session, 55-65 min                 | off                                                              | B2, 80-90 min with 2x20 min at 180-195 W                    |
| 07-03 | 4 mi Easy Run, HR 121-142                              | S3, 1800-2000 m                                                  | off                                                         |
| 07-04 | Stretch & Stability 2, 25 min                          | off                                                              | B4, 40 km rehearsal at 185-200 W, full fueling              |
| 07-05 | 7.5 mi Long Run                                        | 1600-1800 m easy continuous                                      | 45 min recovery spin if HRV is normal                       |
| 07-06 | 3.25 mi Easy Run                                       | S1, 2200 m                                                       | off                                                         |
| 07-07 | Full Body Endurance Session, 55-65 min                 | off                                                              | B1, 75-80 min with 5x4 min at 245-265 W                     |
| 07-08 | 1 km Repeats, 4.5 mi                                   | S2, 2400-2600 m with 3x600 m                                     | 30-45 min easy                                              |
| 07-09 | Legs & Core Endurance, 55-65 min                       | off                                                              | B2, 90-100 min with 3x15 min at 185-200 W                   |
| 07-10 | 5 mi Brick Run                                         | S3, 1800-2000 m                                                  | 45-60 min race-power primer before brick if legs are normal |
| 07-11 | Stretch & Stability 3, 25 min                          | off or 1000 m easy                                               | B5, 75-90 min at race-power blocks, full fueling            |
| 07-12 | 9 mi Progressive Long Run                              | 1800-2000 m continuous, sighting every 6-8 strokes if open water | 45-60 min easy                                              |
| 07-13 | 3.25 mi Easy Run                                       | S1, 1800 m                                                       | off                                                         |
| 07-14 | Ticking Over, 55-65 min                                | off                                                              | B1, 55-60 min with 4x3 min at 245-260 W                     |
| 07-15 | Drop Set, 3.4 mi                                       | S2, 2000-2200 m with 3x400 m                                     | 30-40 min easy                                              |
| 07-16 | Light(er) Work, 55-65 min                              | off                                                              | B2, 65-75 min with 2x12 min at 185-200 W                    |
| 07-17 | 5 mi Easy Run                                          | S3, 1500-1600 m, stop while stroke still looks good              | off                                                         |
| 07-18 | Stretch & Stability 4, 25 min                          | 1200-1500 m race-kit swim, controlled                            | B4, 25-35 km at 185-200 W, full fueling                     |
| 07-19 | 6 mi Long Run                                          | off or 1000 m easy                                               | 30-45 min recovery spin                                     |
| 07-20 | off                                                    | S1, 1400-1600 m                                                  | off                                                         |
| 07-21 | Hard Work Is Done, 55-65 min                           | off                                                              | B1, 45-50 min with 3x3 min at 240-255 W                     |
| 07-22 | Race Pace Practice Half Miles, 3.5 mi                  | S3, 1200-1500 m with 8x50 m fast                                 | 30-40 min easy                                              |
| 07-23 | off                                                    | off                                                              | B2, 40-45 min with 3x5 min at 185-200 W                     |
| 07-24 | 3.25 mi Easy Run                                       | 1000-1200 m easy, 4x50 m fast                                    | off                                                         |
| 07-25 | Stretch & Stability 5, 25 min                          | 10-15 min water feel if available                                | 20 min spin, 3x30 s openers                                 |
| 07-26 | race: 10 km, first 2 km at 5:05-5:10/km                | race: 1.5 km, settle in the first 300 m                          | race: 40 km, cap first 10 min at 180 W, then 185-205 W      |

### session library

S1, technique plus CSS:

- 300 m easy.
- 8x50 m drill/swim by 25 m, 15-20 s rest.
- Main set starts at 6x100 m at 2:20-2:24/100 m and progresses to 10x100 m at 2:18-2:22/100 m.
- 4x50 m fast at 1:55-2:08/100m, full control.
- 200 m easy.

S2, endurance:

- 400 m easy.
- 3-5x400 m at 2:20-2:28/100 m, 45-60 s rest.
- Progress one rep each build week before adding speed.
- 200 m easy.

S3, speed and stroke:

- 300 m easy.
- 12-20x50 m alternating fast/easy. Fast is 1:55-2:08/100m; easy is 2:30/100m or slower.
- 4x100 m pull or strong-form swim.
- 200 m easy.

B1, VO2:

- 15 min warmup.
- 5-6x3-5 min at 245-275 W, equal easy recovery. HR can touch 171+ late, but the first reps should feel controlled.
- 10-15 min cooldown.
- End the session if power falls below target for two consecutive reps.

B2, tempo and race durability:

- 15 min warmup.
- Progress from 3x10 min at 170-185 W to 3x15 min at 185-200 W and then 3x12 min at 195-210 W.
- Keep cadence 80-90 rpm.
- Cool down until HR is boring again.

B3, long aerobic:

- 2:00-3:00 at 125-160 W.
- Add 20-30 min at 170-185 W only when readiness is normal and HR stays under 145.
- Eat 60-90 g carbs/h and drink to thirst plus sodium in heat.

B4, race rehearsal:

- 10 min easy.
- 40 km at 185-200 W, target 29-30 km/h, no power spikes over 270 W unless traffic forces it.
- Hold aero position whenever safe.
- Practice race bottle, carbs, and the first 10 minutes of restraint.
- If this takes more than 86 minutes in normal wind, the sub-3 budget moves to transitions plus run discipline.

B5, race-power durability:

- 15 min warmup.
- 3x15 min at 185-205 W, 5 min easy between.
- Finish with 10 min easy. If legs are sore at minute 45, choose 185 W before 215 W.

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

### body composition and food

The DEXA math:

- Current: 197.6 lb total, 54.2 lb fat, 135.7 lb lean, 7.8 lb bone, 143.5 lb fat-free mass.
- Current total-mass body fat: 27.4%. Reported tissue-fat view: 28.5%.
- Target: 14% total-mass body fat with fat-free mass preserved.
- Target body mass: about 166.9 lb. Target fat mass: about 23.4 lb. Required fat loss: about 30.8 lb.
- Race-block target: 191-193 lb by 07-26 if readiness, sleep, and workout quality stay green. Do not chase a lower number during taper.

The cut rate:

- Use 0.5-0.7% body mass per week, about 1.0-1.4 lb/week right now. That puts the full 14% target about 22-31 weeks away.[^weightloss]
- Keep the deficit mostly on rest/easy days. Hard sessions, long runs, race rehearsals, and race week are maintenance or near-maintenance.
- If sleep score drops under 75 for two nights, RHR stays 5 bpm over baseline, or intervals flatten, remove the deficit before removing training quality.

Protein:

- Daily floor: 160 g.
- Normal target: 180 g.
- High-deficit or strength-heavy days: 190-200 g.
- Split into 4 feedings of 35-50 g. Put one feeding within 2 hours after hard sessions. Whey is allowed because logistics are real.[^protein]

Carbs:

- Easy/rest day: 180-250 g, mostly around training and dinner.
- Normal training day: 250-350 g.
- Hard, brick, long, or rehearsal day: 350-500 g. Fuel the work, then let the deficit come from the rest of the day.
- During training: 30-60 g/h for 75-150 minutes, 60-90 g/h for race rehearsals and race-day bike. Do not use chronic low-carb during this block; evidence for performance gain from aggressive CHO restriction is weak and it can compromise interval quality.[^cho-period]

Fat and calories:

- Keep fat around 60-80 g/day. Do not drop below 50 g/day repeatedly.
- Use measured total expenditure when available: intake equals total calories minus 300-500 kcal on easy/moderate days, and about total calories on hard/long days.
- Practical floor while training: about 2300 kcal/day. Lower than that is how the plan gets a stupid little stress fracture side quest.
- RMR is 1695 kcal/day, so the useful lever is weekly average intake, not starving a single day.

Race fueling:

- Breakfast 3-4 hours before: 1-2 g/kg carbs, low fiber, familiar sodium.
- Bike: 60-80 g carbs/h, 500-800 mg sodium/h if hot, fluid to thirst.
- Run: 30-45 g carbs/h if tolerated. The run is short enough that stomach calm beats maximal gel cosplay.

### race track

![[https://strava-embeds.com/route/3459615466775941042?style=standard&amp;fullWidth=true&amp;clubId=1459904&amp;fromEmbed=true#ns=94460b8c-1ff3-4475-b2f1-a5beeb3d47e5&amp;hostOrigin=https%3A%2F%2Fsupertri.com&amp;hostPath=%2Ftoronto-triathlon%2Folympic%2F&amp;hostTitle=Supertri+Toronto+Olympic+Triathlon]]

![[https://strava-embeds.com/route/3459617551271887794?style=standard&fullWidth=true&clubId=1459904&fromEmbed=true#ns=ba8ee3c9-ca68-41d1-a539-ca0ed0ea208f&hostOrigin=https%3A%2F%2Fsupertri.com&hostPath=%2Ftoronto-triathlon%2Folympic%2F&hostTitle=Supertri+Toronto+Olympic+Triathlon]]

![[https://strava-embeds.com/route/3459617422094043424?style=standard&amp;fullWidth=true&amp;clubId=1459904&amp;fromEmbed=true#ns=6d5683ba-67f2-4753-b660-9b1e4c7ef5f0&amp;hostOrigin=https%3A%2F%2Fsupertri.com&amp;hostPath=%2Ftoronto-triathlon%2Folympic%2F&amp;hostTitle=Supertri+Toronto+Olympic+Triathlon]]

[^seiler]: Seiler, "What is best practice for training intensity and duration distribution in endurance athletes?", 2010. https://pubmed.ncbi.nlm.nih.gov/20861519/

[^css]: Dekerle et al., "Validity and reliability of critical speed, critical stroke rate and anaerobic swimming capacity in relation to front crawl swimming performances", 2002. https://pubmed.ncbi.nlm.nih.gov/11842355/

[^taper]: Bosquet et al., "Effects of tapering on performance: a meta-analysis", 2007. https://pubmed.ncbi.nlm.nih.gov/17762369/ See also Mujika and Padilla, "Scientific bases for precompetition tapering strategies", 2003. https://pubmed.ncbi.nlm.nih.gov/12840640/

[^carb]: Burke et al., "Carbohydrates for training and competition", 2011. https://pubmed.ncbi.nlm.nih.gov/21660838/

[^weightloss]: Manore, "Weight Management for Athletes and Active Individuals", 2015. https://pmc.ncbi.nlm.nih.gov/articles/PMC4672016/ See also Mountjoy et al., "2023 International Olympic Committee's consensus statement on Relative Energy Deficiency in Sport", 2023. https://bjsm.bmj.com/content/57/17/1073

[^protein]: Jager et al., "International Society of Sports Nutrition Position Stand: protein and exercise", 2017. https://pubmed.ncbi.nlm.nih.gov/28642676/ See also Helms et al., "A systematic review of dietary protein during caloric restriction in resistance trained lean athletes", 2014. https://pubmed.ncbi.nlm.nih.gov/24092765/

[^cho-period]: Gejl and Nybo, "Performance effects of periodized carbohydrate restriction in endurance trained athletes: a systematic review and meta-analysis", 2021. https://link.springer.com/article/10.1186/s12970-021-00435-3

<!-- training plan end -->
