import assert from 'node:assert/strict'
import test from 'node:test'
import {
  garminConnectActivities,
  garminConnectActivity,
  garminConnectActivityStartDate,
  garminConnectClimbSegments,
  garminConnectLactateThresholdHistory,
  garminConnectRunWalk,
  garminConnectRunningLactateThreshold,
  garminConnectSleep,
  garminConnectStreams,
  garminConnectWeightSamples,
} from './garmin-connect'

test('normalizes dated Garmin running lactate threshold history without filling missing days', () => {
  const raw = [
    {
      from: '2026-09-08',
      until: '2026-09-08',
      updatedDate: '2026-09-08',
      series: 'running',
      value: 0.37499895,
    },
    {
      from: '2026-09-03',
      until: '2026-09-03',
      updatedDate: '2026-09-03',
      series: 'running',
      value: 0.37777672,
    },
    { from: '2026-09-05', series: 'cycling', value: 0.5 },
    { from: '2026-09-06', series: 'running', value: null },
    { from: '2026-02-30', series: 'running', value: 0.5 },
  ]
  const pace = garminConnectLactateThresholdHistory(raw, 'speedMps')
  assert.equal(pace.length, 2)
  assert.equal(pace[0].date, '2026-09-03')
  assert.ok(Math.abs(pace[0].value - 3.7777672) < 1e-8)
  assert.equal(pace[1].date, '2026-09-08')
  assert.deepEqual(
    garminConnectLactateThresholdHistory(
      [
        { from: '2026-09-03', updatedDate: '2026-09-03', series: 'running', value: 174 },
        { from: '2026-09-08', updatedDate: '2026-09-08', series: 'running', value: 174 },
      ],
      'heartRateBpm',
    ),
    [
      { date: '2026-09-03', value: 174 },
      { date: '2026-09-08', value: 174 },
    ],
  )
  assert.deepEqual(garminConnectLactateThresholdHistory([], 'speedMps'), [])
  assert.throws(() => garminConnectLactateThresholdHistory({}, 'speedMps'), /Invalid Garmin/)
})

test('normalizes Garmin running lactate threshold units and ignores cycling heart rate', () => {
  const threshold = garminConnectRunningLactateThreshold([
    { calendarDate: '2026-09-08T16:00:53.119', speed: 0.37499895, hearRate: null },
    { calendarDate: '2026-09-08T16:00:53.119', speed: null, hearRate: 174 },
    {
      calendarDate: '2026-09-09T12:00:00',
      heartRateCycling: 166,
      rowSpeed: 0.5,
      heartRateRowing: 160,
    },
  ])
  assert.ok(threshold?.speedMps)
  assert.ok(Math.abs(threshold.speedMps.value - 3.7499895) < 1e-9)
  assert.equal(Math.round(1609.344 / threshold.speedMps.value), 429)
  assert.equal(threshold.speedMps.date, '2026-09-08')
  assert.deepEqual(threshold.heartRateBpm, { value: 174, date: '2026-09-08' })
})

test('selects the latest Garmin threshold independently for each metric and time of day', () => {
  assert.deepEqual(
    garminConnectRunningLactateThreshold([
      { calendarDate: '2026-09-08T16:00:00', speed: 0.4 },
      { calendarDate: '2026-09-07T10:00:00', heartRate: 175 },
      { calendarDate: '2026-09-08T08:00:00', speed: 0.3 },
      { calendarDate: '2026-09-06T10:00:00', hearRate: 173 },
    ]),
    {
      speedMps: { value: 4, date: '2026-09-08' },
      heartRateBpm: { value: 175, date: '2026-09-07' },
    },
  )
})

test('missing and invalid Garmin threshold values remain unavailable', () => {
  assert.equal(garminConnectRunningLactateThreshold([]), null)
  assert.equal(
    garminConnectRunningLactateThreshold([
      null,
      { calendarDate: 'invalid', speed: 0.4, hearRate: 175 },
      { calendarDate: '2026-02-30', speed: 0.4, hearRate: 175 },
      { calendarDate: '2026-09-08', speed: 0, hearRate: -1 },
      { calendarDate: '2026-09-08', speed: Infinity, hearRate: NaN },
      { calendarDate: '2026-09-08', heartRateCycling: 170 },
    ]),
    null,
  )
  assert.deepEqual(
    garminConnectRunningLactateThreshold([{ calendarDate: '2026-09-08', heartRate: 174 }]),
    { speedMps: null, heartRateBpm: { value: 174, date: '2026-09-08' } },
  )
  assert.throws(
    () => garminConnectRunningLactateThreshold({ error: 'unexpected' }),
    /Invalid Garmin/,
  )
})

// Garmin Connect dailySleepData response, 2026-09-08, with identity fields removed.
const overnightSleep = {
  dailySleepDTO: {
    calendarDate: '2026-09-08',
    sleepStartTimestampGMT: 1788845456000,
    sleepEndTimestampGMT: 1788875809000,
    sleepFromDevice: true,
    sleepWindowConfirmationType: 'enhanced_confirmed_final',
    averageSpO2Value: 97,
    lowestSpO2Value: 91,
    averageRespirationValue: 15,
    lowestRespirationValue: 11,
    highestRespirationValue: 19,
    avgSleepStress: 11,
    deepSleepSeconds: 2340,
    remSleepSeconds: 7320,
    lightSleepSeconds: 20640,
  },
  sleepBodyBattery: [
    { value: 64, startGMT: 1788845400000 },
    { value: 100, startGMT: 1788875640000 },
  ],
  bodyBatteryChange: 36,
  restlessMomentsCount: 39,
}

test('extracts Garmin overnight physiology without copying sleep stages', () => {
  assert.deepEqual(garminConnectSleep(overnightSleep, '2026-09-08'), {
    source: 'garmin',
    date: '2026-09-08',
    startTime: '2026-09-08T05:30:56.000Z',
    endTime: '2026-09-08T13:56:49.000Z',
    averageBreathsPerMinute: 15,
    lowestBreathsPerMinute: 11,
    highestBreathsPerMinute: 19,
    averageSpO2: 97,
    lowestSpO2: 91,
    bodyBatteryStart: 64,
    bodyBatteryEnd: 100,
    bodyBatteryChange: 36,
    averageStress: 11,
    restlessMoments: 39,
  })
})

test('preserves native overnight respiration timestamps and the clipped first interval', () => {
  const summary = garminConnectSleep(
    {
      ...overnightSleep,
      wellnessEpochRespirationDataDTOList: [
        { startTimeGMT: 1788845456000, respirationValue: 19 },
        { startTimeGMT: 1788845520000, respirationValue: 17 },
        { startTimeGMT: 1788845640000, respirationValue: 18 },
        { startTimeGMT: 1788845760000, respirationValue: 19 },
        { startTimeGMT: 1788845880000, respirationValue: 18 },
      ],
    },
    '2026-09-08',
  )
  assert.deepEqual(summary?.respiration, [
    { timestamp: 1788845456000, breathsPerMinute: 19 },
    { timestamp: 1788845520000, breathsPerMinute: 17 },
    { timestamp: 1788845640000, breathsPerMinute: 18 },
    { timestamp: 1788845760000, breathsPerMinute: 19 },
    { timestamp: 1788845880000, breathsPerMinute: 18 },
  ])
})

test('normalizes respiration gaps, invalid values, and repeated timestamps within the sleep window', () => {
  const start = overnightSleep.dailySleepDTO.sleepStartTimestampGMT
  const end = overnightSleep.dailySleepDTO.sleepEndTimestampGMT
  const summary = garminConnectSleep(
    {
      ...overnightSleep,
      wellnessEpochRespirationDataDTOList: [
        { startTimeGMT: start + 20 * 60_000, respirationValue: 13 },
        { startTimeGMT: start - 1, respirationValue: 17 },
        { startTimeGMT: start, respirationValue: 15 },
        { startTimeGMT: start, respirationValue: -2 },
        { startTimeGMT: start + 120_000, respirationValue: -1 },
        { startTimeGMT: start + 240_000, respirationValue: 0 },
        { startTimeGMT: start + 360_000, respirationValue: null },
        { startTimeGMT: start + 480_000, respirationValue: NaN },
        { startTimeGMT: start + 600_000, respirationValue: Infinity },
        { startTimeGMT: start + 720_000, respirationValue: '16' },
        { startTimeGMT: end + 1, respirationValue: 17 },
        { startTimeGMT: NaN, respirationValue: 17 },
        { respirationValue: 17 },
        null,
      ],
    },
    '2026-09-08',
  )
  assert.deepEqual(summary?.respiration, [
    { timestamp: start, breathsPerMinute: 15 },
    { timestamp: start + 120_000, breathsPerMinute: null },
    { timestamp: start + 240_000, breathsPerMinute: null },
    { timestamp: start + 360_000, breathsPerMinute: null },
    { timestamp: start + 480_000, breathsPerMinute: null },
    { timestamp: start + 600_000, breathsPerMinute: null },
    { timestamp: start + 720_000, breathsPerMinute: null },
    { timestamp: start + 20 * 60_000, breathsPerMinute: 13 },
  ])
})

test('keeps native respiration without a summary average and omits unusable series', () => {
  const dto = {
    calendarDate: '2026-09-08',
    sleepStartTimestampGMT: 1788845456000,
    sleepEndTimestampGMT: 1788875809000,
    sleepFromDevice: true,
  }
  const summary = garminConnectSleep(
    {
      dailySleepDTO: dto,
      wellnessEpochRespirationDataDTOList: [{ startTimeGMT: 1788845456000, respirationValue: 15 }],
    },
    '2026-09-08',
  )
  assert.equal(summary?.averageBreathsPerMinute, null)
  assert.deepEqual(summary?.respiration, [{ timestamp: 1788845456000, breathsPerMinute: 15 }])
  assert.equal(
    garminConnectSleep(
      {
        dailySleepDTO: dto,
        wellnessEpochRespirationDataDTOList: [
          { startTimeGMT: 1788845456000, respirationValue: -2 },
        ],
      },
      '2026-09-08',
    ),
    null,
  )
  assert.equal(garminConnectSleep(overnightSleep, '2026-09-08')?.respiration, undefined)
  assert.equal(
    garminConnectSleep(
      {
        ...overnightSleep,
        dailySleepDTO: { ...dto, sleepEndTimestampGMT: 0, averageRespirationValue: 15 },
        wellnessEpochRespirationDataDTOList: [
          { startTimeGMT: 1788845456000, respirationValue: 15 },
        ],
      },
      '2026-09-08',
    )?.respiration,
    undefined,
  )
})

test('preserves validated provider clock offsets without treating an offset as sleep physiology', () => {
  const utcStart = overnightSleep.dailySleepDTO.sleepStartTimestampGMT
  const utcEnd = overnightSleep.dailySleepDTO.sleepEndTimestampGMT
  const withOffset = (minutes: number, endMinutes = minutes) =>
    garminConnectSleep(
      {
        ...overnightSleep,
        dailySleepDTO: {
          ...overnightSleep.dailySleepDTO,
          sleepStartTimestampLocal: utcStart + minutes * 60_000,
          sleepEndTimestampLocal: utcEnd + endMinutes * 60_000,
        },
      },
      '2026-09-08',
    )
  assert.equal(withOffset(-240)?.utcOffsetMinutes, -240)
  assert.equal(withOffset(0)?.utcOffsetMinutes, 0)
  assert.equal(withOffset(16 * 60)?.utcOffsetMinutes, undefined)
  assert.equal(withOffset(0.5)?.utcOffsetMinutes, undefined)
  assert.equal(withOffset(-240, -300)?.utcOffsetMinutes, undefined)
  assert.equal(
    garminConnectSleep(
      {
        dailySleepDTO: {
          calendarDate: '2026-09-08',
          sleepStartTimestampGMT: utcStart,
          sleepStartTimestampLocal: utcStart - 240 * 60_000,
          sleepEndTimestampGMT: utcEnd,
        },
      },
      '2026-09-08',
    ),
    null,
  )
})

test('keeps valid zero stress, restlessness, and battery readings while rejecting sentinel values', () => {
  const summary = garminConnectSleep(
    {
      ...overnightSleep,
      dailySleepDTO: {
        ...overnightSleep.dailySleepDTO,
        averageRespirationValue: 0,
        lowestRespirationValue: -2,
        highestRespirationValue: Infinity,
        averageSpO2Value: 0,
        lowestSpO2Value: 101,
        avgSleepStress: 0,
      },
      sleepBodyBattery: [{ startGMT: 1788845400000, value: 0 }],
      bodyBatteryChange: 0,
      restlessMomentsCount: 0,
    },
    '2026-09-08',
  )
  assert.equal(summary?.averageBreathsPerMinute, null)
  assert.equal(summary?.lowestBreathsPerMinute, null)
  assert.equal(summary?.highestBreathsPerMinute, null)
  assert.equal(summary?.averageSpO2, null)
  assert.equal(summary?.lowestSpO2, null)
  assert.equal(summary?.bodyBatteryStart, 0)
  assert.equal(summary?.bodyBatteryEnd, null)
  assert.equal(summary?.bodyBatteryChange, 0)
  assert.equal(summary?.averageStress, 0)
  assert.equal(summary?.restlessMoments, 0)
})

test('requires measured body battery near each sleep boundary and keeps native overnight change', () => {
  const summary = garminConnectSleep(
    {
      ...overnightSleep,
      sleepBodyBattery: [
        { startGMT: 1788875640000, value: 100 },
        { startGMT: 1788850000000, value: 70 },
        { startGMT: 1788845400000, value: -1 },
      ],
      bodyBatteryChange: -3,
      bodyBatteryChargedValue: 90,
      bodyBatteryDrainedValue: 40,
    },
    '2026-09-08',
  )
  assert.equal(summary?.bodyBatteryStart, null)
  assert.equal(summary?.bodyBatteryEnd, 100)
  assert.equal(summary?.bodyBatteryChange, -3)
})

test('does not derive overnight charge from daily body battery or partial endpoint samples', () => {
  const summary = garminConnectSleep(
    { ...overnightSleep, bodyBatteryChange: null, bodyBatteryChargedValue: 50 },
    '2026-09-08',
  )
  assert.equal(summary?.bodyBatteryChange, null)
})

test('ignores empty and manually imported sleep without relabeling Oura stages as Garmin', () => {
  assert.equal(garminConnectSleep(null, '2026-09-08'), null)
  assert.equal(garminConnectSleep({ dailySleepDTO: null }, '2026-09-08'), null)
  assert.equal(
    garminConnectSleep(
      { dailySleepDTO: { calendarDate: '2026-09-08', deepSleepSeconds: 2340 } },
      '2026-09-08',
    ),
    null,
  )
  assert.equal(
    garminConnectSleep(
      {
        ...overnightSleep,
        dailySleepDTO: { ...overnightSleep.dailySleepDTO, sleepFromDevice: false },
      },
      '2026-09-08',
    ),
    null,
  )
})

test('rejects the wrong wake-day and invalid windows independently of valid physiology', () => {
  assert.throws(() => garminConnectSleep(overnightSleep, '2026-09-07'), /does not match/)
  const summary = garminConnectSleep(
    {
      ...overnightSleep,
      dailySleepDTO: {
        ...overnightSleep.dailySleepDTO,
        sleepEndTimestampGMT: 1788840000000,
        averageRespirationValue: NaN,
      },
    },
    '2026-09-08',
  )
  assert.equal(summary?.startTime, null)
  assert.equal(summary?.endTime, null)
  assert.equal(summary?.bodyBatteryStart, null)
  assert.equal(summary?.bodyBatteryEnd, null)
  assert.equal(summary?.averageBreathsPerMinute, null)
  assert.equal(summary?.averageSpO2, 97)
})

test('normalizes Garmin GraphQL numeric activity timestamps', () => {
  const record = {
    activityId: 123,
    beginTimestamp: 1_788_209_285_000,
    startTimeGMT: '2026-08-31 20:48:05',
    distance: 36_127,
  }

  assert.equal(garminConnectActivityStartDate(record), '2026-08-31T20:48:05.000Z')
  assert.equal(garminConnectActivity(null, record, 0)?.startDate, '2026-08-31T20:48:05.000Z')
})

test('normalizes Garmin Connect activity details into the Garmin cache shape', () => {
  const raw = {
    activities: [
      {
        activityId: 123,
        activityName: 'Morning ride',
        activityType: { typeKey: 'road_biking' },
        startTimeGMT: '2026-06-01 12:00:00',
      },
    ],
  }
  const items = garminConnectActivities(raw)
  const activity = garminConnectActivity(
    {
      activityId: 123,
      activityName: 'Morning ride',
      activityType: { typeKey: 'road_biking' },
      startTimeGMT: '2026-06-01 12:00:00',
      startTimeLocal: '2026-06-01 08:00:00',
      distance: 48_200,
      duration: 7500,
      movingDuration: 7200,
      calories: 1403,
      averageHR: 135,
      maxHR: 168,
      averagePower: 108,
      normalizedPower: 151,
      maxPower: 530,
      averageBikeCadence: 82,
      elevationGain: 430,
      elevationLoss: 421,
      kilojoules: 775.4,
      trainingStressScore: 67.8,
      intensityFactor: 0.713,
      summaryDTO: {
        trainingEffect: 4.5,
        anaerobicTrainingEffect: 0,
        activityTrainingLoad: 301.7034912109375,
        trainingEffectLabel: 'AEROBIC_BASE',
        aerobicTrainingEffectMessage: 'HIGHLY_IMPROVING_AEROBIC_ENDURANCE_10',
        anaerobicTrainingEffectMessage: 'NO_ANAEROBIC_BENEFIT_0',
      },
      deviceName: 'Edge 1050',
    },
    items[0].record,
    0,
  )

  assert.equal(items.length, 1)
  assert.equal(activity?.id, 'connect:123')
  assert.equal(activity?.sport, 'bike')
  assert.equal(activity?.startDate, '2026-06-01T12:00:00.000Z')
  assert.equal(activity?.startDateLocal, '2026-06-01T08:00:00')
  assert.equal(activity?.distanceM, 48_200)
  assert.equal(activity?.movingTimeS, 7200)
  assert.equal(activity?.elapsedTimeS, 7500)
  assert.equal(activity?.sourceDevice, 'Edge 1050')
  assert.deepEqual(activity?.metrics, {
    totalCalories: 1403,
    metabolicCalories: null,
    avgHeartRate: 135,
    maxHeartRate: 168,
    avgPower: 108,
    normalizedPower: 151,
    maxPower: 530,
    avgCadence: 82,
    totalAscentM: 430,
    totalDescentM: 421,
    totalWorkKJ: 775.4,
    trainingStressScore: 67.8,
    intensityFactor: 0.713,
    aerobicTrainingEffect: 4.5,
    anaerobicTrainingEffect: 0,
    exerciseLoad: 301.7,
    trainingEffectLabel: 'AEROBIC_BASE',
    aerobicTrainingEffectMessage: 'HIGHLY_IMPROVING_AEROBIC_ENDURANCE_10',
    anaerobicTrainingEffectMessage: 'NO_ANAEROBIC_BENEFIT_0',
  })
})

test('prefers elapsed duration over timer duration', () => {
  const fallback = {
    activityId: 124,
    activityName: 'Race swim',
    activityType: { typeKey: 'open_water_swimming' },
    startTimeGMT: '2026-07-26 12:43:52',
  }
  const activity = garminConnectActivity(
    {
      ...fallback,
      distance: 1_500,
      duration: 2_460,
      movingDuration: 2_460,
      elapsedDuration: 2_468,
    },
    fallback,
    0,
  )

  assert.equal(activity?.movingTimeS, 2_460)
  assert.equal(activity?.elapsedTimeS, 2_468)
})

test('preserves canonical Garmin running-dynamics summary values', () => {
  const fallback = {
    activityId: 24227986086,
    activityName: 'Easy Miles',
    activityType: { typeKey: 'running' },
    startTimeGMT: '2026-09-03 19:38:48',
  }
  const activity = garminConnectActivity(
    {
      ...fallback,
      summaryDTO: {
        avgRespirationRate: 35.150001525878906,
        strideLength: 108.21999511718751,
        verticalRatio: 11.300000190734863,
        verticalOscillation: 12.380000305175782,
        groundContactBalanceLeft: 49.2599983215332,
        groundContactTime: 246.5,
        stepSpeedLoss: 0.07900000095367432,
        stepSpeedLossPercent: 2.7799999713897705,
        impactLoad: 5_320,
      },
    },
    fallback,
    0,
  )

  assert.deepEqual(activity?.runningDynamics, {
    source: 'garmin',
    averageRespirationRate: 35.15,
    averageStrideLengthCm: 108.22,
    averageVerticalRatioPct: 11.3,
    averageVerticalOscillationCm: 12.38,
    averageGroundContactBalanceLeftPct: 49.26,
    averageGroundContactTimeMs: 246.5,
    averageStepSpeedLossMps: 0.079,
    averageStepSpeedLossPct: 2.78,
    impactLoadM: 5_320,
  })
})

test('keeps Garmin Connect fueling fields when they appear in nested records', () => {
  const fallback = {
    activityId: 456,
    activityName: 'Long ride',
    activityType: { typeKey: 'cycling' },
    startTimeGMT: '2026-06-07 11:29:55',
    distance: 96_000,
    duration: 14_400,
  }
  const activity = garminConnectActivity(
    {
      activityDetailDTO: fallback,
      nutrition: { caloriesConsumed: 520, carbsConsumedG: 74 },
      hydration: { fluidConsumedMl: 1180, sweatLossMl: 2100 },
    },
    fallback,
    0,
  )

  assert.deepEqual(activity?.fueling, {
    caloriesConsumed: 520,
    carbsConsumedG: 74,
    fluidMl: 1180,
    carbsRecommendedG: null,
    fluidRecommendedMl: null,
    sweatLossMl: 2100,
    sourceDevice: null,
  })
})

test('deduplicates Garmin Connect list records by activity id', () => {
  const items = garminConnectActivities({
    activities: [
      { activityId: 1, startTimeGMT: '2026-06-01 12:00:00' },
      { activityIdStr: '1', startTimeGMT: '2026-06-01 12:00:00' },
      { activityId: 2, startTimeGMT: '2026-06-02 12:00:00' },
    ],
  })

  assert.deepEqual(
    items.map(item => item.id),
    ['1', '2'],
  )
})

test('normalizes Garmin Connect GraphQL activity scalars', () => {
  const items = garminConnectActivities({
    data: {
      searchActivitiesScalar: [
        { activityId: 3, startTimeGMT: '2026-06-03 12:00:00' },
        { activityId: 4, startTimeGMT: '2026-06-04 12:00:00' },
      ],
    },
  })

  assert.deepEqual(
    items.map(item => item.id),
    ['3', '4'],
  )
})

test('normalizes Garmin Connect weight samples across dayview aliases', () => {
  const samples = garminConnectWeightSamples({
    dailyWeightSummaries: [
      {
        summaryDate: '2026-06-18',
        weightMetrics: [
          {
            samplePk: 1_781_826_649,
            weightInGrams: 91_130,
            bodyMassIndex: 25.8,
            bodyFatPercentage: 22.2,
            bodyWaterPercentage: 56.8,
            muscleMassInGrams: 37_750,
            boneMassInGrams: 6_240,
          },
        ],
      },
    ],
  })

  assert.deepEqual(samples, [
    {
      ts: 1_781_826_649_000,
      date: '2026-06-18',
      weightKg: 91.13,
      bmi: 25.8,
      bodyFatPct: 22.2,
      bodyWaterPct: 56.8,
      muscleMassKg: 37.75,
      boneMassKg: 6.24,
    },
  ])
})

test('normalizes Garmin Connect weight records with timestamp strings and pound units', () => {
  const samples = garminConnectWeightSamples({
    weightMetrics: [
      {
        weighInTimestampGMT: '2026-06-19 11:03:00',
        weight: 200.6,
        weightUnit: 'pounds',
        bodyFatPercent: 22.1,
        bodyWaterPct: 56.7,
        muscleMassKg: 37.8,
        boneMassKg: 6.2,
      },
    ],
  })

  assert.equal(samples.length, 1)
  assert.equal(samples[0].date, '2026-06-19')
  assert.equal(samples[0].weightKg, 90.99)
  assert.equal(samples[0].bodyFatPct, 22.1)
  assert.equal(samples[0].bodyWaterPct, 56.7)
  assert.equal(samples[0].muscleMassKg, 37.8)
  assert.equal(samples[0].boneMassKg, 6.2)
})

test('normalizes Garmin ClimbPro parent splits and excludes their sections', () => {
  const climbs = garminConnectClimbSegments({
    splits: [
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB',
        startTimeGMT: '2026-07-09T22:12:18.0',
        endTimeGMT: '2026-07-09T22:15:24.0',
        distance: 1045.82,
        duration: 187.552,
        movingDuration: 186,
        elapsedDuration: 187.552,
        elevationGain: 29,
        elevationLoss: 3,
        startElevation: 81.8,
        averageGrade: 2.730000019,
        maxGrade: 8.739999771,
        averageSpeed: 5.576000213,
        averageHR: 155,
        maxHR: 165,
        averagePower: 225,
        normalizedPower: 262,
        maxPower: 451,
        averageBikeCadence: 80,
        climbProDifficulty: 'NONE',
      },
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB_SECTION',
        startTimeGMT: '2026-07-09T22:06:26.0',
        endTimeGMT: '2026-07-09T22:07:28.0',
        distance: 268.51,
        duration: 62,
        elevationGain: 18.2,
        averageGrade: 7.59,
        averagePower: 338.24,
        climbProDifficulty: 'STEEP',
      },
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB',
        startTimeGMT: '2026-07-09T22:06:14.0',
        endTimeGMT: '2026-07-09T22:09:34.0',
        distance: 1067.37,
        duration: 200.14,
        movingDuration: 200,
        elapsedDuration: 200.14,
        elevationGain: 27,
        elevationLoss: 4,
        startElevation: 83.8,
        averageGrade: 2.569999933,
        maxGrade: 9.029999733,
        averageSpeed: 5.333000183,
        averageHR: 150,
        maxHR: 162,
        averagePower: 203,
        normalizedPower: 257,
        maxPower: 473,
        averageBikeCadence: 79,
        climbProDifficulty: 'NONE',
      },
      {
        type: 'SURFACE_TYPE_PAVED',
        startTimeGMT: '2026-07-09T21:38:32.0',
        endTimeGMT: '2026-07-09T22:57:35.0',
        distance: 29298.34,
        duration: 4134.357,
      },
      {
        type: 'CLIMB_PRO_CYCLING_CLIMB',
        startTimeGMT: '2026-07-09T23:00:00.0',
        endTimeGMT: '2026-07-09T23:01:00.0',
        distance: 300,
      },
    ],
  })

  assert.deepEqual(climbs, [
    {
      startDate: '2026-07-09T22:06:14.000Z',
      endDate: '2026-07-09T22:09:34.000Z',
      distanceM: 1067.37,
      durationS: 200.14,
      movingTimeS: 200,
      elapsedTimeS: 200.14,
      elevationGainM: 27,
      elevationLossM: 4,
      startElevationM: 83.8,
      avgGradePct: 2.57,
      maxGradePct: 9.03,
      avgSpeedMps: 5.333,
      avgHeartRate: 150,
      maxHeartRate: 162,
      avgPower: 203,
      normalizedPower: 257,
      maxPower: 473,
      avgCadence: 79,
      difficulty: 'NONE',
    },
    {
      startDate: '2026-07-09T22:12:18.000Z',
      endDate: '2026-07-09T22:15:24.000Z',
      distanceM: 1045.82,
      durationS: 187.552,
      movingTimeS: 186,
      elapsedTimeS: 187.552,
      elevationGainM: 29,
      elevationLossM: 3,
      startElevationM: 81.8,
      avgGradePct: 2.73,
      maxGradePct: 8.74,
      avgSpeedMps: 5.576,
      avgHeartRate: 155,
      maxHeartRate: 165,
      avgPower: 225,
      normalizedPower: 262,
      maxPower: 451,
      avgCadence: 80,
      difficulty: 'NONE',
    },
  ])
})

test('normalizes Garmin Connect detail metrics into streams', () => {
  const streams = garminConnectStreams({
    metricDescriptors: [
      { metricsIndex: 7, key: 'sumElapsedDuration' },
      {
        metricsIndex: 11,
        key: 'connectIQDeveloperField-07',
        appID: '6957fe68-83fe-4ed6-8613-413f70624bb5',
        developerFieldNumber: 0,
      },
      { metricsIndex: 0, key: 'sumDistance' },
      { metricsIndex: 1, key: 'directLatitude' },
      {
        metricsIndex: 10,
        key: 'connectIQDeveloperField-11',
        appID: '6957fe68-83fe-4ed6-8613-413f70624bb5',
        developerFieldNumber: 95,
      },
      { metricsIndex: 2, key: 'directLongitude' },
      { metricsIndex: 3, key: 'directElevation' },
      { metricsIndex: 4, key: 'directPower' },
      { metricsIndex: 15, key: 'directRightBalance' },
      { metricsIndex: 13, key: 'directAvailableStamina' },
      { metricsIndex: 14, key: 'directPotentialStamina' },
      { metricsIndex: 5, key: 'directHeartRate' },
      {
        metricsIndex: 12,
        key: 'connectIQDeveloperField-08',
        appID: '6957fe68-83fe-4ed6-8613-413f70624bb5',
        developerFieldNumber: 10,
      },
      { metricsIndex: 6, key: 'directBikeCadence' },
      { metricsIndex: 8, key: 'directRespirationRate' },
      { metricsIndex: 16, key: 'directSaturatedHemoglobinPercent' },
      {
        metricsIndex: 9,
        key: 'connectIQDeveloperField-12',
        appID: '6957fe68-83fe-4ed6-8613-413f70624bb5',
        developerFieldNumber: 81,
      },
    ],
    activityDetailMetrics: [
      {
        metrics: [0, 43.1, -79.1, 101, 120, 135, 82, 0, null, 98.6, 0, 37, null, 100, 100, 48, 64],
      },
      {
        metrics: [
          500, 43.2, -79.2, 104, 180, 142, 88, 15, 27.42, 99.5, 1.4, 37.01, 33.45, 78, 86, 49.5, 61,
        ],
      },
      {
        metrics: [
          1000, 43.3, -79.3, 109, 210, 149, 91, 30, 31.08, 100.4, 3, 37.03, 33.5, 60, 71, 51, 58,
        ],
      },
    ],
  })

  assert.deepEqual(streams?.latlng, [
    [43.1, -79.1],
    [43.2, -79.2],
    [43.3, -79.3],
  ])
  assert.deepEqual(streams?.distance, [0, 500, 1000])
  assert.deepEqual(streams?.altitude, [101, 104, 109])
  assert.deepEqual(streams?.watts, [120, 180, 210])
  assert.deepEqual(streams?.rightBalance, [48, 49.5, 51])
  assert.deepEqual(streams?.heartrate, [135, 142, 149])
  assert.deepEqual(streams?.cadence, [82, 88, 91])
  assert.deepEqual(streams?.time, [0, 15, 30])
  assert.deepEqual(streams?.stamina, [100, 78, 60])
  assert.deepEqual(streams?.potentialStamina, [100, 86, 71])
  assert.deepEqual(streams?.respiration, [0, 27.42, 31.08])
  assert.deepEqual(streams?.muscleOxygenPercent, [64, 61, 58])
  assert.deepEqual(streams?.heatStrainIndex, [0, 1.4, 3])
  assert.deepEqual(streams?.coreTemperatureC, [37, 37.01, 37.03])
  assert.deepEqual(streams?.skinTemperatureC, [-1, 33.45, 33.5])
})

test('preserves nullable native Forerunner running dynamics', () => {
  const streams = garminConnectStreams({
    metricDescriptors: [
      { metricsIndex: 0, key: 'sumElapsedDuration' },
      { metricsIndex: 1, key: 'directPerformanceCondition' },
      { metricsIndex: 2, key: 'directStrideLength' },
      { metricsIndex: 3, key: 'directVerticalRatio' },
      { metricsIndex: 4, key: 'directVerticalOscillation' },
      { metricsIndex: 5, key: 'directGroundContactBalanceLeft' },
      { metricsIndex: 6, key: 'directGroundContactTime' },
      { metricsIndex: 7, key: 'directStepSpeedLoss' },
      { metricsIndex: 8, key: 'directStepSpeedLossPercent' },
      { metricsIndex: 9, key: 'directImpactLoadFactor' },
    ],
    activityDetailMetrics: [
      { metrics: [0, null, null, null, null, null, null, null, null, 0] },
      { metrics: [15, -4, 108, 11.3, 12.4, 49.3, 246.5, 0.079, 2.78, 1] },
      { metrics: [30, -10, 77, 16.1, 12.5, 50.1, 248, 0.07, 2.5, 0.96] },
    ],
  })

  assert.deepEqual(streams?.performanceCondition, [null, -4, -10])
  assert.deepEqual(streams?.strideLengthCm, [null, 108, 77])
  assert.deepEqual(streams?.verticalRatioPct, [null, 11.3, 16.1])
  assert.deepEqual(streams?.verticalOscillationCm, [null, 12.4, 12.5])
  assert.deepEqual(streams?.groundContactBalanceLeftPct, [null, 49.3, 50.1])
  assert.deepEqual(streams?.groundContactTimeMs, [null, 246.5, 248])
  assert.deepEqual(streams?.stepSpeedLossMps, [null, 0.079, 0.07])
  assert.deepEqual(streams?.stepSpeedLossPct, [null, 2.78, 2.5])
  assert.deepEqual(streams?.impactLoadFactor, [0, 1, 0.96])
})

test('normalizes positive Garmin run/walk intervals onto the active elapsed axis', () => {
  assert.deepEqual(
    garminConnectRunWalk({
      splits: [
        { type: 'RWD_RUN', duration: 10 },
        { type: 'RWD_WALK', duration: 2.345 },
        { type: 'RWD_STAND', duration: 0 },
        { type: 'RWD_STAND', duration: 0.655 },
        { type: 'RWD_RUN', duration: 5.5 },
        { type: 'SURFACE_TYPE_PAVED', duration: 18.5 },
      ],
    }),
    {
      source: 'garmin',
      elapsedTimeS: 18.5,
      runTimeS: 15.5,
      walkTimeS: 2.345,
      idleTimeS: 0.655,
      segments: [
        { state: 'run', startElapsedS: 0, endElapsedS: 10 },
        { state: 'walk', startElapsedS: 10, endElapsedS: 12.345 },
        { state: 'idle', startElapsedS: 12.345, endElapsedS: 13 },
        { state: 'run', startElapsedS: 13, endElapsedS: 18.5 },
      ],
    },
  )
})
