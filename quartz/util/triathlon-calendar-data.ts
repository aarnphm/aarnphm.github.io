import type { TriathlonCalendarDetails, TriathlonCalendarEvent } from './triathlon-calendar'

interface TriathlonCalendarMetadata extends Omit<
  TriathlonCalendarEvent,
  'id' | 'url' | 'series' | 'details' | 'results' | 'participated'
> {
  recordKey?: string
  /** Fixed details for an archived edition, independent of the current organiser page. */
  details?: TriathlonCalendarDetails
}

interface TriathlonCalendarSeason {
  checked: string
  sources: Readonly<Record<string, TriathlonCalendarMetadata>>
}

export const TRIATHLON_CALENDAR_CATALOG: Readonly<Record<number, TriathlonCalendarSeason>> = {
  2026: {
    checked: '2026-09-30',
    sources: {
      'https://bikeforbrainhealth.ca/': {
        name: 'Bike for Brain Health',
        location: 'Toronto, Canada',
        kind: 'cycling',
        format: '60 km ride',
        date: '2026-05-31',
        endDate: null,
        note: null,
        recordKey: 'b4bh--tor--26',
        details: {
          sources: [
            'https://bikeforbrainhealth.ca/body-break/',
            'https://www.tbn.ca/event-6351867',
            'https://www.strava.com/activities/18778156664',
          ],
          fetched: '2026-09-30',
          venue: 'Aga Khan Museum, 77 Wynford Drive, Toronto, Canada',
          venueMap: null,
          athleteGuide: null,
          terrain: {},
          airC: null,
          waterC: null,
          course: [
            {
              leg: 'bike',
              title: 'B4BH 2026 · Strava recording',
              source: 'https://www.strava.com/activities/18778156664',
              edition: 2026,
              mapUrl: null,
              gpxUrl: '/triathlon/routes/b4bh-toronto-2026.gpx',
              distanceM: 59225.1,
              raceDistanceM: 60000,
              laps: null,
              aidStationsPerLap: null,
              elevationGainM: 468,
              polyline:
                '{p{iGrcecNyRhAcc@~Hod@`@i[fE{`@xBaWtE_HoDeENzEfS|HqKjXcG|]gBx]gFn\\Czk@qGzn@}BtM{D~]iUrNFlFhDnSrXd]`InIjIvHbVfK~OpBfHl@xLoBva@nBnMjEvHla@~d@fM~ElTxA`JnDtE|IbIvEzNfWt@vCuAbBeAq@[eFpFgHnBb@MzC}E`AaNyU{HyE[yDhZaBnRoD`JcEtVcRdbAcS`d@}`@|GFvFvEfEvODj`@pIh_@lNleAjVpn@dF`VjKjfAQ~_@dCd~@x\\pmBRpKsAnMoW|x@{ElTmE~l@NjUbDt[lTbu@H`CgAzBhC_CT_GaFsGoI_W}EwU}ByVGcYlEwi@xDkQlWkx@`BoL?qM_]ioBeC__AH_`@eKsdAmKcc@gQ_b@_OohAiIa_@Bo^wFkSwEyDcGFoe@jb@gbApSgVtQ_LtEoPxCgW|@iSsGkRoAoI}BeI}E{MiOaPqSyCcHgAqMfBm_@m@uKmCoIqKmPoHoU{I{Ie\\uH{Uo[cH{DkMZiYtTgSjGog@zAyq@|Hi]Fw\\tEi[|A{[jF_GoD{E?pElSjI{KvW}Fx`@uBb]uE|_@Qbf@cGpUBvAbBHy@',
              profile: [
                138, 106, 131, 145, 159, 137, 126, 110, 135, 126, 112, 92, 100, 101, 89, 86, 82, 83,
                89, 77, 77, 82, 89, 83, 84, 82, 89, 87, 78, 71, 74, 76, 78, 79, 77, 76, 76, 82, 91,
                91, 82, 84, 83, 84, 88, 71, 75, 85, 89, 86, 96, 100, 94, 115, 127, 133, 106, 128,
                140, 160, 141, 130, 108, 139,
              ],
              aidM: [],
            },
          ],
          schedule: null,
          schedulePending: false,
          kept: {},
        },
      },
      'https://supertri.com/toronto-triathlon/': {
        name: 'Supertri Toronto',
        location: 'Toronto, Canada',
        kind: 'triathlon',
        format: 'olympic',
        date: '2026-07-26',
        endDate: null,
        note: null,
        recordKey: 'supertri--tor--26',
      },
    },
  },
  2027: {
    checked: '2026-09-29',
    sources: {
      'https://www.ironman.com/races/im703-oman': {
        name: 'IRONMAN 70.3 Muscat',
        location: 'Muscat, Oman',
        kind: 'triathlon',
        format: '70.3',
        date: '2027-02-06',
        endDate: null,
        note: null,
      },
      'https://hyrox.com/event/sweat-pals-hyrox-miami-beach-26-27/': {
        name: 'Sweatpals HYROX Miami Beach',
        location: 'Miami Beach, Florida, USA',
        kind: 'hyrox',
        format: 'HYROX',
        date: '2027-03-26',
        endDate: '2027-03-28',
        note: 'Event weekend; division to be decided.',
      },
      'https://www.ironman.com/races/im-lanzarote/register': {
        name: 'IRONMAN Lanzarote',
        location: 'Lanzarote, Spain',
        kind: 'triathlon',
        format: '140.6',
        date: '2027-05-15',
        endDate: null,
        note: null,
      },
      'https://supertri.com/toronto-triathlon/': {
        name: 'Supertri Toronto',
        location: 'Toronto, Canada',
        kind: 'triathlon',
        format: 'olympic',
        date: '2027-07-25',
        endDate: null,
        note: null,
      },
      'https://www.ironman.com/races/im-canada-ottawa': {
        name: 'IRONMAN Canada-Ottawa',
        location: 'Ottawa, Canada',
        kind: 'triathlon',
        format: '140.6',
        date: '2027-08-08',
        endDate: null,
        note: null,
      },
      'https://t100triathlon.com/vancouver/participate/': {
        name: 'Vancouver T100 Triathlon',
        location: 'Vancouver, Canada',
        kind: 'triathlon',
        format: 'T100 (100 km)',
        date: '2027-08-15',
        endDate: null,
        note: null,
      },
      'https://www.torontowaterfrontmarathon.com/': {
        name: 'TCS Toronto Waterfront Marathon',
        location: 'Toronto, Canada',
        kind: 'running',
        format: 'marathon (42.2 km)',
        date: '2027-10-18',
        endDate: null,
        note: 'Provisional: same date as the 2026 race (October 18). Awaiting the official 2027 date.',
        details: {
          sources: [
            'https://www.torontowaterfrontmarathon.com/event-info/',
            'https://connect.garmin.com/app/course/401712199',
          ],
          fetched: '2026-09-30',
          venue: null,
          venueMap: null,
          athleteGuide: null,
          terrain: {},
          airC: null,
          waterC: null,
          course: [
            {
              leg: 'run',
              title: '2025 TCS Toronto Waterfront Marathon FULL OFFICIAL',
              source: 'https://connect.garmin.com/app/course/401712199',
              edition: 2025,
              mapUrl: null,
              gpxUrl: '/triathlon/routes/tcs-toronto-waterfront-marathon-2025.gpx',
              distanceM: 42330,
              raceDistanceM: 42195,
              laps: null,
              aidStationsPerLap: null,
              elevationGainM: null,
              polyline:
                'c|miGzeocNqlAl_@d^jxCpcDmaAvExMl@rCXxKV`B`AtAtBXhDOp@bErBxD|HnEfApBtG`ZlCfYf@lIfAr\\qEpSuIjS{A~M]pM]tAmKvZiCdKmClIoDfUiBtSR|^hBxQ`@bLd@fCtCfH|@pDNBJWo@sD{CaIg@aDKcI{AmMW}EImGNqGdAeP~B_U?qKPoAvCcJhCkKtG_S|AaEhD}EjByFzAuMjA_P~Gq[@aCcBca@_CoYuG}[k@_ByBsBiFsCo@s@_BqD{@}DyA{]{@sFoE}P_E_JiAeEkGyq@yBeSiO{`@uB^aUtHqFwb@Jm@kGcUwKqv@u@{AeCiByDmI`FcB_E{Z}BRaAIcFwB_A@m@VcHrHoEnAen@`MTJhs@iOl@UvGcHpAUzFzBrDFjDtXcF`BqBgE_LwZk@wCg@mHBuCbAaFZ_DCgBwJmt@vScHkU{p@}@_EcJkk@uNgdAuDkVk@gBaBgCqBeB}Bw@qBTqLlDiToaB`Gzf@hLpz@zM_EdAEfC|@rApA|BfErCfP`[dzBbArD~Trn@cSrGvJ|u@?xAyAtIMfBVdHZdC~AvFvIhUtJxR`BhAv@pA`Ldw@hFjShGfe@}YfJa@XcAnC[J',
              profile: [
                92, 101, 106, 108, 111, 111, 107, 101, 92, 86, 78, 75, 75, 74, 80, 80, 74, 75, 75,
                75, 75, 74, 80, 77, 75, 75, 75, 78, 83, 81, 81, 83, 80, 77, 75, 75, 75, 75, 75, 75,
                75, 75, 75, 75, 75, 75, 75, 77, 83, 89, 87, 81, 75, 75, 75, 75, 75, 75, 75, 75, 78,
                81, 83, 89,
              ],
              aidM: [],
            },
          ],
          schedule: null,
          schedulePending: false,
          kept: {},
        },
      },
    },
  },
}
