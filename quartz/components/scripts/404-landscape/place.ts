import type { LandscapeScope } from './scope'
import { almanac, readSky, type Site } from '../404-almanac'
import { daySeed, parkEnv } from '../404-riders'
import { hillEnv } from '../404-skiers'
import { repaint } from './repaint'

const sameWeather = (a: Site['weather'], b: Site['weather']) =>
  a === b ||
  (!!a &&
    !!b &&
    a.code === b.code &&
    a.cloud === b.cloud &&
    a.wind === b.wind &&
    a.temp === b.temp &&
    a.snow === b.snow)

export function moveLandscape(st: LandscapeScope, next: Site) {
  if (
    next.lat === st.site.lat &&
    next.lon === st.site.lon &&
    sameWeather(next.weather, st.site.weather)
  )
    return
  const prevLon = st.site.lon
  st.site = next
  st.al = almanac(new Date(), st.site)
  st.weather = readSky(st.site.weather)
  const nextPark = parkEnv(st.al, st.weather, st.weekend())
  const nextHill = hillEnv(st.al, st.weather, st.weekend())
  if (
    daySeed(new Date(), st.site.lon) !== daySeed(new Date(), prevLon) ||
    nextPark.open !== st.park.env.open ||
    nextPark.crowd !== st.park.env.crowd ||
    nextPark.grip !== st.park.env.grip ||
    nextPark.loose !== st.park.env.loose ||
    nextPark.herd.out !== st.park.env.herd.out ||
    nextHill.open !== st.slopes.env.open ||
    nextHill.crowd !== st.slopes.env.crowd ||
    nextHill.groom !== st.slopes.env.groom ||
    nextHill.fresh !== st.slopes.env.fresh ||
    nextHill.tour !== st.slopes.env.tour ||
    nextHill.control !== st.slopes.env.control ||
    nextHill.camp !== st.slopes.env.camp ||
    nextHill.hold !== st.slopes.env.hold
  ) {
    ;({ park: st.park, slopes: st.slopes } = st.day())
  } else {
    st.park.env = nextPark
    st.slopes.env = nextHill
  }
  repaint(st)
}
