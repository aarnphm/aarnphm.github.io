import { getFullSlug } from '../../../util/path'
import { sectionOf } from '../../../util/section-icon'
import { STORM_PERIOD, visitorPrecipitation, wetFall } from '../../../util/visitor-precipitation'
import { nextSunEvent, readVisitorForecast } from '../../../util/visitor-weather'
import { fetchVisitorWeather } from '../../../util/visitor-weather-request'
import {
  center,
  friendCenter,
  friendScale,
  friendState,
  height,
  later,
  petState,
  readerMode,
  trailState,
  type State,
} from './actor-model'

export const waitAfterMs = 7000
export const workAfterMs = 20000

export const state = (now: number): State => {
  if (petState.override && now < petState.override.until) return petState.override.state
  if (petState.carry) return 'slide'
  if (petState.speaking) return petState.speaking.playing ? 'talk' : 'think'
  if (petState.hovering) return 'happy'
  if (petState.reminding) return 'wave'
  if (petState.perch?.kind === 'hang') return 'idle'
  const quiet = now - Math.max(petState.lastMove, petState.settled)
  if (quiet > workAfterMs) return 'work'
  if (quiet > waitAfterMs) return 'wait'
  return 'idle'
}

export const weatherState = {
  // The weather where the visitor is, from the worker's /api/weather (cached there and in the browser
  // for 15 minutes). How cold: the wind chill, as Environment Canada works it out for 10 °C and under
  // with 5 km/h of wind or more; 10 °C and under is 'cool', 0 °C and under 'cold'. Rain: drizzle, rain
  // or a thunderstorm by its weather code, and snow. A lantern in fog, at dusk and at night; sunglasses in the heat; a hand on its hat in the wind; the umbrella blown inside out in a storm (pet.scss).
  chill: '' as '' | 'cool' | 'cold',
  rain: false,
  snow: false,
  lantern: false,
  heat: false,
  windy: false,
  storm: false,
  beach: false,
  tan: 0,
  tanFrom: null as number | null,
}
// How tanned it is, 0 to 1: it deepens while it lies on its lounger, full after tanMs of sun, and
// lasts as long as the page is open.
const tanMs = 120_000
export const loadWeather = (signal: AbortSignal) =>
  fetchVisitorWeather({ signal })
    .then(body => {
      if (signal.aborted) return
      const forecast = readVisitorForecast(body)
      const w = forecast?.weather
      if (!w) return
      const v = w.wind ** 0.16
      const feels =
        w.temp <= 10 && w.wind >= 4.8
          ? 13.12 + 0.6215 * w.temp - 11.37 * v + 0.3965 * w.temp * v
          : w.temp
      weatherState.chill = feels <= 0 ? 'cold' : feels <= 10 ? 'cool' : ''
      const { fall, fog, storm: thunder } = visitorPrecipitation(w.code)
      weatherState.rain = wetFall(fall)
      weatherState.snow = fall === 'snow'
      // Dusk is the 45 minutes before sunset; after it, night until sunrise.
      const now = Date.now()
      const sun = nextSunEvent(forecast.solar, now)
      const dark = sun ? sun.kind === 'sunrise' || sun.at - now < 45 * 60_000 : w.day === false
      weatherState.lantern = fog || dark
      // 27 °C and over is heat; a clear or partly cloudy one by day is a day for the beach.
      weatherState.heat = w.temp >= 27
      // Wind that takes hats: a fresh breeze (30 km/h) or gusts of 50.
      weatherState.windy = w.wind >= 30 || (w.gust ?? 0) >= 50
      // A storm: a thunderstorm, or rain in that wind.
      weatherState.storm = weatherState.rain && (thunder || weatherState.windy)
      weatherState.beach =
        weatherState.heat && !weatherState.lantern && fall === 'none' && w.code <= 2
      render()
    })
    .catch(() => {})

// A part's animation restarts when it shows again. Align that animation to the lightning's wall
// clock after style has started it, without flushing the whole pet subtree during each render.
export const syncStorm = (event: AnimationEvent) => {
  const el = event.target
  if (
    !event.animationName.startsWith('pet-storm') ||
    !(el instanceof Element) ||
    !(
      petState.svg?.contains(el) ||
      friendState.friendSvg?.contains(el) ||
      trailState.trailArt?.contains(el)
    )
  )
    return
  const at = Date.now() % (STORM_PERIOD * 1000)
  for (const a of el.getAnimations())
    if ('animationName' in a && a.animationName === event.animationName) {
      const t = Number(a.currentTime ?? 0) % (STORM_PERIOD * 1000)
      if (Math.abs(t - at) > 120) a.currentTime = at
    }
}

export const render = () => {
  if (!petState.root || !petState.svg || (!petState.doc && !petState.carry)) return
  const next = state(performance.now())
  if (petState.svg.dataset.state !== next) petState.svg.dataset.state = next
  petState.svg.dataset.kind = petState.perch?.kind ?? 'stand'
  const now = performance.now()
  const lying =
    weatherState.beach &&
    next === 'work' &&
    (petState.svg.dataset.kind === 'stand' || petState.svg.dataset.kind === 'floor')
  if (lying && weatherState.tanFrom === null) weatherState.tanFrom = now
  if (!lying && weatherState.tanFrom !== null) {
    weatherState.tan = Math.min(1, weatherState.tan + (now - weatherState.tanFrom) / tanMs)
    weatherState.tanFrom = null
  }
  petState.svg.style.setProperty('--pet-tan', weatherState.tan.toFixed(3))
  petState.svg.style.setProperty(
    '--pet-tan-left',
    `${Math.round((1 - weatherState.tan) * tanMs)}ms`,
  )
  // On the home page's trail it wears its walking hat, from the hop on until it leaves the path.
  petState.svg.toggleAttribute(
    'data-hat',
    !petState.carry && !!petState.perch?.el?.classList.contains('trail-path'),
  )
  const outfit =
    sectionOf({
      slug: getFullSlug(window),
      base: document.body.dataset.isBase === 'true',
      slides: document.body.dataset.slides === 'true',
    }) ?? ''
  if (petState.svg.dataset.outfit !== outfit) petState.svg.dataset.outfit = outfit
  // Away from the sections, a page's tags show on it: glasses on technical pages, wild hair on
  // philosophy ones.
  const tags = new Set((document.body.dataset.tags ?? '').split(' '))
  const traits = outfit ? '' : ['technical', 'philosophy'].filter(t => tags.has(t)).join(' ')
  if (petState.svg.dataset.traits !== traits) petState.svg.dataset.traits = traits
  if (petState.svg.dataset.cold !== weatherState.chill)
    petState.svg.dataset.cold = weatherState.chill
  petState.svg.toggleAttribute('data-rain', weatherState.rain)
  petState.svg.toggleAttribute('data-snow', weatherState.snow)
  petState.svg.toggleAttribute('data-lantern', weatherState.lantern)
  petState.svg.toggleAttribute('data-heat', weatherState.heat)
  petState.svg.toggleAttribute('data-windy', weatherState.windy)
  petState.svg.toggleAttribute('data-storm', weatherState.storm)
  petState.svg.toggleAttribute('data-beach', weatherState.beach)
  petState.svg.toggleAttribute('data-reading', readerMode() || !!petState.speaking)
  // On the rail it faces the page, its back to the pole.
  petState.gazeArt?.face(petState.carry ? 1 : petState.face)
  syncTrailArt()
  if (petState.speaking?.playing || friendState.chat || petState.reminding) {
    // On the trail the bubble sits on the pixel pet's head (reserveTrailFoot), not the hidden vector's.
    const top =
      trailState.trailHeadY === null
        ? center().y - height(petState.perch?.scale) / 2
        : trailState.trailHeadY - window.scrollY
    petState.root.dataset.bubble = top < 64 ? 'below' : 'above'
    placeBubble()
  }
  renderFriend()
}

// What the bubble says: the phrase being read aloud, else the hydration reminder.
export const showBubble = () => {
  if (!petState.root) return
  const text = petState.root.querySelector<HTMLElement>('.garden-pet-bubble-text')
  const said = petState.speaking?.playing
    ? petState.speaking
    : friendState.chat
      ? { text: friendState.chat, lang: 'fr' }
      : null
  petState.root.classList.toggle('is-speaking', !!said)
  petState.root.classList.toggle('is-reminding', petState.reminding && !said)
  if (text) {
    text.textContent = said ? said.text : petState.reminding ? 'drink water' : ''
    text.lang = said ? said.lang : ''
  }
  render()
}

// Keeps a shown bubble inside the view: it centres on the pet, so a long phrase near an edge
// shifts sideways instead of running off the page.
const placeBubble = () => {
  const bubble = petState.root?.querySelector<HTMLElement>('.garden-pet-bubble')
  if (
    !petState.root ||
    !bubble ||
    !(petState.speaking?.playing || friendState.chat || petState.reminding)
  )
    return
  petState.root.style.removeProperty('--pet-bubble-x')
  const c = center().x
  const half = bubble.offsetWidth / 2
  const gutter = 8
  const shift =
    Math.max(gutter - (c - half), 0) - Math.max(c + half - (window.innerWidth - gutter), 0)
  if (shift) petState.root.style.setProperty('--pet-bubble-x', `${shift.toFixed(0)}px`)
}

// Shows `next` for `ms`, then falls back to whatever the pet would otherwise be doing.
export const play = (next: State, ms: number) => {
  petState.override = { state: next, until: performance.now() + ms }
  render()
  window.setTimeout(render, ms + 20)
}

// Re-renders when the pointer has been quiet long enough for the pet to fidget, then to work.
export const armQuietTimer = () => {
  window.clearTimeout(petState.stateTimer)
  const quiet = performance.now() - Math.max(petState.lastMove, petState.settled)
  const due = quiet < waitAfterMs ? waitAfterMs : quiet < workAfterMs ? workAfterMs : null
  if (due === null) return
  petState.stateTimer = window.setTimeout(
    () => {
      render()
      armQuietTimer()
    },
    due - quiet + 20,
  )
}

export const syncTrailArt = () => {
  if (!trailState.trailArt || !petState.svg) return
  for (const name of trailState.trailArt.getAttributeNames())
    if (name.startsWith('data-') && !petState.svg.hasAttribute(name))
      trailState.trailArt.removeAttribute(name)
  for (const attribute of petState.svg.attributes)
    if (
      attribute.name.startsWith('data-') &&
      trailState.trailArt.getAttribute(attribute.name) !== attribute.value
    )
      trailState.trailArt.setAttribute(attribute.name, attribute.value)
  petState.gazeArt?.syncTrail(trailState.trailArt)
}

export function renderFriend() {
  if (!friendState.friend || !friendState.friendSvg) return
  const now = performance.now()
  const main = petState.svg?.dataset.state
  const next: State =
    friendState.friendOverride && now < friendState.friendOverride.until
      ? friendState.friendOverride.state
      : friendState.friendHovering
        ? 'happy'
        : main === 'work' || main === 'wait'
          ? main
          : 'idle'
  if (friendState.friendSvg.dataset.state !== next) friendState.friendSvg.dataset.state = next
  friendState.friendSvg.dataset.kind = friendState.friendPerch?.kind ?? 'stand'
  if (friendState.friendSvg.dataset.cold !== weatherState.chill)
    friendState.friendSvg.dataset.cold = weatherState.chill
  friendState.friendSvg.toggleAttribute('data-rain', weatherState.rain)
  friendState.friendSvg.toggleAttribute('data-snow', weatherState.snow)
  friendState.friendSvg.toggleAttribute('data-lantern', weatherState.lantern)
  friendState.friendSvg.toggleAttribute('data-heat', weatherState.heat)
  friendState.friendSvg.toggleAttribute('data-storm', weatherState.storm)
  friendState.friendSvg.toggleAttribute('data-beach', weatherState.beach)
  friendState.friendSvg.toggleAttribute('data-reading', readerMode())
  friendState.friendGaze?.face(friendState.friendFace)
  if (friendState.friend.classList.contains('is-speaking')) {
    const c = friendCenter()
    if (c)
      friendState.friend.dataset.bubble = c.y - height(friendScale()) / 2 < 64 ? 'below' : 'above'
  }
}

export const friendPlay = (next: State, ms: number) => {
  friendState.friendOverride = { state: next, until: performance.now() + ms }
  renderFriend()
  later(renderFriend, ms + 20)
}

export const showFriendBubble = (text: string, ms: number) => {
  if (!friendState.friend) return
  const bubble = friendState.friend.querySelector<HTMLElement>('.garden-pet-bubble-text')
  if (bubble) {
    bubble.textContent = text
    bubble.lang = 'fr'
  }
  friendState.friend.classList.add('is-speaking')
  renderFriend()
  later(() => friendState.friend?.classList.remove('is-speaking'), ms)
}

export const say = (text: string, ms: number) => {
  friendState.chat = text
  showBubble()
  later(() => {
    friendState.chat = null
    showBubble()
  }, ms)
}
