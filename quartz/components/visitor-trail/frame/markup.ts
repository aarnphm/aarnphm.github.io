import { trailSceneHeight } from '../../../util/trail-scene'
import { trailBackpacker } from '../backpacker'
import { trailCampsite } from '../campsite'
import { trailIds } from '../layout'
import { mountSceneReveal } from '../scene-reveal'
import { mountSceneSurface } from '../scene-surface'

export function createFrameSurface(document: Document, host: HTMLElement, signal: AbortSignal) {
  // Use the viewport before the first layout; the homepage header is a narrow containing block.
  if (typeof document.documentElement.moveBefore === 'function')
    document.documentElement.moveBefore(host, null)
  host.style.height = `${trailSceneHeight(host)}px`
  const scene = document.createElement('div')
  scene.className = 'trail-scene'
  scene.dataset.trailRendered = 'false'
  scene.dataset.trailVisible = 'false'
  scene.dataset.campPositioned = 'false'
  const clip = `trail-clear-${crypto.randomUUID()}`
  scene.innerHTML = `
      <svg class="trail-canvas" aria-hidden="true">
        <defs>
          <clipPath id="${clip}"><path data-trail-clear clip-rule="evenodd"/></clipPath>
          <clipPath id="${clip}-backdrop"><path data-trail-backdrop-clear clip-rule="evenodd"/></clipPath>
          <g data-trail-navigation></g>
        </defs>
        <g data-trail-model>
        ${trailIds.map(id => `<path class="trail-path" data-trail-route="${id}"/>`).join('')}
        <g data-pet-walk-paths></g>
        <path class="trail-path" data-pet-expedition/>
        </g>
        <rect class="trail-paper" data-trail-layer="paper" width="100%" height="100%" fill="var(--light)"/>
        <g class="trail-sky" aria-hidden="true"></g>
        <g clip-path="url(#${clip})">
        <g data-trail-backdrop-container></g>
        <g class="trail-far-gate-layer" data-trail-layer="far-gate" aria-hidden="true"></g>
        <g class="trail-ground-base" data-trail-layer="ground-base"></g>
        <g class="trail-woodland" data-trail-depth="0.8"></g>
        <g class="trail-ground" data-trail-layer></g>
        <g class="trail-ground-cover" data-trail-layer="ground-cover"></g>
        <g class="trail-hinterland" data-trail-layer="hinterland"></g>
        <g class="trail-weather" aria-hidden="true"></g>
        <g class="trail-steps" data-trail-layer="steps"></g>
        <g class="trail-steps trail-camp-approach" data-trail-layer="camp-approach"></g>
        <g class="trail-hiker-tracks"></g>
        <g class="trail-reader-tracks"></g>
        <g class="trail-camp-tracks"></g>
        <g class="trail-visitor-tracks"></g>
        <g class="trail-ranger-tracks"></g>
        <g class="trail-athlete-tracks"></g>
        <g class="trail-park-patrol"></g>
        <g class="trail-landmarks"></g>
        <g class="trail-extension-forest"></g>
        <g class="trail-sunlight" aria-hidden="true"></g>
        <g class="trail-fauna"></g>
        <g class="trail-sunlight trail-sunlight-canopy" aria-hidden="true"></g>
        <g class="trail-holidays" aria-hidden="true"></g>
        <g class="trail-wayfarers"></g>
        <g class="trail-community"></g>
        <g class="trail-visitors"></g>
        <g class="trail-athletes" aria-hidden="true"></g>
        ${trailBackpacker()}
        <g class="trail-depth-objects"></g>
        <g class="trail-birds" aria-hidden="true"></g>
        </g>
      </svg>
      <div class="trail-visitor-labels"></div>
      <div class="trail-campsite" aria-hidden="true">
        <svg viewBox="0 0 128 96"><g class="trail-camp" transform="translate(64 64)">${trailCampsite(false)}</g></svg>
      </div>`
  const styleReady = mountSceneSurface(host, scene, signal)
  const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
  const count = document.querySelector<HTMLElement>('[data-trail-connection]')
  const retry = document.querySelector<HTMLButtonElement>('[data-trail-retry]')
  const campsite = scene.querySelector<HTMLElement>('.trail-campsite')
  if (!svg || !count || !retry || !campsite) throw new Error('Shared trail markup is incomplete')
  const opening = mountSceneReveal(scene, host, signal)
  return { scene, clip, styleReady, svg, count, retry, campsite, opening }
}
