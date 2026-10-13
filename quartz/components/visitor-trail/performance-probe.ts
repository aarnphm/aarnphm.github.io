import type {
  PerformanceDebugProbe,
  PerformanceDebugProfile,
  PerformanceDebugReading,
} from '../scripts/performance-debug'
import { trailScene } from '../../util/trail-scene'

const megabytes = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MB`

function sceneProbe(document: Document): PerformanceDebugProbe {
  let observed: HTMLElement | null = null
  let writes = 0
  let counted = 0
  let lastRead = performance.now()
  // Each attribute, style, or child change in the scene invalidates style for the next frame.
  const mutations = new MutationObserver(records => {
    writes += records.length
  })
  const observe = (scene: HTMLElement | null) => {
    if (scene === observed) return
    mutations.disconnect()
    observed = scene
    if (scene)
      mutations.observe(scene, {
        attributes: true,
        characterData: true,
        childList: true,
        subtree: true,
      })
  }
  return {
    read(): PerformanceDebugReading[] {
      const scene = trailScene(document)
      observe(scene)
      const now = performance.now()
      const rate = ((writes - counted) * 1000) / Math.max(1, now - lastRead)
      counted = writes
      lastRead = now
      if (!scene) return [{ label: 'scene', value: 'not mounted' }]
      const canvases = [...scene.getElementsByTagName('canvas')]
      const bytes = canvases.reduce((sum, canvas) => sum + canvas.width * canvas.height * 4, 0)
      const running = scene
        .getAnimations({ subtree: true })
        .filter(animation => animation.playState === 'running').length
      return [
        { label: 'scene nodes', value: String(scene.getElementsByTagName('*').length) },
        { label: 'animations', value: String(running) },
        { label: 'canvases', value: `${canvases.length} · ${megabytes(bytes)}` },
        { label: 'scene writes', value: `${rate.toFixed(0)} /s` },
      ]
    },
    stop() {
      mutations.disconnect()
      observed = null
    },
  }
}

export function trailPerformanceProfile(host: HTMLElement): PerformanceDebugProfile {
  return {
    label: 'trail performance debug',
    query: 'trail-debug',
    sources: ['cursor'],
    state: () => (host.hidden ? null : 'trail'),
    probe: () => sceneProbe(host.ownerDocument),
  }
}
