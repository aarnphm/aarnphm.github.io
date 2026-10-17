import { playhtml, type PageDataChannel } from 'playhtml'

export type Watering = { count: number; at: number }
type GroveData = Record<string, Watering>
type LegacyGroveData = { trees: Record<string, Record<string, Watering>> }
const cooldown = 45_000

function contribution(value: unknown): Watering | null {
  if (!value || typeof value !== 'object' || !('count' in value) || !('at' in value)) return null
  return typeof value.count === 'number' &&
    Number.isFinite(value.count) &&
    typeof value.at === 'number' &&
    Number.isFinite(value.at)
    ? { count: Math.max(0, Math.min(32, Math.floor(value.count))), at: value.at }
    : null
}

export function mountSharedGrove(
  scene: HTMLElement,
  host: HTMLElement,
  signal: AbortSignal,
  onGrowth: () => void,
  // Each tree's waterings, for the forest to age with: water lengthens a tree's life.
  onWater: (water: Map<string, Watering[]>) => void,
) {
  const document = scene.ownerDocument
  const view = document.defaultView
  if (!view) throw new Error('The grove needs its homepage')
  const total = document.querySelector<HTMLElement>('[data-grove-water-total]')
  const hits = document.createElement('div')
  hits.className = 'trail-tree-waterings'
  hits.inert = Boolean(host.hidden)
  const status = document.createElement('div')
  status.className = 'trail-water-feedback'
  status.dataset.treeWaterStatus = ''
  status.setAttribute('role', 'status')
  status.dataset.visible = 'false'
  scene.append(hits, status)
  const motion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let channel: PageDataChannel<GroveData> | null = null
  let visitor = ''
  let data: GroveData = {}
  let legacy: PageDataChannel<LegacyGroveData> | null = null
  let unsubscribeLegacy = () => {}
  let unsubscribe = () => {}
  let feedbackTimer = 0
  const gusts = new Set<Animation>()
  let targets: { tree: SVGGElement; button: HTMLButtonElement }[] = []
  const placeTargets = () => {
    if (
      signal.aborted ||
      host.hidden ||
      document.hidden ||
      scene.dataset.trailRendered !== 'true' ||
      scene.dataset.trailOpening !== 'complete' ||
      scene.dataset.trailReveal !== undefined
    )
      return
    // The compositor has measured the finished forest before these controls become usable.
    const bounds = targets.map(({ tree, button }) => ({ tree, button, box: tree.getBBox() }))
    for (const { tree, button, box } of bounds) {
      button.style.left = `${Number(tree.dataset.groundX) + box.x + box.width / 2}px`
      button.style.top = `${Number(tree.dataset.groundY) + box.y + box.height / 2}px`
      button.style.width = `${Math.max(28, Math.min(54, box.width))}px`
      button.style.height = `${Math.max(28, Math.min(54, box.height - 6))}px`
    }
    hits.inert = false
  }
  host.addEventListener('trail:open', () => (hits.inert = true), { signal })
  host.addEventListener('trail:ready', placeTargets, { signal })
  host.addEventListener('trail:show', placeTargets, { signal })

  const feedback = (message: string) => {
    view.clearTimeout(feedbackTimer)
    status.dataset.visible = 'true'
    status.replaceChildren(document.createTextNode(message))
    feedbackTimer = view.setTimeout(() => {
      status.dataset.visible = 'false'
    }, 4000)
  }
  // Watering rustles the leaf clusters in whole cells, outward from the middle of the crown; the
  // trunk does not move.
  const gust = (tree: SVGGElement) => {
    const clusters = [...tree.querySelectorAll<SVGGElement>('.trail-tree-leaves')]
    if (!clusters.length || motion.matches) return
    const direction = Number.parseFloat(scene.style.getPropertyValue('--trail-wind-x')) < 0 ? -1 : 1
    tree.dataset.gust = 'active'
    const running = clusters.map(cluster => {
      const cell = Number.parseFloat(cluster.style.getPropertyValue('--leaf-cell')) || 1
      const lift = Number(cluster.style.getPropertyValue('--leaf-lift')) || 0
      const reach = lift > 0.3 ? cell : 0
      const animation = cluster.animate(
        [0, 1, -1, 1, 0, 0].map(step => ({
          transform: `translate(${step * reach * direction}px, ${step === 1 ? -reach : 0}px)`,
          easing: 'steps(1, end)',
        })),
        {
          duration: 1100,
          delay: Math.abs(Number(cluster.style.getPropertyValue('--leaf-x')) || 0) * 6,
        },
      )
      gusts.add(animation)
      return animation
    })
    void Promise.all(running.map(animation => animation.finished))
      .then(() => {
        tree.dataset.gust = 'idle'
      })
      .catch(() => {})
      .finally(() => {
        for (const animation of running) gusts.delete(animation)
      })
  }
  const apply = () => {
    if (signal.aborted) return
    if (total && channel) {
      const waters = Object.values(data).reduce(
        (sum, value) => sum + (contribution(value)?.count ?? 0),
        0,
      )
      const label = `${waters.toLocaleString()} ${waters === 1 ? 'watering' : 'waterings'}`
      total.textContent = label
      total.dataset.groveWaterTotal = String(waters)
      total.setAttribute('aria-label', `Shared grove, ${label}`)
      total.title = 'Shared grove · all visitors'
      total.hidden = false
    }
    let changed = false
    targets = []
    for (const tree of scene.querySelectorAll<SVGGElement>('.trail-tree')) {
      const id = tree.dataset.treeId
      if (!id) continue
      const waters = Object.entries(data)
        .filter(([key]) => key.startsWith(`${id}|`))
        .reduce((total, [, value]) => total + (contribution(value)?.count ?? 0), 0)
      const growth = Math.min(
        Number(tree.dataset.growthLimit),
        1 + 0.3 * (1 - Math.exp(-waters / 9)),
      )
      changed ||=
        tree.dataset.communityGrowth !== undefined &&
        tree.dataset.communityGrowth !== growth.toFixed(4)
      const radius = (Number(tree.dataset.baseTrunkRadius) * growth).toFixed(4)
      if (tree.dataset.water !== String(waters)) tree.dataset.water = String(waters)
      if (tree.dataset.communityGrowth !== growth.toFixed(4))
        tree.dataset.communityGrowth = growth.toFixed(4)
      if (tree.dataset.trunkRadius !== radius) tree.dataset.trunkRadius = radius
      const community = tree.querySelector('.trail-tree-community')
      const transform = `scale(${growth.toFixed(4)})`
      if (community?.getAttribute('transform') !== transform)
        community?.setAttribute('transform', transform)
      const button = hits.querySelector<HTMLButtonElement>(`[data-water-tree="${id}"]`)
      if (button) {
        button.setAttribute(
          'aria-label',
          `Water ${tree.dataset.tree} ${id.slice(5)}, ${waters} waterings`,
        )
        button.title = `${waters} waterings${tree.dataset.treeAge ? ` · ${tree.dataset.treeAge} years old` : ''} · click to water`
        targets.push({ tree, button })
      }
    }
    placeTargets()
    if (changed) onGrowth()
  }
  const refresh = () => {
    const retained = new Map(
      [...hits.querySelectorAll<HTMLButtonElement>('[data-water-tree]')].map(button => [
        button.dataset.waterTree,
        button,
      ]),
    )
    for (const tree of scene.querySelectorAll<SVGGElement>('.trail-tree')) {
      if (tree.dataset.treeDepth === 'foothill') continue
      const id = tree.dataset.treeId
      if (!id) continue
      if (retained.delete(id)) continue
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'trail-tree-water'
      button.dataset.waterTree = id
      hits.append(button)
    }
    for (const button of retained.values()) button.remove()
    apply()
  }
  hits.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof view.HTMLButtonElement)) return
      const id = event.target.dataset.waterTree
      const tree = scene.querySelector<SVGGElement>(`.trail-tree[data-tree-id="${id}"]`)
      if (!id || !tree) return
      gust(tree)
      if (!channel) {
        feedback('connecting to the grove')
        return
      }
      const key = `${id}|${visitor}`
      const previous = contribution(channel.getData()[key])
      const now = Date.now()
      if (previous && now - previous.at < cooldown) {
        feedback(
          `tree ${id.slice(5)} · watered · give it ${Math.ceil((cooldown - now + previous.at) / 1000)}s`,
        )
        return
      }
      if (previous && previous.count >= 32) {
        feedback('this tree has your water · let others help it grow')
        return
      }
      channel.setData(draft => {
        draft[key] = { count: (previous?.count ?? 0) + 1, at: now }
      })
      feedback(`tree ${id.slice(5)} · watered together`)
    },
    { signal },
  )
  const pause = () => {
    for (const animation of gusts) animation.cancel()
    gusts.clear()
    for (const tree of scene.querySelectorAll<SVGGElement>('[data-gust="active"]'))
      tree.dataset.gust = 'idle'
    view.clearTimeout(feedbackTimer)
    status.dataset.visible = 'false'
  }
  host.addEventListener('trail:hide', pause, { signal })
  document.addEventListener(
    'visibilitychange',
    () => {
      if (document.hidden) pause()
      else placeTargets()
    },
    { signal },
  )
  signal.addEventListener(
    'abort',
    () => {
      pause()
      unsubscribe()
      unsubscribeLegacy()
      legacy?.destroy()
      channel?.destroy()
    },
    { once: true },
  )
  return {
    refresh,
    connect(id: string) {
      const current = playhtml.createPageData<GroveData>('community-grove-v2', {})
      channel = current
      visitor = id
      const render = (next: GroveData) => {
        data = next
        const water = new Map<string, Watering[]>()
        for (const [key, value] of Object.entries(next)) {
          const watering = contribution(value)
          const id = key.slice(0, key.indexOf('|'))
          if (!watering || !id) continue
          water.set(id, [...(water.get(id) ?? []), watering])
        }
        onWater(water)
        apply()
      }
      unsubscribe = current.onUpdate(render)
      render(current.getData())
      legacy = playhtml.createPageData<LegacyGroveData>('community-grove-v1', { trees: {} })
      const migrate = (old: LegacyGroveData) => {
        const updates: [string, Watering][] = []
        for (const [tree, visitors] of Object.entries(old.trees ?? {}))
          for (const [person, value] of Object.entries(visitors)) {
            const water = contribution(value)
            const key = `${tree}|${person}`
            const saved = contribution(current.getData()[key])
            if (water && (!saved || saved.count < water.count)) updates.push([key, water])
          }
        if (updates.length)
          current.setData(draft => {
            for (const [key, water] of updates) draft[key] = water
          })
      }
      unsubscribeLegacy = legacy.onUpdate(migrate)
      migrate(legacy.getData())
    },
  }
}
