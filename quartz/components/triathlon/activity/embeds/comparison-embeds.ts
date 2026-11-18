import type { TriathlonContext } from '../../runtime/context'
import type { DetailPayload } from '../data'
import { scopeDomIds } from '../../../../util/scope-element-ids'
import { buildActivityComparison, powerViewActivity } from '../../../../util/triathlon-card'
import { decodeActivityComparisonAnchor } from '../../../../util/triathlon-comparison'
import { rootNavSignal } from '../../../scripts/root-lifecycle'
import { createDomFactory, applyI18n } from '../../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../../runtime/preferences'
import { wireActivityComparison } from '../comparison'
import { detailContextFromPayload } from '../data'

export const setupActivityComparisonEmbeds = (context: TriathlonContext): (() => void) | null => {
  let live = true
  const teardowns: (() => void)[] = []
  const registered = new Map<HTMLElement, () => void>()
  const upgradeByEmbed = new Map<HTMLElement, () => void>()
  const observer = new IntersectionObserver(
    entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting || !(entry.target instanceof HTMLElement)) continue
        upgradeByEmbed.get(entry.target)?.()
      }
    },
    { rootMargin: '600px 0px' },
  )
  teardowns.push(() => {
    observer.disconnect()
    upgradeByEmbed.clear()
  })

  const setupEmbed = (embed: HTMLElement): void => {
    if (registered.has(embed)) return
    const activityIds = decodeActivityComparisonAnchor(embed.dataset.compareAnchor ?? '')
    if (!activityIds) return
    const lifecycleSignal = rootNavSignal(embed)
    const embedTeardowns: (() => void)[] = []
    let cleaned = false
    const cleanupEmbed = (): void => {
      if (cleaned) return
      cleaned = true
      lifecycleSignal.removeEventListener('abort', cleanupEmbed)
      observer.unobserve(embed)
      upgradeByEmbed.delete(embed)
      registered.delete(embed)
      for (let index = embedTeardowns.length - 1; index >= 0; index -= 1) {
        embedTeardowns[index]()
      }
    }
    registered.set(embed, cleanupEmbed)
    lifecycleSignal.addEventListener('abort', cleanupEmbed, { once: true })
    const detailPath = embed.dataset.detailPath ?? '/static/strava-detail.json'
    let upgraded = false
    let payload: DetailPayload | null = null
    let interactionCleanup: (() => void) | null = null
    const render = (data: DetailPayload) => {
      interactionCleanup?.()
      const activities = activityIds.flatMap(activityId => {
        const activity = data.details[activityId]
        return activity ? [powerViewActivity(context.presentation, activity)] : []
      })
      const comparisonContext = detailContextFromPayload(data)
      const comparison = buildActivityComparison(
        createDomFactory(context.presentation),
        activities,
        comparisonContext,
        { removable: false, health: data.health },
      )
      applyI18n(comparison, context.presentation)
      if (embed.dataset.triathlonOccurrence)
        scopeDomIds(comparison, embed.dataset.triathlonOccurrence)
      embed.replaceChildren(comparison)
      interactionCleanup =
        comparison.dataset.compareState === 'ready'
          ? wireActivityComparison(context.presentation, comparison, activities, {
              ctx: comparisonContext,
              health: data.health,
            })
          : null
    }
    const upgrade = () => {
      if (upgraded) return
      upgraded = true
      observer.unobserve(embed)
      upgradeByEmbed.delete(embed)
      void context.resources.detail.load(detailPath).then(result => {
        if (!live || lifecycleSignal.aborted || !embed.isConnected || result.status !== 'ready')
          return
        payload = result.value
        render(result.value)
      })
    }
    const onPresentationChange = () => (payload ? render(payload) : upgrade())
    window.addEventListener('tri:unit', onPresentationChange)
    window.addEventListener('tri:locale', onPresentationChange)
    window.addEventListener(TRI_POWER_FILTER_EVENT, onPresentationChange)
    const events = ['pointerdown', 'touchstart'] as const
    for (const event of events)
      embed.addEventListener(event, upgrade, { once: true, passive: true })
    upgradeByEmbed.set(embed, upgrade)
    observer.observe(embed)
    embedTeardowns.push(() => {
      interactionCleanup?.()
      window.removeEventListener('tri:unit', onPresentationChange)
      window.removeEventListener('tri:locale', onPresentationChange)
      window.removeEventListener(TRI_POWER_FILTER_EVENT, onPresentationChange)
      for (const event of events) embed.removeEventListener(event, upgrade)
    })
  }

  const setupWithin = (root: ParentNode): void => {
    if (root instanceof HTMLElement && root.matches('.tri-compare-embed[data-compare-anchor]'))
      setupEmbed(root)
    for (const embed of root.querySelectorAll<HTMLElement>(
      '.tri-compare-embed[data-compare-anchor]',
    ))
      setupEmbed(embed)
  }
  const onContentMounted = (event: CustomEventMap['contentdecrypted']): void => {
    setupWithin(event.detail.content)
  }
  setupWithin(document)
  document.addEventListener('contentdecrypted', onContentMounted)
  teardowns.push(() => document.removeEventListener('contentdecrypted', onContentMounted))

  return () => {
    live = false
    for (const cleanup of Array.from(registered.values())) cleanup()
    for (const teardown of teardowns) teardown()
  }
}
