import {
  fetchCuriusLinks,
  fetchSearchLinks,
  fetchTrails,
  createTrailMetadata,
  createTrailList,
  curiusSearch,
  createLinkEl,
} from './curius'
import { updateNavigation } from './curius-navigation.inline'
import { setupCuriusPreview } from './curius-preview'
import { currentNavSignal } from './nav-lifecycle'

document.addEventListener('nav', async e => {
  if (e.detail.url !== 'curius') return

  const fragment = document.querySelector<HTMLElement>('#curius-fragments')
  if (!fragment) return
  const signal = currentNavSignal()
  const friends = document.querySelector<HTMLElement>('.curius-friends')
  const trails = document.querySelector<HTMLElement>('.curius-trail')
  const trailStatus = document.querySelector<HTMLElement>('#curius-trails-status')
  const profile = document.querySelector<HTMLAnchorElement>('.curius-profile')

  const updateProfile = () => {
    if (profile)
      profile.hidden =
        friends?.getAttribute('aria-busy') !== 'false' ||
        trails?.getAttribute('aria-busy') !== 'false'
  }
  document.addEventListener('curius:sidebar-settled', updateProfile, { signal })
  if (profile) profile.hidden = true
  trails?.setAttribute('aria-busy', 'true')
  if (trailStatus) {
    trailStatus.hidden = false
    trailStatus.textContent = 'Chargement des sentiers…'
  }

  setupCuriusPreview()
  delete window.curiusState
  updateNavigation()
  window.addCleanup(() => {
    delete window.curiusState
  })
  fragment.setAttribute('aria-busy', 'true')
  document.dispatchEvent(
    new CustomEvent('toast', { detail: { message: 'Récupération des liens curius…' } }),
  )

  const linksRequest = fetchCuriusLinks()
  const searchRequest = fetchSearchLinks()
  const trailsRequest = fetchTrails()

  const renderLinks = async () => {
    const response = await linksRequest
    if (signal.aborted) return

    const seen = new Set<number | string>()
    const links = (response?.links ?? []).filter(link => {
      const key = link.id ?? link.link
      if (link.trails.length > 0 || seen.has(key)) return false
      seen.add(key)
      return true
    })
    fragment.replaceChildren(...links.map(createLinkEl))
    const canLoadMore = Boolean(response?.links?.length) && response?.hasMore !== false
    if (links.length === 0 && !canLoadMore) {
      const message = response
        ? 'Aucun lien disponible pour le moment.'
        : 'Impossible de charger les liens curius.'
      const status = document.createElement('li')
      status.className = 'curius-list-status'
      status.setAttribute('role', 'status')
      status.textContent = message
      fragment.appendChild(status)
      if (!response)
        document.dispatchEvent(new CustomEvent('toast', { detail: { message, durationMs: 5000 } }))
    }
    fragment.removeAttribute('aria-busy')
    document.dispatchEvent(new CustomEvent('curius:links-ready', { detail: response }))
  }

  const renderTrails = async () => {
    const trailData = await trailsRequest
    if (signal.aborted) return
    try {
      let metadata = await createTrailMetadata({ trails: trailData ?? [] })
      if (signal.aborted) return
      if (metadata.size === 0) {
        const response = await linksRequest
        if (signal.aborted) return
        metadata = await createTrailMetadata({ links: response?.links ?? [] })
        if (signal.aborted) return
      }
      createTrailList(metadata)
      if (trailStatus) {
        trailStatus.hidden = metadata.size > 0
        trailStatus.textContent = trailData
          ? 'Aucun sentier disponible pour le moment.'
          : 'Impossible de charger les sentiers.'
      }
    } catch (error) {
      if (signal.aborted) return
      console.error(error)
      if (trailStatus) {
        trailStatus.hidden = false
        trailStatus.textContent = 'Impossible de charger les sentiers.'
      }
    } finally {
      if (!signal.aborted) {
        trails?.setAttribute('aria-busy', 'false')
        updateProfile()
      }
    }
  }

  const initialiseSearch = async () => {
    const links = await searchRequest
    if (!signal.aborted) await curiusSearch(links)
  }

  const results = await Promise.allSettled([renderLinks(), renderTrails(), initialiseSearch()])
  for (const result of results)
    if (result.status === 'rejected' && !signal.aborted) console.error(result.reason)
})
