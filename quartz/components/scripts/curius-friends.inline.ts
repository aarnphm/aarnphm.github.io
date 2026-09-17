import { fetchFollowing, timeSince } from './curius'
import { rootNavSignal } from './root-lifecycle'

const configuredFriends = new WeakMap<HTMLUListElement, AbortSignal>()

document.addEventListener('nav', async () => {
  const friends = document.querySelector<HTMLUListElement>('#friends-list')
  const section = document.querySelector<HTMLElement>('.curius-friends')
  const status = document.querySelector<HTMLElement>('#curius-friends-status')
  const seeMore = document.querySelector<HTMLButtonElement>('#see-more-friends')
  if (!friends || !section || !status || !seeMore) return

  const signal = rootNavSignal(friends)
  if (configuredFriends.get(friends) === signal) return
  configuredFriends.set(friends, signal)
  signal.addEventListener(
    'abort',
    () => {
      if (configuredFriends.get(friends) === signal) configuredFriends.delete(friends)
    },
    { once: true },
  )

  section.setAttribute('aria-busy', 'true')
  status.textContent = 'Chargement des amis…'
  status.hidden = false
  seeMore.hidden = true
  seeMore.classList.remove('expand')
  seeMore.setAttribute('aria-expanded', 'false')
  const moreText = seeMore.querySelector<HTMLSpanElement>('#more')
  const chevron = seeMore.querySelector('svg')
  if (moreText) moreText.textContent = 'de plus'
  chevron?.classList.remove('fold')

  try {
    const response = await fetchFollowing()
    if (signal.aborted) return

    friends.replaceChildren()
    status.hidden = Boolean(response?.length)
    status.textContent = response ? 'Aucun ami à afficher.' : 'Impossible de charger les amis.'
    if (!response?.length) return

    const rows = response.map(({ user, link }, index) => {
      const row = document.createElement('li')
      row.className = 'friend-li'
      row.classList.toggle('active', index < 4)
      row.addEventListener(
        'click',
        event => {
          if (
            event.defaultPrevented ||
            event.button !== 0 ||
            event.altKey ||
            event.ctrlKey ||
            event.metaKey
          )
            return
          if (event.target instanceof Element && event.target.closest('a')) return
          window.open(link.link, '_blank', 'noopener,noreferrer')
        },
        { signal },
      )

      const title = document.createElement('div')
      title.className = 'friend-title'
      const name = document.createElement('a')
      name.className = 'friend-name'
      name.textContent = `${user.firstName} ${user.lastName}`
      name.href = `https://curius.app/${user.userLink}`
      name.target = '_blank'
      name.rel = 'noopener noreferrer'

      const time = document.createElement('time')
      const createdDate = link.createdDate ?? new Date().toISOString()
      const modifiedDate = link.modifiedDate ?? createdDate
      time.dateTime = modifiedDate
      const modified = new Date(modifiedDate)
      if (!Number.isNaN(modified.getTime())) time.title = modified.toUTCString()
      time.textContent = timeSince(createdDate)
      title.append(name, time)

      const description = document.createElement('div')
      description.className = 'friend-shortcut'
      description.textContent = `in ${link.title}`
      row.append(title, description)
      return row
    })
    friends.append(...rows)
    seeMore.hidden = rows.length <= 4

    seeMore.addEventListener(
      'click',
      () => {
        const expanded = seeMore.getAttribute('aria-expanded') !== 'true'
        seeMore.setAttribute('aria-expanded', String(expanded))
        seeMore.classList.toggle('expand', expanded)
        rows.slice(4).forEach(row => row.classList.toggle('active', expanded))
        chevron?.classList.toggle('fold', expanded)
        if (moreText) moreText.textContent = expanded ? 'moins' : 'de plus'
        if (!expanded) friends.scrollTop = 0
      },
      { signal },
    )
  } catch (error) {
    if (signal.aborted) return
    console.error(error)
    status.hidden = false
    status.textContent = 'Impossible de charger les amis.'
  } finally {
    if (!signal.aborted) {
      section.setAttribute('aria-busy', 'false')
      document.dispatchEvent(new CustomEvent<void>('curius:sidebar-settled'))
    }
  }
})
