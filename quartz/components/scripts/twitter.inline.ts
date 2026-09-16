import { mountTwitterEmbeds } from './twitter'

document.addEventListener('nav', () => {
  mountTwitterEmbeds(document.body)
})
