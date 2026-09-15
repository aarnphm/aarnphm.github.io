import { createElement, render } from 'preact'
import { ArenaReader } from '../arena-feed/reader'
import { currentNavSignal } from './nav-lifecycle'

let mountedSignal: AbortSignal | undefined

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (mountedSignal === signal) return
  const root = document.querySelector<HTMLElement>('[data-arena-feed-mount]')
  if (!root) return
  mountedSignal = signal
  root.replaceChildren()
  render(createElement(ArenaReader, { signal }), root)
  window.addCleanup(() => {
    render(null, root)
    if (mountedSignal === signal) mountedSignal = undefined
  })
})
