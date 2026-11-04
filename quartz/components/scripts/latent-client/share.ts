import type { Runtime } from './runtime'
import { readSharedView, sharedViewUrl, type SharedView } from '../../../util/latent/share'
import { Toast } from '../toast'
import { openBench } from './bench'
import { restoreFeatures, snapshotFeatures } from './features'
import { goTo } from './reader'
import { runQuery } from './search'

function snapshot(rt: Runtime): SharedView {
  const selected = rt.probe.querySelector<HTMLElement>('.lat-tab[aria-selected="true"]')?.dataset
    .tab
  const bench = rt.bench
  return {
    v: 1,
    base: rt.base,
    focus: rt.focus,
    tab: selected === 'features' || selected === 'axes' ? selected : 'bench',
    features: snapshotFeatures(rt),
    bench: bench
      ? {
          yaw: bench.yaw,
          pitch: bench.pitch,
          zoom: bench.zoom,
          panX: bench.panX / Math.max(1, bench.w),
          panY: bench.panY / Math.max(1, bench.h),
          axes: [...bench.axes],
          level: bench.level,
          hidden: bench.hidden,
          featsOn: bench.featsOn,
          picked: bench.picked,
          feature: bench.feature ? { ...bench.feature } : null,
        }
      : null,
  }
}

/** Bind copying once per page; restore a shared view before the default idle bench opens. */
export async function bindShare(rt: Runtime): Promise<boolean> {
  const head = rt.probe.querySelector('.lat-pane-head')
  if (!head) return false
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'lat-share'
  button.textContent = 'copy link'
  button.title = 'copy the focus, steering, features and 3-d view'
  head.append(button)
  const toast = new Toast()
  const notice = (text: string) => {
    const handle = toast.show(text, { durationMs: 5000 })
    rt.signal.addEventListener('abort', () => handle.dismiss(), { once: true })
  }
  const dialog = document.createElement('dialog')
  dialog.className = 'lat-share-dialog'
  const label = document.createElement('label')
  label.textContent = 'copy this link'
  const input = document.createElement('textarea')
  input.readOnly = true
  label.append(input)
  const close = document.createElement('button')
  close.type = 'button'
  close.textContent = 'close'
  close.addEventListener('click', () => dialog.close(), { signal: rt.signal })
  dialog.append(label, close)
  rt.root.append(dialog)
  rt.signal.addEventListener(
    'abort',
    () => {
      dialog.close()
      dialog.remove()
      button.remove()
    },
    { once: true },
  )
  button.addEventListener(
    'click',
    async () => {
      const url = sharedViewUrl(window.location.href, snapshot(rt))
      try {
        await navigator.clipboard.writeText(url)
        if (!rt.signal.aborted) notice('link copied')
      } catch {
        if (rt.signal.aborted) return
        input.value = url
        dialog.showModal()
        input.focus()
        input.select()
      }
    },
    { signal: rt.signal },
  )

  const hash = window.location.hash
  // A pasted share URL can change only the fragment, bypassing page navigation.
  window.addEventListener(
    'hashchange',
    () => {
      if (window.location.hash.startsWith('#view=') && window.location.hash !== hash)
        window.location.reload()
    },
    { signal: rt.signal },
  )
  if (!hash.startsWith('#view=')) return false
  const saved = readSharedView(hash)
  if (!saved || saved.focus >= rt.docByRow.length) {
    notice('this shared view is invalid')
    return true
  }
  if (saved.base !== rt.base) {
    notice('this shared view uses a different embedding build')
    return true
  }
  button.disabled = true
  try {
    await goTo(rt, saved.focus)
    for (const tab of rt.probe.querySelectorAll<HTMLElement>('.lat-tab[data-tab]'))
      tab.setAttribute('aria-selected', String(tab.dataset.tab === saved.tab))
    for (const panel of rt.probe.querySelectorAll<HTMLElement>('.lat-panel'))
      panel.hidden = panel.dataset.panel !== saved.tab
    if (saved.bench || saved.tab === 'bench') await openBench(rt, saved.bench)
    if (saved.features?.query) {
      const box = rt.links.querySelector<HTMLInputElement>('form.lat-search input')
      if (box) box.value = saved.features.query
      await runQuery(rt, saved.features.query)
      if (snapshotFeatures(rt)?.query !== saved.features.query)
        throw new Error('shared query could not be restored')
    }
    if (saved.features) await restoreFeatures(rt, saved.features)
  } catch (error) {
    if (!rt.signal.aborted) {
      console.warn('[latent] shared view', error)
      notice('could not restore the shared view; reload to retry')
    }
  } finally {
    button.disabled = false
  }
  return true
}
