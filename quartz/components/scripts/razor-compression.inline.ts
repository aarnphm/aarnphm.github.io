import katex from 'katex'

const P_RET_FALLBACK = 0.15

const rzcTex = (t: string): string => {
  try {
    return katex.renderToString(t, {
      displayMode: false,
      output: 'html',
      throwOnError: false,
      strict: false,
    })
  } catch {
    return t
  }
}

const setText = (root: HTMLElement, sel: string, html: string) => {
  const el = root.querySelector<HTMLElement>(sel)
  if (el) el.innerHTML = html
}

const setAttr = (root: HTMLElement, sel: string, name: string, value: string) => {
  const el = root.querySelector(sel)
  if (el) el.setAttribute(name, value)
}

const setupRazorCompression = () => {
  const roots = document.querySelectorAll<HTMLElement>('[data-razor-compression]')
  for (const root of roots) {
    if (root.dataset.rzcBound === 'true') continue
    root.dataset.rzcBound = 'true'

    const trackX = Number(root.dataset.trackX ?? '150')
    const trackW = Number(root.dataset.trackW ?? '460')
    const sinkW = Number(root.dataset.sinkW ?? '18')
    const compCx = Number(root.dataset.compCx ?? '380')
    const compY = Number(root.dataset.compY ?? '214')
    const row2Y = Number(root.dataset.row2Y ?? '140')
    const trackH = Number(root.dataset.trackH ?? '34')
    const pRet = Number(root.dataset.pret ?? String(P_RET_FALLBACK))
    const stops = (root.dataset.stops ?? '2,3,4,5,8,10').split(',').map(Number)

    const slider = root.querySelector<HTMLInputElement>('[data-rzc-slider]')
    if (!slider) continue

    const render = (c: number) => {
      const rollW = trackW / c
      const dropX = trackX + sinkW
      const dropW = Math.max(0, trackW - sinkW - rollW)
      const rollX = dropX + dropW
      const dropCenter = dropX + dropW / 2
      const retained = pRet + (1 - pRet) / c
      const ratio = 1 / retained

      setAttr(root, '[data-rzc-drop]', 'width', String(dropW))
      setAttr(root, '[data-rzc-drop-hatch]', 'width', String(dropW))
      setAttr(root, '[data-rzc-drop-label]', 'width', String(dropW))
      setAttr(root, '[data-rzc-roll]', 'x', String(rollX))
      setAttr(root, '[data-rzc-roll]', 'width', String(rollW))
      setAttr(
        root,
        '[data-rzc-comp-arrow]',
        'd',
        `M ${dropCenter} ${row2Y + trackH} L ${compCx} ${compY}`,
      )

      setText(root, '[data-rzc-buffer]', rzcTex(`1/${c}`))
      setText(
        root,
        '[data-rzc-retained]',
        rzcTex(`p + (1-p)\\tfrac{1}{C} = ${retained.toFixed(2)}`),
      )
      setText(root, '[data-rzc-compression]', rzcTex(`${ratio.toFixed(2)}\\times`))

      for (const tick of root.querySelectorAll<HTMLElement>('[data-rzc-tick]')) {
        tick.dataset.rzcActive = Number(tick.dataset.rzcTick) === c ? 'true' : 'false'
      }

      slider.setAttribute('aria-valuenow', String(c))
      slider.setAttribute('aria-valuetext', `C = ${c}`)
    }

    const handleInput = () => {
      const idx = Number(slider.value)
      render(stops[idx] ?? stops[0])
    }

    slider.addEventListener('input', handleInput)
    handleInput()

    window.addCleanup(() => {
      slider.removeEventListener('input', handleInput)
      delete root.dataset.rzcBound
    })
  }
}

document.addEventListener('nav', setupRazorCompression)
