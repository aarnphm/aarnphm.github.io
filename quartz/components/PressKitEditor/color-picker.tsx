import { autoUpdate, computePosition, flip, offset, shift, size } from '@floating-ui/dom'
import { useEffect, useId, useRef, useState } from 'preact/hooks'
import { hueBasis, okhsvToRgb, rgbToOkhsv, type Hsv, type Rgb } from './okhsv'

// Flexoki 600 accents, then the two site favourites. Black and white live on the segments above.
const PRESETS = [
  '#af3029',
  '#bc5215',
  '#ad8301',
  '#66800b',
  '#24837b',
  '#205ea6',
  '#5e409d',
  '#a02f6f',
  '#fdb2a2',
  '#cdd597',
]

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value))

const toRgb = (hex: string): Rgb => {
  const n = Number.parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}
const toHex = (rgb: Rgb) =>
  `#${rgb
    .map(channel =>
      Math.round(clamp(channel, 0, 255))
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`

const fromHsv = ({ h, s, v }: Hsv): Rgb => okhsvToRgb(s, v, hueBasis(h))
const hsvOf = (hex: string, previous?: Hsv) => rgbToOkhsv(toRgb(hex), previous)

// Bitmap sizes: the browser's bilinear upscale is smooth enough at these and keeps a hue drag
// to ~2.3k conversions a frame.
const PLANE = [64, 36] as const
const STRIP = 180

const paint = (
  canvas: HTMLCanvasElement | null,
  width: number,
  height: number,
  at: (x: number, y: number) => Rgb,
) => {
  const context = canvas?.getContext('2d')
  if (!canvas || !context) return
  canvas.width = width
  canvas.height = height
  const image = context.createImageData(width, height)
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const [r, g, b] = at(x, y)
      const i = (y * width + x) * 4
      image.data[i] = r
      image.data[i + 1] = g
      image.data[i + 2] = b
      image.data[i + 3] = 255
    }
  context.putImageData(image, 0, 0)
}

const STEPS: Record<string, [number, number]> = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, 1],
  ArrowDown: [0, -1],
}

interface Props {
  /** `#rrggbb`, or nothing when the selection is on auto or mixed. */
  value?: string
  label: string
  /** Called while a drag or a held key moves the colour; the host should not record history. */
  onPreview: (hex: string) => void
  /** Called once per gesture with the colour to keep. */
  onCommit: (hex: string) => void
  /** Colours taken from the card, read each time the popover opens. */
  sample?: () => string[]
}

/**
 * Swatch button and its popover: a saturation/brightness plane, a hue strip, r/g/b fields and
 * palette chips. The popover sits in the top layer (light dismiss and Escape come from the
 * browser) and is placed under the swatch's row, at that row's width.
 */
export function ColorPicker({ value, label, onPreview, onCommit, sample }: Props) {
  const id = useId()
  const button = useRef<HTMLButtonElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const plane = useRef<HTMLCanvasElement>(null)
  const strip = useRef<HTMLCanvasElement>(null)
  const [open, setOpen] = useState(false)
  const [sampled, setSampled] = useState<string[]>([])
  const [hsv, setHsv] = useState<Hsv>(() => hsvOf(value ?? '#000000'))
  const [draft, setDraft] = useState<(string | undefined)[]>([])
  const live = useRef<string | null>(null)
  const sampleRef = useRef(sample)
  sampleRef.current = sample
  // Pointer moves can land between renders, so gestures read the latest colour from here.
  const latest = useRef(hsv)
  latest.current = hsv
  const hex = toHex(fromHsv(hsv))
  const rgb = toRgb(hex)

  useEffect(() => {
    if (value && value !== hex) setHsv(previous => hsvOf(value, previous))
  }, [value])

  // The plane only changes with hue; the strip is fixed (Okhsv at full s and v), drawn on first open.
  useEffect(() => {
    if (!open) return
    const basis = hueBasis(hsv.h)
    const [w, h] = PLANE
    paint(plane.current, w, h, (x, y) => okhsvToRgb(x / (w - 1), 1 - y / (h - 1), basis))
  }, [open, hsv.h])
  const stripPainted = useRef(false)
  useEffect(() => {
    if (!open || stripPainted.current) return
    stripPainted.current = true
    paint(strip.current, STRIP, 1, x => okhsvToRgb(1, 1, hueBasis((x / (STRIP - 1)) * 360)))
  }, [open])

  useEffect(() => {
    const popover = panel.current
    const trigger = button.current
    if (!popover || !trigger) return
    let stop: (() => void) | undefined
    const place = () => {
      const reference = trigger.parentElement ?? trigger
      void computePosition(reference, popover, {
        strategy: 'fixed',
        placement: 'bottom-start',
        middleware: [
          offset(3),
          flip({ padding: 8 }),
          shift({ padding: 8 }),
          size({
            padding: 8,
            apply: ({ rects, availableWidth }) => {
              popover.style.width = `${Math.min(availableWidth, Math.max(rects.reference.width, 224))}px`
            },
          }),
        ],
      }).then(({ x, y }) => {
        popover.style.left = `${x}px`
        popover.style.top = `${y}px`
        popover.dataset.placed = ''
      })
    }
    const onToggle = (event: Event) => {
      stop?.()
      stop = undefined
      const opening = (event as ToggleEvent).newState === 'open'
      setOpen(opening)
      if (!opening) {
        delete popover.dataset.placed
        return
      }
      stop = autoUpdate(trigger, popover, place)
      setSampled(sampleRef.current?.() ?? [])
    }
    popover.addEventListener('toggle', onToggle)
    return () => {
      popover.removeEventListener('toggle', onToggle)
      stop?.()
    }
  }, [])

  const preview = (next: Hsv) => {
    latest.current = next
    setHsv(next)
    const color = toHex(fromHsv(next))
    live.current = color
    onPreview(color)
  }
  const commit = (next?: Hsv) => {
    if (next) setHsv(next)
    const color = next ? toHex(fromHsv(next)) : live.current
    live.current = null
    if (color) onCommit(color)
  }
  const pick = (color: string) => commit(hsvOf(color, hsv))

  // One pointer capture per gesture: preview on every move, commit once on release.
  const drag = (read: (x: number, y: number) => Partial<Hsv>) => ({
    onPointerDown(event: PointerEvent) {
      if (event.button !== 0) return
      const target = event.currentTarget as HTMLElement
      target.setPointerCapture(event.pointerId)
      target.focus({ preventScroll: true })
      const move = (at: PointerEvent) => {
        const rect = target.getBoundingClientRect()
        const x = clamp((at.clientX - rect.left) / rect.width)
        const y = clamp((at.clientY - rect.top) / rect.height)
        preview({ ...latest.current, ...read(x, y) })
      }
      const end = () => {
        target.removeEventListener('pointermove', move)
        target.removeEventListener('pointerup', end)
        target.removeEventListener('pointercancel', end)
        commit()
      }
      move(event)
      target.addEventListener('pointermove', move)
      target.addEventListener('pointerup', end)
      target.addEventListener('pointercancel', end)
    },
  })
  const keys = (step: (dx: number, dy: number, scale: number) => Hsv | undefined) => ({
    onKeyDown(event: KeyboardEvent) {
      const delta = STEPS[event.key]
      if (!delta) return
      const next = step(delta[0], delta[1], event.shiftKey ? 10 : 1)
      if (!next) return
      event.preventDefault()
      preview(next)
    },
    onKeyUp(event: KeyboardEvent) {
      if (STEPS[event.key]) commit()
    },
  })

  const channel = (index: number, input: string) => {
    setDraft(current => current.map((value, i) => (i === index ? undefined : value)))
    const n = Number.parseInt(input, 10)
    if (Number.isNaN(n)) return
    const next = [...rgb] as Rgb
    next[index] = clamp(n, 0, 255)
    pick(toHex(next))
  }
  const eyeDropper =
    typeof window !== 'undefined' && 'EyeDropper' in window
      ? (window as unknown as { EyeDropper: new () => { open(): Promise<{ sRGBHex: string }> } })
          .EyeDropper
      : undefined
  const cells = PRESETS.length + (eyeDropper ? 1 : 0)

  return (
    <>
      <button
        ref={button}
        type="button"
        class="press-kit-editor-swatch"
        popovertarget={id}
        aria-label={label}
        data-empty={value ? undefined : ''}
      >
        <span style={value ? { background: value } : undefined} />
      </button>
      <div
        ref={panel}
        id={id}
        popover="auto"
        class="press-kit-color-picker"
        role="group"
        aria-label={label}
      >
        <div
          class="press-kit-color-plane"
          role="slider"
          tabIndex={0}
          aria-label="saturation et luminosité"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(hsv.s * 100)}
          aria-valuetext={`saturation ${Math.round(hsv.s * 100)} %, luminosité ${Math.round(hsv.v * 100)} %`}
          style={{ '--x': `${hsv.s * 100}%`, '--y': `${(1 - hsv.v) * 100}%` }}
          {...drag((x, y) => ({ s: x, v: 1 - y }))}
          {...keys((dx, dy, scale) => ({
            ...hsv,
            s: clamp(hsv.s + (dx * scale) / 100),
            v: clamp(hsv.v + (dy * scale) / 100),
          }))}
        >
          <canvas ref={plane} aria-hidden="true" />
        </div>
        <div
          class="press-kit-color-hue"
          role="slider"
          tabIndex={0}
          aria-label="teinte"
          aria-valuemin={0}
          aria-valuemax={360}
          aria-valuenow={Math.round(hsv.h)}
          style={{ '--x': `${(hsv.h / 360) * 100}%` }}
          {...drag(x => ({ h: x * 360 }))}
          {...keys((dx, dy, scale) => ({ ...hsv, h: clamp(hsv.h + (dx || dy) * scale, 0, 360) }))}
        >
          <canvas ref={strip} aria-hidden="true" />
        </div>
        <div class="press-kit-color-channels">
          {(['r', 'g', 'b'] as const).map((name, index) => (
            <label>
              <span>{name}</span>
              <input
                type="text"
                inputMode="numeric"
                maxLength={3}
                spellcheck={false}
                value={draft[index] ?? String(rgb[index])}
                onInput={event => {
                  const input = event.currentTarget.value
                  setDraft(current => {
                    const next = [...current]
                    next[index] = input
                    return next
                  })
                }}
                onChange={event => channel(index, event.currentTarget.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') channel(index, event.currentTarget.value)
                  if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
                  event.preventDefault()
                  const step = (event.key === 'ArrowUp' ? 1 : -1) * (event.shiftKey ? 10 : 1)
                  channel(index, String(rgb[index] + step))
                }}
              />
            </label>
          ))}
        </div>
        {sampled.length > 0 && (
          <div class="press-kit-color-row" role="group" aria-label="couleurs de la carte">
            <span>carte</span>
            <div class="press-kit-color-presets" style={{ '--cells': `${cells}` }}>
              {sampled.map(color => (
                <button
                  type="button"
                  aria-label={color}
                  aria-pressed={color === value}
                  style={{ background: color }}
                  onClick={() => pick(color)}
                />
              ))}
              {Array.from({ length: Math.max(0, cells - sampled.length) }, () => (
                <span aria-hidden="true" />
              ))}
            </div>
          </div>
        )}
        <div class="press-kit-color-presets" style={{ '--cells': `${cells}` }}>
          {PRESETS.map(color => (
            <button
              type="button"
              aria-label={color}
              aria-pressed={color === value}
              style={{ background: color }}
              onClick={() => pick(color)}
            />
          ))}
          {eyeDropper && (
            <button
              type="button"
              class="press-kit-color-dropper"
              aria-label="pipette : prendre une couleur à l’écran"
              title="pipette"
              onClick={() =>
                void new eyeDropper()
                  .open()
                  .then(({ sRGBHex }) => pick(sRGBHex.toLowerCase()))
                  .catch(() => {})
              }
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M2.75 13.25 9 7M7 5l4 4" />
                <rect
                  x="9.6"
                  y="1.4"
                  width="5"
                  height="3.4"
                  rx="1.7"
                  transform="rotate(45 12.1 3.1)"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
    </>
  )
}
