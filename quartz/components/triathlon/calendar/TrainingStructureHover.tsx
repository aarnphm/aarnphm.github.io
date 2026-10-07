import { autoUpdate, computePosition, flip, hide, offset, shift } from '@floating-ui/dom'
import { createContext, type ComponentChildren } from 'preact'
import { useContext, useLayoutEffect, useRef, useState } from 'preact/hooks'

interface StructureHover {
  workoutId: string
  step: string
  lap: number
  title: string
  details: string[]
  anchor: HTMLElement
}

const HoverContext = createContext<{
  active: StructureHover | null
  tooltipId: string
  show: (hover: StructureHover) => void
  clear: (anchor: HTMLElement) => void
} | null>(null)

export const useStructureHover = () => useContext(HoverContext)

export const TrainingStructureHover = ({
  children,
  id,
  resetKey,
}: {
  children: ComponentChildren
  id: string
  resetKey: string
}) => {
  const [active, setActive] = useState<StructureHover | null>(null)
  const tip = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => setActive(null), [resetKey])
  useLayoutEffect(() => {
    const tooltip = tip.current
    if (!active || !tooltip) return
    let disposed = false
    const update = async () => {
      const { x, y, middlewareData } = await computePosition(active.anchor, tooltip, {
        strategy: 'fixed',
        placement: 'top',
        middleware: [offset(8), flip({ padding: 8 }), shift({ padding: 8 }), hide()],
      })
      if (disposed) return
      Object.assign(tooltip.style, {
        left: `${x}px`,
        top: `${y}px`,
        visibility: middlewareData.hide?.referenceHidden ? 'hidden' : 'visible',
      })
    }
    const cleanup = autoUpdate(active.anchor, tooltip, update)
    return () => {
      disposed = true
      cleanup()
    }
  }, [active])
  return (
    <HoverContext.Provider
      value={{
        active,
        tooltipId: id,
        show: setActive,
        clear: anchor => setActive(current => (current?.anchor === anchor ? null : current)),
      }}
    >
      {children}
      {active && (
        <div ref={tip} id={id} class="tri-training-structure-tip" role="tooltip">
          <strong>{active.title}</strong>
          {active.details.map((detail, index) => (
            <span key={index}>{detail}</span>
          ))}
        </div>
      )}
    </HoverContext.Provider>
  )
}
