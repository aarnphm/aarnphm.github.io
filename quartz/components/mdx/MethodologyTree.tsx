import {
  cloneElement,
  type ComponentChildren,
  type ComponentType,
  type FunctionalComponent,
  toChildArray,
  type VNode,
} from "preact"
//@ts-ignore
import script from "../scripts/methodology-tree.inline"
import style from "../styles/methodologyTree.scss"
import {
  registerMdxComponent,
  type QuartzMdxComponent,
  type QuartzMdxConstructor,
} from "./registry"

type StepPath = number[]

type BaseStepProps = {
  title: string
  summary?: string
  badge?: string
  points?: string[]
  highlight?: string
  children?: ComponentChildren
  depth?: number
  path?: StepPath
}

type StepProps = Omit<BaseStepProps, "depth" | "path">

const isVNodeOf = <T,>(node: ComponentChildren, component: ComponentType<T>): node is VNode<T> => {
  return typeof node === "object" && node !== null && "type" in node && node.type === component
}

function toStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item))
  }

  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return []
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) {
        return parsed.map((item) => String(item))
      }
    } catch {
      try {
        // fallback for JS array syntax with trailing commas/newlines
        const evaluated = Function(`return (${trimmed})`)()
        if (Array.isArray(evaluated)) {
          return evaluated.map((item: unknown) => String(item))
        }
      } catch {
        // ignore
      }
    }

    return trimmed
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
  }

  return []
}

const MethodologyStepImpl: FunctionalComponent<BaseStepProps> = ({
  title,
  summary,
  badge,
  points,
  highlight,
  children,
  depth = 0,
  path = [],
}) => {
  const normalizedPoints = toStringArray(points)
  const childArray = toChildArray(children).filter(Boolean)
  const nested = childArray.filter((child) => isVNodeOf(child, MethodologyStepImpl))
  const inline = childArray.filter((child) => !isVNodeOf(child, MethodologyStepImpl))

  const sequenceIndex = path[path.length - 1] ?? depth + 1
  const letterCode = ((sequenceIndex - 1) % 26) + 65
  const marker = String.fromCharCode(letterCode)

  const decoratedChildren = nested.map((child, idx) =>
    cloneElement(child as VNode<BaseStepProps>, {
      depth: depth + 1,
      path: [...path, idx + 1],
    }),
  )

  const pathKey = path.length > 0 ? path.join("-") : `${depth}-${sequenceIndex}`
  const toggleId = `methodology-step-toggle-${pathKey}`
  const bodyId = `methodology-step-body-${pathKey}`

  return (
    <div
      class={`methodology-step depth-${depth} is-open`}
      data-methodology-step
      data-initial-open="true"
    >
      <button
        type="button"
        class="step-toggle"
        id={toggleId}
        aria-expanded="true"
        aria-controls={bodyId}
        data-step-toggle
      >
        <span class="step-rail" aria-hidden="true">
          <span class="step-line step-line--before"></span>
          <span class="step-node">
            <span class="step-sequence">{marker}</span>
          </span>
          <span class="step-line step-line--after"></span>
        </span>
        <span class="step-header">
          <span class="step-title">{title}</span>
          {badge && <span class="step-badge">{badge}</span>}
        </span>
      </button>
      <div class="step-body" id={bodyId} role="region" aria-labelledby={toggleId} data-step-body>
        <span class="step-rail step-rail--body" aria-hidden="true">
          <span class="step-line step-line--body"></span>
        </span>
        <div class="step-body-content">
          {summary && <p class="step-summary">{summary}</p>}
          {highlight && <p class="step-highlight">{highlight}</p>}
          {normalizedPoints.length > 0 && (
            <ul class="step-points">
              {normalizedPoints.map((point) => (
                <li>{point}</li>
              ))}
            </ul>
          )}
          {inline.length > 0 && <div class="step-inline">{inline}</div>}
          {decoratedChildren.length > 0 && (
            <div class="methodology-children">{decoratedChildren}</div>
          )}
        </div>
      </div>
    </div>
  )
}

const MethodologyStepComponent = MethodologyStepImpl as QuartzMdxComponent<StepProps>
MethodologyStepComponent.css = style
MethodologyStepComponent.afterDOMLoaded = script
export const MethodologyStep = registerMdxComponent(
  ["MethodologyStep", "MethodologyNode", "MethodologyLeaf"],
  MethodologyStepComponent,
)

const isStepVNode = (node: ComponentChildren): node is VNode<BaseStepProps> =>
  isVNodeOf(node, MethodologyStep)

type TreeProps = {
  title?: string
  description?: string
  children?: ComponentChildren
  compact?: boolean
  asTitle?: boolean
}

const MethodologyTreeImpl: QuartzMdxComponent<TreeProps> = ({
  title,
  description,
  children,
  compact = false,
  asTitle = true,
}) => {
  const childArray = toChildArray(children).filter(Boolean)
  const steps = childArray.filter(isStepVNode)
  const trailing = childArray.filter((child) => !isStepVNode(child))

  const decoratedSteps = steps.map((child, idx) =>
    cloneElement(child as VNode<BaseStepProps>, {
      depth: 0,
      path: [idx + 1],
    }),
  )

  return (
    <section class={`methodology-tree${compact ? " compact" : ""}`}>
      {(title || description) && (
        <header class="tree-header">
          {title && (asTitle ? <h4>{title}</h4> : <p>{title}</p>)}
          {description && <p>{description}</p>}
        </header>
      )}
      <div class="tree-root">{decoratedSteps}</div>
      {trailing.length > 0 && <div class="tree-trailing">{trailing}</div>}
    </section>
  )
}
MethodologyTreeImpl.css = style
MethodologyTreeImpl.afterDOMLoaded = script

export const MethodologyTree = registerMdxComponent("MethodologyTree", MethodologyTreeImpl)

export default (() => MethodologyTree) satisfies QuartzMdxConstructor
