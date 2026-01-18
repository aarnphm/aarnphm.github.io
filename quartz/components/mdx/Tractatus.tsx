import {
  cloneElement,
  type ComponentChildren,
  type ComponentType,
  type FunctionalComponent,
  toChildArray,
  type VNode,
} from "preact"
import {
  registerMdxComponent,
  type QuartzMdxComponent,
  type QuartzMdxConstructor,
} from "./registry"

type BasePropoProps = {
  suffix?: string
  proposition?: ComponentChildren
  children?: ComponentChildren
  parentNumber?: string
  _index?: number
  _depth?: number
}

type PropoProps = Omit<BasePropoProps, "parentNumber">

const isVNodeOf = <T,>(node: ComponentChildren, component: ComponentType<T>): node is VNode<T> => {
  return typeof node === "object" && node !== null && "type" in node && node.type === component
}

const normalizeChildren = (children?: ComponentChildren): ComponentChildren[] =>
  toChildArray(children).filter((child) => {
    if (child === null || child === undefined) return false
    if (typeof child === "string") return child.trim() !== ""
    return true
  })

const splitChildren = <T,>(
  children: ComponentChildren,
  isNested: (node: ComponentChildren) => node is VNode<T>,
) => {
  const childArray = normalizeChildren(children)
  const nested: ComponentChildren[] = []
  const leading: ComponentChildren[] = []
  const trailing: ComponentChildren[] = []
  let seenNested = false

  for (const child of childArray) {
    if (isNested(child)) {
      seenNested = true
      nested.push(child)
      continue
    }
    if (!seenNested) {
      leading.push(child)
    } else {
      trailing.push(child)
    }
  }

  return { nested, leading, trailing }
}

const computeChildNumber = (parentNumber: string, index: number): string => {
  const segments = parentNumber.split(".")
  const last = segments[segments.length - 1]
  const prefix = segments.slice(0, -1).join(".")
  const suffix = `${last}${index + 1}`
  if (prefix.length === 0) {
    return `${last}.${suffix}`
  }
  return `${prefix}.${suffix}`
}

const TractatusPropoImpl: FunctionalComponent<BasePropoProps> = ({
  suffix,
  proposition,
  children,
  parentNumber = "",
  _index,
  _depth,
}) => {
  const autoNumber =
    suffix === undefined && parentNumber.length > 0 && _index !== undefined
      ? computeChildNumber(parentNumber, _index)
      : null
  const manualNumber =
    suffix === undefined
      ? parentNumber
      : parentNumber.length > 0
        ? `${parentNumber}${suffix}`
        : suffix
  const fullNumber = autoNumber ?? manualNumber ?? ""
  const { nested, leading, trailing } = splitChildren(children, (child) =>
    isVNodeOf(child, TractatusPropoImpl),
  )
  const propositionContent = proposition ?? (leading.length > 0 ? leading : undefined)
  const content = proposition ? [...leading, ...trailing] : trailing

  const decoratedChildren = (nested as VNode<BasePropoProps>[]).map((child, idx) =>
    cloneElement(child as VNode<BasePropoProps>, {
      parentNumber: fullNumber,
      _index: idx,
      _depth: (_depth ?? (fullNumber.match(/\./g) || []).length) + 1,
    }),
  )

  const depth = _depth ?? (fullNumber.match(/\./g) || []).length

  return (
    <li
      class="tractatus-item"
      id={`tractatus-${fullNumber.replace(/[^0-9A-Za-z.-]/g, "")}`}
      data-tractatus-number={fullNumber}
      data-tractatus-depth={depth}
      style={`--tractatus-depth: ${depth};`}
    >
      <div class="tractatus-row">
        <span class="tractatus-number">{fullNumber}</span>
        <div class="tractatus-text">
          {propositionContent && <p>{propositionContent}</p>}
          {content.length > 0 && <div class="tractatus-content">{content}</div>}
        </div>
      </div>
      {decoratedChildren.length > 0 && <ul class="tractatus-list">{decoratedChildren}</ul>}
    </li>
  )
}

const TractatusPropoComponent = TractatusPropoImpl as QuartzMdxComponent<PropoProps>
export const TractatusPropo = registerMdxComponent(
  ["TractatusPropo", "TractatusProposition"],
  TractatusPropoComponent,
)

type TractatusProps = {
  proposition?: ComponentChildren
  children?: ComponentChildren
  number?: number
}

type TractatusInternalProps = TractatusProps & {
  _index?: number
}

const isPropoVNode = (node: ComponentChildren): node is VNode<BasePropoProps> =>
  isVNodeOf(node, TractatusPropo)

const TractatusImpl: FunctionalComponent<TractatusInternalProps> = ({
  proposition,
  children,
  number,
  _index = 1,
}) => {
  const baseNumber = number !== undefined ? String(number) : String(_index)
  const { nested: propos, leading, trailing } = splitChildren(children, isPropoVNode)
  const propositionContent = proposition ?? (leading.length > 0 ? leading : undefined)
  const content = proposition ? [...leading, ...trailing] : trailing

  const decoratedPropos = (propos as VNode<BasePropoProps>[]).map((child, idx) =>
    cloneElement(child as VNode<BasePropoProps>, {
      parentNumber: baseNumber,
      _index: idx,
      _depth: 1,
    }),
  )

  return (
    <li
      class="tractatus-item"
      id={`tractatus-${baseNumber}`}
      data-tractatus-number={baseNumber}
      data-tractatus-depth={0}
      style="--tractatus-depth: 0;"
    >
      <div class="tractatus-row">
        <span class="tractatus-number">{baseNumber}</span>
        <div class="tractatus-text">
          {propositionContent && <p>{propositionContent}</p>}
          {content.length > 0 && <div class="tractatus-content">{content}</div>}
        </div>
      </div>
      {decoratedPropos.length > 0 && <ul class="tractatus-list">{decoratedPropos}</ul>}
    </li>
  )
}

const TractatusComponent = TractatusImpl as QuartzMdxComponent<TractatusProps>
export const Tractatus = registerMdxComponent("Tractatus", TractatusComponent)

type TractatusRootProps = {
  children?: ComponentChildren
}

const isTractatus = (node: ComponentChildren): node is VNode<TractatusInternalProps> =>
  isVNodeOf(node, Tractatus)

const TractatusRootImpl: QuartzMdxComponent<TractatusRootProps> = ({ children }) => {
  const childArray = toChildArray(children).filter(Boolean)
  const tractati = childArray.filter(isTractatus)
  const trailing = childArray.filter((child) => !isTractatus(child))

  const decoratedTractati = tractati.map((child, idx) =>
    cloneElement(child as VNode<TractatusInternalProps>, {
      _index: idx + 1,
    }),
  )

  return (
    <section class="tractatus-root">
      <ul class="tractatus-list">{decoratedTractati}</ul>
      {trailing.length > 0 && <div class="tractatus-trailing">{trailing}</div>}
    </section>
  )
}

export const TractatusRoot = registerMdxComponent("TractatusRoot", TractatusRootImpl)

export default (() => TractatusRoot) satisfies QuartzMdxConstructor
