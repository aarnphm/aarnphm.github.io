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
  suffix: string
  proposition: string
  children?: ComponentChildren
  parentNumber?: string
}

type PropoProps = Omit<BasePropoProps, "parentNumber">

const isVNodeOf = <T,>(node: ComponentChildren, component: ComponentType<T>): node is VNode<T> => {
  return typeof node === "object" && node !== null && "type" in node && node.type === component
}

const TractatusPropoImpl: FunctionalComponent<BasePropoProps> = ({
  suffix,
  proposition,
  children,
  parentNumber = "",
}) => {
  const fullNumber = `${parentNumber}${suffix}`
  const childArray = toChildArray(children).filter(Boolean)
  const nested = childArray.filter((child) => isVNodeOf(child, TractatusPropoImpl))
  const content = childArray.filter((child) => !isVNodeOf(child, TractatusPropoImpl))

  const decoratedChildren = nested.map((child) =>
    cloneElement(child as VNode<BasePropoProps>, {
      parentNumber: fullNumber,
    }),
  )

  const depth = (fullNumber.match(/\./g) || []).length

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
          <p>{proposition}</p>
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
  proposition: string
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
  const childArray = toChildArray(children).filter(Boolean)
  const propos = childArray.filter(isPropoVNode)
  const content = childArray.filter((child) => !isPropoVNode(child))

  const decoratedPropos = propos.map((child) =>
    cloneElement(child as VNode<BasePropoProps>, {
      parentNumber: baseNumber,
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
          <p>{proposition}</p>
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
