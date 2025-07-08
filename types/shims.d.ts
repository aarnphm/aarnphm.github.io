declare module "hastscript" {
  import { Element, ElementContent, Properties } from "hast"
  // eslint-disable-next-line @typescript-eslint/ban-types
  export function h<T extends string>(
    tagName: T,
    properties?: Properties | null,
    children?: ElementContent | ElementContent[],
  ): Element
  export function h(tagName: string, children?: ElementContent | ElementContent[]): Element
  export default h
}

declare module "hast" {
  export interface Node {
    type: string
  }
  export interface Element extends Node {
    tagName: string
    properties?: Record<string, any>
    children: Element[] | any[]
  }
  export interface Root extends Node {
    children: Element[] | any[]
  }
  export type ElementContent = Element | any
}

declare module "unist-util-visit" {
  import { Node } from "unist"
  type Visitor = (node: Node, index?: number, parent?: Node) => void
  // simplified typing
  export function visit(node: Node, test: any, visitor: Visitor): void
}