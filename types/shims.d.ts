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
  export * from "@types/hast"
}

declare module "unist-util-visit" {
  import { Node } from "unist"
  type Visitor = (node: Node, index?: number, parent?: Node) => void
  // simplified typing
  export function visit(node: Node, test: any, visitor: Visitor): void
}