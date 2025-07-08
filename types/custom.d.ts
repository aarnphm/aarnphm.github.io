declare module "preact/src/jsx" {
  import { JSX } from "preact"
  export type JSXInternal = JSX
  export { JSX }
}

declare module "node:util" {
  export * from "util"
  /**
   * Minimal type for Node.js 20+ `styleText` helper used in codebase
   */
  export function styleText(color: string, text: string): string
}

declare module "hast-util-from-html" {
  import type { Root } from "hast"
  interface Options {
    fragment?: boolean
  }
  export function fromHtml(value: string, options?: Options): Root
}

// Fallback typings for satori when not resolved correctly
declare module "satori" {
  const satori: any
  export = satori
}