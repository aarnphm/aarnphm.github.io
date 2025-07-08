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

// Minimal fallback typings for `satori/wasm` if not present
declare module "satori/wasm" {
  export type FontWeight = number
  export interface SatoriOptions {
    width?: number
    height?: number
    fonts?: Array<{
      name: string
      data: ArrayBuffer | Buffer
      weight?: FontWeight
      style?: "normal" | "italic"
    }>
    /** additional unknown fields */
    [key: string]: unknown
  }
}

// Minimal fallback if JSX namespace is not found (ensures TSX compiles in isolated modules)
declare namespace JSX {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  interface IntrinsicElements {
    // Allow any tag
    [elemName: string]: any
  }
}