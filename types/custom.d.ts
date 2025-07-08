declare module "preact/jsx-runtime" {
  export { JSX } from "preact"
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

// Minimal stub for 'preact' types in case they are not resolved
// This provides just enough typing information for JSX generics used in the codebase.
declare module "preact" {
  export namespace JSX {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    interface Element extends HTMLElement {}
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    interface IntrinsicElements {
      [elemName: string]: any
    }
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const h: any
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Fragment: any
}

// Stub for '@types/sharp' when not installed
declare module "sharp" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const sharp: any
  export = sharp
}

declare module "reading-time" {
  export interface ReadingTimeResult {
    text: string
    minutes: number
    time: number
    words: number
  }
  export default function readingTime(
    text: string,
    options?: { wpm?: number }
  ): ReadingTimeResult
}