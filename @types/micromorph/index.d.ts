declare module "micromorph" {
  export interface Patch {
    type: number
    [key: string]: unknown
  }

  export default function micromorph(from: Node, to: Node): Promise<void>
  export function diff(from: Node | undefined, to: Node | undefined): Patch | undefined
  export function patch(container: Node, patch: Patch): Promise<void>
}
