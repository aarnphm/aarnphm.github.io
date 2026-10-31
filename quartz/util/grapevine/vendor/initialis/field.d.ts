export class Field {
  constructor(W: number, H: number, cell: number, cap: number)
  readonly W: number
  readonly H: number
  readonly cell: number
  readonly cap: number
  readonly nx: number
  readonly ny: number
  readonly d: Float32Array
  setBlocked(blocked: Uint8Array): void
  at(x: number, y: number): number
  atBase(x: number, y: number): number
  grad(x: number, y: number): [number, number]
  stamp(x: number, y: number, r: number): void
  biggest(): { x: number; y: number; d: number }
}
