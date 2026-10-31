export interface KindModule {
  family: 'leaf' | 'flower'
  name: string
  head?: boolean
  draw(k: Record<string, any>): unknown
}
export const HEADS: Set<string>
export function register(m: KindModule | KindModule[]): void
export function kind(family: string, name: string): KindModule | null
