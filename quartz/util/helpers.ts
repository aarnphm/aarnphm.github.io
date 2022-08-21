// Linear Congruential Generator implementation
export class LCG {
  private seed: number
  private m: number = 0x80000000 // 2**31
  private a: number = 1103515245
  private c: number = 42069

  constructor(seed: number) {
    this.seed = seed
  }

  next(): number {
    this.seed = (this.a * this.seed + this.c) % this.m
    return this.seed / this.m
  }

  // Get a random integer between min (inclusive) and max (exclusive)
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min) + min)
  }

  // Get a random item from an array
  choice<T>(array: T[]): T {
    return array[this.nextInt(0, array.length)]
  }

  // Shuffle array using Fisher-Yates algorithm
  shuffle<T>(array: T[]): T[] {
    const result = [...array]
    for (let i = result.length - 1; i > 0; i--) {
      const j = this.nextInt(0, i + 1)
      ;[result[i], result[j]] = [result[j], result[i]]
    }
    return result
  }
}
