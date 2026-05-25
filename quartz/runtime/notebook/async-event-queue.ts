export class AsyncEventQueue<T> {
  private values: T[] = []
  private waiters: ((next: IteratorResult<T>) => void)[] = []
  private failures: ((error: unknown) => void)[] = []
  private closed = false
  private error: unknown

  push(value: T): void {
    if (this.closed) return
    const waiter = this.waiters.shift()
    this.failures.shift()
    if (waiter) {
      waiter({ done: false, value })
    } else {
      this.values.push(value)
    }
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    for (const waiter of this.waiters.splice(0)) waiter({ done: true, value: undefined })
    this.failures.length = 0
  }

  fail(error: unknown): void {
    if (this.closed) return
    this.closed = true
    this.error = error
    for (const reject of this.failures.splice(0)) reject(error)
    this.waiters.length = 0
  }

  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    while (true) {
      if (this.values.length > 0) {
        yield this.values.shift()!
        continue
      }
      if (this.error !== undefined) throw this.error
      if (this.closed) return
      const next = await new Promise<IteratorResult<T>>((resolve, reject) => {
        this.waiters.push(resolve)
        this.failures.push(reject)
      })
      if (next.done) return
      yield next.value
    }
  }
}
