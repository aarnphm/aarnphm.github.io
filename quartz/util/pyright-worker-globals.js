import { Buffer as PyrightBuffer } from 'buffer'

const pyrightProcess = {
  env: {},
  execArgv: [],
  cwd: () => '/',
  memoryUsage: () => ({ heapUsed: 0, rss: 1 }),
}

const pyrightGlobal = globalThis

export { PyrightBuffer as Buffer, pyrightGlobal as global, pyrightProcess as process }
