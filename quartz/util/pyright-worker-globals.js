import { Buffer as PyrightBuffer } from 'buffer'

const pyrightProcess = {
  env: {},
  execArgv: [],
  cwd: () => '/',
  memoryUsage: () => ({ heapUsed: 0, rss: 1 }),
}

const pyrightGlobal = globalThis
const pyrightDirname = '/'
const pyrightFilename = '/pyright-worker.js'

export {
  PyrightBuffer as Buffer,
  pyrightDirname as __dirname,
  pyrightFilename as __filename,
  pyrightGlobal as global,
  pyrightProcess as process,
}
