import type { Mutex } from 'async-mutex'
import type { BuildOptions, Metafile } from 'esbuild'
import type { Server } from 'node:http'
import type { WebSocketServer } from 'ws'
import type { Argv } from '../util/ctx'

export function createBuildConfig(): BuildOptions
export function parseSourceInputs(metafile: Metafile): string[]
export function courseSourceInputs(metafile: Metafile): Set<string>
export function createPreviewServer(argv: Argv & { baseDir: string }, mutex: Mutex): Server
export function createReloadServer(port: number): { server: WebSocketServer; refresh(): void }
