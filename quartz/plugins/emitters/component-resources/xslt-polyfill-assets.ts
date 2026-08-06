import fs from 'node:fs/promises'
import { createRequire } from 'node:module'
import type { BuildCtx } from '../../../util/ctx'
import type { FilePath } from '../../../util/path'
import { xsltPolyfillPath } from './asset-paths'
import { writeRawAsset } from './asset-writer'

const requireResolve = createRequire(import.meta.url).resolve

export async function writeXsltPolyfillAsset(ctx: BuildCtx): Promise<FilePath> {
  const source = await fs.readFile(requireResolve('xslt-polyfill'))
  return writeRawAsset(ctx, xsltPolyfillPath, source)
}
