import { build as bundle } from 'esbuild'
import fs from 'node:fs/promises'
import path from 'path'

export async function bundleInlineScript(scriptPath: string): Promise<string> {
  let text = await fs.readFile(scriptPath, 'utf8')
  text = text.replace('export default', '')
  text = text.replace('export', '')

  const sourcefile = path.relative(path.resolve('.'), scriptPath)
  const transpiled = await bundle({
    stdin: {
      contents: text,
      loader: path.extname(scriptPath) === '.js' ? 'js' : 'ts',
      resolveDir: path.dirname(sourcefile),
      sourcefile,
    },
    write: false,
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    loader: { '.py': 'text' },
  })
  return transpiled.outputFiles[0].text
}
