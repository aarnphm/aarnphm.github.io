import path from 'node:path'
import type { BuildCtx } from '../../util/ctx'
import { CodeSearchBlocks } from '../transformers/code-search'
export function codeSearchTestContext(root: string): BuildCtx {
  const color = {
    light: '#fff',
    lightgray: '#eee',
    gray: '#aaa',
    darkgray: '#444',
    dark: '#000',
    secondary: '#00f',
    tertiary: '#080',
    highlight: '#eee',
    textHighlight: '#ff0',
  }
  return {
    buildId: 'code-publication-test',
    incremental: true,
    argv: {
      directory: path.join(root, 'content'),
      output: path.join(root, 'public'),
      verbose: false,
      serve: false,
      watch: false,
      port: 0,
      wsPort: 0,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'test',
        enableSPA: false,
        enablePopovers: false,
        analytics: null,
        ignorePatterns: ['ignored/**'],
        defaultDateType: 'modified',
        locale: 'en-US',
        theme: {
          typography: { header: 'sans', body: 'sans', code: 'mono' },
          cdnCaching: false,
          fontOrigin: 'local',
          colors: { lightMode: color, darkMode: color },
        },
      },
      plugins: { transformers: [CodeSearchBlocks()], filters: [], emitters: [] },
    },
    allFiles: [],
    allSlugs: [],
  }
}
