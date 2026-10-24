export const CODE_FILE_EXTENSIONS = new Set<string>([
  '.py',
  '.pyi',
  '.mojo',
  '.rs',
  '.go',
  '.c',
  '.cu',
  '.cuh',
  '.cc',
  '.cpp',
  '.h',
  '.hpp',
  '.m',
  '.mm',
  '.java',
  '.kt',
  '.swift',
  '.scala',
  '.ts',
  '.tsx',
  '.js',
  '.cjs',
  '.mjs',
  '.jsx',
  '.sh',
  '.bash',
  '.zsh',
  '.fish',
  '.sql',
  '.yaml',
  '.yml',
  '.toml',
  '.json',
  '.mdx',
  '.css',
  '.scss',
  '.hs',
  '.rb',
  '.php',
  '.lean',
  '.wat',
  '.wast',
  '.ml',
  '.mli',
  '.nix',
  '.zig',
  '.lua',
])

export function languageFromExt(ext: string): string | undefined {
  const e = ext.replace(/^\./, '').toLowerCase()
  switch (e) {
    case 'py':
    case 'pyi':
      return 'python'
    case 'mojo':
      return 'mojo'
    case 'ts':
    case 'tsx':
      return e
    case 'js':
    case 'jsx':
      return e
    case 'cjs':
    case 'mjs':
      return 'js'
    case 'rs':
      return 'rust'
    case 'go':
      return 'go'
    case 'c':
      return 'c'
    case 'cc':
    case 'cpp':
    case 'hpp':
      return 'cpp'
    case 'h':
      return 'c'
    case 'm':
      return 'matlab'
    case 'mm':
      return 'objective-c'
    case 'java':
      return 'java'
    case 'kt':
      return 'kotlin'
    case 'swift':
      return 'swift'
    case 'scala':
      return 'scala'
    case 'sh':
    case 'bash':
    case 'zsh':
    case 'fish':
      return 'bash'
    case 'sql':
      return 'sql'
    case 'yaml':
    case 'yml':
      return 'yaml'
    case 'toml':
      return 'toml'
    case 'json':
      return 'json'
    case 'mdx':
      return 'mdx'
    case 'css':
    case 'scss':
      return e
    case 'hs':
      return 'haskell'
    case 'rb':
      return 'ruby'
    case 'php':
      return 'php'
    case 'cu':
    case 'cuh':
      return 'cuda'
    case 'lean':
      return 'lean'
    case 'wat':
    case 'wast':
      return 'wasm'
    case 'ml':
    case 'mli':
      return 'ocaml'
    default:
      return e
  }
}

/** Canonical parser identity; source labels remain available for display. */
export function canonicalCodeLanguage(language: string): string {
  const normalized = language.trim().replace(/^\./, '').toLowerCase()
  switch (normalized) {
    case 'js':
    case 'jsx':
    case 'javascript':
    case 'cjs':
    case 'mjs':
      return 'javascript'
    case 'ts':
    case 'typescript':
      return 'typescript'
    case 'py':
    case 'pyi':
    case 'py3':
    case 'python3':
      return 'python'
    case 'c++':
      return 'cpp'
    case 'shell':
    case 'shellscript':
    case 'console':
      return 'bash'
    case 'objc':
    case 'objectivec':
      return 'objective-c'
    case 'golang':
      return 'go'
    case 'lean4':
      return 'lean'
    case 'webassembly':
      return 'wasm'
    case '':
    case 'text':
    case 'plaintext':
    case 'plain':
      return 'plain'
    default:
      return languageFromExt(normalized) ?? normalized
  }
}
