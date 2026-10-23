export const CODE_FILE_EXTENSIONS = new Set<string>([
  '.py',
  '.mojo',
  '.rs',
  '.go',
  '.c',
  '.cu',
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
])

export function languageFromExt(ext: string): string | undefined {
  const e = ext.replace(/^\./, '').toLowerCase()
  switch (e) {
    case 'py':
      return 'python'
    case 'mojo':
      return 'mojo'
    case 'ts':
    case 'tsx':
      return e
    case 'js':
    case 'jsx':
      return e
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
      return 'cuda'
    case 'lean':
      return 'lean'
    default:
      return e
  }
}
