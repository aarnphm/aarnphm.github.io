export const MIME_MAPPINGS: Record<string, string[]> = {
  '.js': ['application/javascript', 'javascript'],
  '.jsx': ['application/javascript', 'javascriptreact'],
  '.mjs': ['application/javascript', 'javascript'],
  '.cjs': ['application/javascript', 'coffeescript'],
  '.ts': ['application/typescript', 'typescript'],
  '.tsx': ['application/typescript', 'typescript'],
  '.css': ['text/css', 'css'],
  '.scss': ['text/x-scss', 'sass'],
  '.sass': ['text/x-sass', 'sass'],
  '.m': ['text/x-matlab', 'matlab'],
  '.c': ['text/x-c', 'c'],
  '.h': ['text/x-c', 'c'],
  '.cpp': ['text/x-c++src', 'cpp'],
  '.cxx': ['text/x-c++src', 'cpp'],
  '.cc': ['text/x-c++src', 'cpp'],
  '.hpp': ['text/x-c++hdr', 'cpp'],
  '.hxx': ['text/x-c++hdr', 'cpp'],
  '.hh': ['text/x-c++hdr', 'cpp'],
  '.rs': ['text/rust', 'rust'],
  '.go': ['text/x-go', 'go'],
  '.zig': ['text/x-zig', 'zig'],
  '.cu': ['text/x-cuda', 'cuda'],
  '.cuh': ['text/x-cuda', 'cuda'],
  '.py': ['text/x-python', 'python'],
  '.lua': ['text/x-lua', 'lua'],
  '.json': ['application/json', 'json'],
  '.java': ['text/x-java', 'java'],
  '.sql': ['text/x-sql', 'sql'],
  '.txt': ['text/plain', 'plaintext'],
}

export const TEXT_EXTENSIONS = new Set(Object.keys(MIME_MAPPINGS))

export function getContentType(url: URL): string {
  const ext = url.pathname.toLowerCase().split('.').at(-1)
  return MIME_MAPPINGS[`.${ext}`] ? MIME_MAPPINGS[`.${ext}`][0] : 'text/plain'
}

export function getLang(url: URL): string {
  const ext = url.pathname.toLowerCase().split('.').at(-1)
  return MIME_MAPPINGS[`.${ext}`] ? MIME_MAPPINGS[`.${ext}`][1] : 'plaintext'
}
