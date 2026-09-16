declare module 'defuddle/full' {
  const source: string
  export default source
}

declare module 'dompurify/dist/purify.js' {
  const source: string
  export default source
}

interface Env {
  ARENA_READER_DEV?: string
  ARENA_OWNER_LOGIN?: string
}
