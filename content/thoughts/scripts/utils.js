export function escapePipe(str) {
  return String(str).replaceAll('|', '\\|')
}
