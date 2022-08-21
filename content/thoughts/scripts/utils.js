export function escapePipe(str) {
  return new String(str).replace(/\|/, "\\|")
}
