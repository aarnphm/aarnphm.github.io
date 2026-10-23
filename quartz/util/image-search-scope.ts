export function isCourseMaterialPath(sourcePath: string): boolean {
  const relative = sourcePath
    .replace(/\\/g, '/')
    .replace(/^\.?\//, '')
    .replace(/^content\//, '')
  return ['courses', 'thoughts/university', 'lectures', 'psets', 'statements'].some(
    prefix => relative === prefix || relative.startsWith(`${prefix}/`),
  )
}
