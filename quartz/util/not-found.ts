export function missingSlugMarkdown(pathname: string): string {
  const slug = pathname.replace(/\.md$/, '').replace(/\/$/, '') || '/'
  return `slug is not found or private: ${slug}\n`
}
