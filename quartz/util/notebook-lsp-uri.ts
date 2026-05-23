export function notebookDocumentPath(runtimeId: string): string {
  return `/notebook/${runtimeId}/notebook.py`
}

export function notebookDocumentUri(runtimeId: string): string {
  return `file://${notebookDocumentPath(runtimeId)}`
}

export function notebookWorkspaceRootUri(runtimeId: string): string {
  return `file:///notebook/${runtimeId}/`
}
