import { parseNotebookDoc } from '../../util/notebook/parse'
import { isNotebookParseError } from '../../util/notebook/types'

export function notebookRuntimeModuleSource(raw: string, sourcePath: string): string {
  const parsed = parseNotebookDoc(raw, sourcePath)
  if (isNotebookParseError(parsed)) {
    throw new Error(`${sourcePath} is not a valid notebook`)
  }
  return parsed.cells
    .filter(cell => cell.cellType === 'code')
    .map(cell => cell.source.trim())
    .filter(source => source.length > 0)
    .join('\n\n')
}
