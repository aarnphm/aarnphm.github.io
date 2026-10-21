import { z } from 'zod'
import type { NotebookRuntimeData } from '../runtime/notebook/types'
import type { NotebookDoc } from './notebook/types'

const count = z.number().int().nonnegative()
const notebookPreviewSchema = z.object({
  runtimes: z.array(z.string().min(1).max(64)),
  blocks: z.object({ code: count, markdown: count, raw: count }),
  description: z.string(),
})

export type NotebookPreview = z.infer<typeof notebookPreviewSchema>

export function buildNotebookPreview(
  doc: NotebookDoc,
  runtime: NotebookRuntimeData | undefined,
): NotebookPreview {
  const blocks = { code: 0, markdown: 0, raw: 0 }
  for (const cell of doc.cells) blocks[cell.cellType]++
  return {
    runtimes: [...new Set(runtime?.cells.map(cell => cell.language) ?? [])],
    blocks,
    description:
      typeof doc.metadata.description === 'string' ? doc.metadata.description.trim() : '',
  }
}

export function readNotebookPreview(raw: string | null | undefined): NotebookPreview | null {
  if (!raw) return null
  try {
    const result = notebookPreviewSchema.safeParse(JSON.parse(raw))
    return result.success ? result.data : null
  } catch {
    return null
  }
}
