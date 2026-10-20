import type { PdfMarkKind } from '../../util/pdf-marks'
import { kindMeta } from '../pdf-reader/parts'
import { SelectionToolbar } from '../selection-toolbar'

export const noteKinds: PdfMarkKind[] = ['mark', 'question', 'contra']
export type SelectionAction = PdfMarkKind | 'note' | 'scratchpad'

export function SelectionActions({
  range,
  onAction,
}: {
  range: Range
  onAction(action: SelectionAction): void
}) {
  return (
    <SelectionToolbar range={range} label="Lecture selection">
      {noteKinds.map(kind => (
        <button type="button" onClick={() => onAction(kind)}>
          {kindMeta[kind].glyph} {kindMeta[kind].label}
        </button>
      ))}
      <button type="button" onClick={() => onAction('note')}>
        note
      </button>
      <button type="button" onClick={() => onAction('scratchpad')}>
        → scratchpad
      </button>
    </SelectionToolbar>
  )
}
