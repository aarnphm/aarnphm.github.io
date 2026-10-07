import { useState } from 'preact/hooks'
import type { TrainingPeaksCalendarNote } from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'

export interface TrainingNote {
  key: string
  title: string | null
  body: string
}

/** The owner keeps `open` and decides which hover area opens the tooltip. */
export const TrainingNotesButton = ({
  id,
  label,
  open,
  setOpen,
}: {
  id: string
  label: string
  open: boolean
  setOpen: (open: boolean) => void
}) => (
  <button
    type="button"
    class="tri-training-notes-button"
    aria-label={label}
    aria-describedby={id}
    onFocus={() => setOpen(true)}
    onBlur={() => setOpen(false)}
    onClick={() => setOpen(true)}
    onKeyDown={event => {
      if (event.key !== 'Escape' || !open) return
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
    }}
  >
    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path d="M4 2.5h5l3 3v8H4zM9 2.5v3h3M6 8.5h4M6 11h3" />
    </svg>
  </button>
)

export const TrainingNotesTip = ({
  id,
  notes,
  open,
}: {
  id: string
  notes: readonly TrainingNote[]
  open: boolean
}) => (
  <span class="tri-training-notes-tip" id={id} role="tooltip" hidden={!open}>
    {notes.map(note => (
      <span key={note.key} class="tri-training-note">
        {note.title && <strong>{note.title}</strong>}
        {note.body}
      </span>
    ))}
  </span>
)

/** Calendar notes for one day, shown while the day header is hovered. */
export const TrainingDayNotes = ({
  id,
  notes,
  formatter,
}: {
  id: string
  notes: readonly TrainingPeaksCalendarNote[]
  formatter: TriathlonFormatter
}) => {
  const [open, setOpen] = useState(false)
  if (notes.length === 0) return null
  return (
    <span
      class="tri-training-day-notes"
      onPointerEnter={() => setOpen(true)}
      onPointerLeave={() => setOpen(false)}
    >
      <TrainingNotesButton
        id={id}
        label={`${formatter.text('day notes')}: ${notes.length}`}
        open={open}
        setOpen={setOpen}
      />
      <TrainingNotesTip
        id={id}
        open={open}
        notes={notes.map(note => ({ key: note.id, title: note.title, body: note.description }))}
      />
    </span>
  )
}
