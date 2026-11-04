import type { DocumentData } from './data'
import { kindMeta } from '../parts'

interface ReaderShortcutsProps {
  dialogRef: { current: HTMLDialogElement | null }
  canWrite: boolean
  signInUrl: string
  chrome: DocumentData['chrome']
}

export function ReaderShortcuts({ dialogRef, canWrite, signInUrl, chrome }: ReaderShortcutsProps) {
  return (
    <dialog ref={dialogRef} class="pdf-reader-keys" aria-labelledby="pdf-reader-keys-title">
      <h2 id="pdf-reader-keys-title">shortcuts</h2>
      <dl>
        {canWrite && (
          <>
            <dt>a · q · x</dt>
            <dd>
              mark the selection as {kindMeta.mark.label}, {kindMeta.question.label} or{' '}
              {kindMeta.contra.label}; on an active mark, change its kind
            </dd>
            <dt>n</dt>
            <dd>mark with a note, or a note on this page</dd>
            <dt>r</dt>
            <dd>drag a box around a figure or equation</dd>
            <dt>delete</dt>
            <dd>delete the active mark (⌘Z restores it)</dd>
          </>
        )}
        <dt>c</dt>
        <dd>copy the selection or active mark as a garden quote</dd>
        <dt>[ · ]</dt>
        <dd>previous or next mark</dd>
        <dt>+ · − · 0</dt>
        <dd>zoom in, out, fit width</dd>
        {chrome === 'course' && (
          <>
            <dt>&lt; · &gt; · u</dt>
            <dd>previous, next, up to the course</dd>
          </>
        )}
        <dt>esc</dt>
        <dd>clear the selection, tool or active mark</dd>
      </dl>
      {!canWrite && (
        <p>
          Marks here are aarnphm's. <a href={signInUrl}>Owner sign-in</a>
        </p>
      )}
      <form method="dialog">
        <button type="submit">close</button>
      </form>
    </dialog>
  )
}
