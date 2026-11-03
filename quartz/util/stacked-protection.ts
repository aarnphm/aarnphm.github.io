// These nodes never enter a notes payload, including already-decrypted wrappers.
export const STACKED_PROTECTED_SELECTOR =
  '[data-notes-protected="true"], [data-protected="true"], [data-encrypted-content], .stream-entry-private, [data-protected-unlock-panel], [data-protected-unlock-all]'

export function protectedStackedNote(slug: string) {
  return {
    slug,
    title: 'protected note',
    content: '<div class="stacked-note-status">protected content is hidden in notes view.</div>',
    state: 'protected' as const,
  }
}
