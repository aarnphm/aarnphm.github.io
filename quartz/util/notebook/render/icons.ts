export type NotebookIcon = 'run' | 'stop' | 'edit' | 'save' | 'revert' | 'copy' | 'check'

export const notebookIconSvg: Record<NotebookIcon, string> = {
  run: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13l10-6.5z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M7 7h10v10H7z"/></svg>',
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m4 16.5-.5 4 4-.5L19 8.5 15.5 5z"/><path d="m14 6.5 3.5 3.5"/></svg>',
  save: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8V4"/><path d="M8 20v-6h8v6"/></svg>',
  revert:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>',
  copy: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 8h11v11H8z"/><path d="M5 16H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>',
  check:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m5 12 4 4L19 6"/></svg>',
}
