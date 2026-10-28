import { notebookCLanguageLogoSvg } from './icons/c'
import { notebookCppLanguageLogoSvg } from './icons/cpp'
import { notebookGoLanguageLogoSvg } from './icons/go'
import { notebookOcamlLanguageLogoSvg } from './icons/ocaml'
import { notebookRustLanguageLogoSvg } from './icons/rust'
import { notebookWasmLanguageLogoSvg } from './icons/wasm'

export type NotebookIcon =
  | 'run'
  | 'stop'
  | 'reset'
  | 'debug'
  | 'edit'
  | 'save'
  | 'revert'
  | 'copy'
  | 'check'
  | 'expand'
  | 'vim'

export const notebookIconSvg: Record<NotebookIcon, string> = {
  run: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5v14l11-7z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 6h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z"/></svg>',
  debug:
    '<svg class="notebook-debug-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M19 12a7 7 0 1 1-14 0 7 7 0 1 1 14 0"/><path class="notebook-debug-icon-core" d="M16 12a4 4 0 1 1-8 0 4 4 0 1 1 8 0"/></svg>',
  reset:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5.7 12a6.3 6.3 0 1 0 6.3-6.3 6.83 6.83 0 0 0-4.72 1.92L5.7 9.2"/><path d="M5.7 5.7v3.5h3.5"/></svg>',
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m4 16.5-.5 4 4-.5L19 8.5 15.5 5z"/><path d="m14 6.5 3.5 3.5"/></svg>',
  save: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v6h8V4"/><path d="M8 20v-6h8v6"/></svg>',
  revert:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>',
  copy: '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 8h11v11H8z"/><path d="M5 16H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>',
  check:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m5 12 4 4L19 6"/></svg>',
  expand:
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 5v14"/><path d="m6 13 6 6 6-6"/></svg>',
  vim: [
    '<svg class="notebook-vim-icon" viewBox="0 0 602 734" aria-hidden="true" focusable="false">',
    '<g transform="translate(2 3)">',
    '<path class="notebook-vim-icon-left" d="M0 155.5704 155-1l-.000003 728L0 572.237919z"/>',
    '<path class="notebook-vim-icon-right" d="M443.060403 156.982405 600-1l-3.181208 728L442 572.219941z" transform="translate(521 363.5) scale(-1 1) translate(-521 -363.5)"/>',
    '<path class="notebook-vim-icon-cross" d="M154.986294 0 558 615.189696 445.224605 728 42 114.172017z"/>',
    '</g>',
    '</svg>',
  ].join(''),
}

export const notebookLanguageIconSvg: Readonly<Record<string, string>> = {
  bash: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6.5h16v11H4zM7 10l2 2-2 2m5 0h4"/></svg>',
  c: notebookCLanguageLogoSvg,
  cpp: notebookCppLanguageLogoSvg,
  css: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M5 3h14l-1.3 15.1L12 21l-5.7-2.9z"/><path fill="var(--light)" d="M9 7h7l-.2 2H11l.1 1.5h4.5l-.5 5.2-3.1.9-3.1-.9-.2-2.4h2l.1 1 1.2.3 1.2-.3.2-1.8H8.6z"/></svg>',
  go: notebookGoLanguageLogoSvg,
  html: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M5 3h14l-1.3 15.1L12 21l-5.7-2.9z"/><path fill="var(--light)" d="M8.6 7h6.8l-.2 2h-4.5l.1 1.5H15l-.5 5.2-2.5.8-2.5-.8-.2-2.4h2l.1.9.6.2.6-.2.2-1.7H9z"/></svg>',
  java: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.9" d="M9 17h6a3 3 0 0 0 3-3H6a3 3 0 0 0 3 3m0 2h6m-4-8c-1.5-1.2 2.2-2.3.7-3.6M14 11c-1.3-1 1.8-2 .6-3"/></svg>',
  javascript:
    '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect width="18" height="18" x="3" y="3" fill="currentColor" rx="2"/><text x="12" y="16.5" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="7.5" font-weight="900" fill="var(--light)">JS</text></svg>',
  json: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7.5A2.5 2.5 0 0 0 5 7.5V9a2 2 0 0 1-2 2 2 2 0 0 1 2 2v1.5A2.5 2.5 0 0 0 7.5 17H9m6-12h1.5A2.5 2.5 0 0 1 19 7.5V9a2 2 0 0 0 2 2 2 2 0 0 0-2 2v1.5a2.5 2.5 0 0 1-2.5 2.5H15"/></svg>',
  markdown:
    '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" d="M3 6h18v12H3z"/><path fill="currentColor" d="M6 15V9h2l2 2.4L12 9h2v6h-2v-3l-2 2.3L8 12v3zm10-6h2v3h1.5L17 15.5 14.5 12H16z"/></svg>',
  ocaml: notebookOcamlLanguageLogoSvg,
  python:
    '<svg class="notebook-language-svg notebook-python-icon" viewBox="0 0 111 112" aria-hidden="true" focusable="false"><path fill="#3776ab" d="M54.918785.00091927421C50.335132.02221727 45.957846.41313697 42.106285 1.0946693 30.760069 3.0991731 28.700036 7.2947714 28.700035 15.032169v10.21875h26.8125v3.40625h-36.875c-7.792459 0-14.6157588 4.683717-16.7499998 13.59375-2.46181998 10.212966-2.57101508 16.586023 0 27.25 1.9059283 7.937852 6.4575432 13.593748 14.2499998 13.59375h9.21875v-12.25c0-8.849902 7.657144-16.656248 16.75-16.65625h26.78125c7.454951 0 13.406253-6.138164 13.40625-13.625v-25.53125c0-7.2663386-6.12998-12.7247771-13.40625-13.9374997C64.281548.32794397 59.502438-.02037903 54.918785.00091927421zM40.418785 8.2196694c2.769547 0 5.03125 2.2986456 5.03125 5.1249996-.000002 2.816336-2.261703 5.09375-5.03125 5.09375-2.779476-.000001-5.03125-2.277415-5.03125-5.09375-.000001-2.826353 2.251774-5.1249996 5.03125-5.1249996z"/><path fill="#ffd43b" d="M85.637535 28.657169v11.90625c0 9.230755-7.825895 16.999999-16.75 17h-26.78125c-7.335833 0-13.406249 6.278483-13.40625 13.625v25.531247c0 7.266344 6.318588 11.540324 13.40625 13.625004 8.487331 2.49561 16.626237 2.94663 26.78125 0 6.750155-1.95439 13.406253-5.88761 13.40625-13.625004V86.500919h-26.78125v-3.40625h40.187504c7.792461 0 10.696251-5.435408 13.406241-13.59375 2.79933-8.398886 2.68022-16.475776 0-27.25-1.92578-7.757441-5.60387-13.59375-13.406241-13.59375zm-15.0625 64.65625c2.779478.000003 5.03125 2.277417 5.03125 5.093747-.000002 2.826354-2.251775 5.125004-5.03125 5.125004-2.76955 0-5.03125-2.29865-5.03125-5.125004.000002-2.81633 2.261697-5.093747 5.03125-5.093747z"/></svg>',
  rust: notebookRustLanguageLogoSvg,
  text: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="2" d="M6 7h12M6 12h12M6 17h8"/></svg>',
  typescript:
    '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect width="18" height="18" x="3" y="3" fill="currentColor" rx="2"/><text x="12" y="16.5" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="7.5" font-weight="900" fill="var(--light)">TS</text></svg>',
  wasm: notebookWasmLanguageLogoSvg,
  zig: '<svg class="notebook-language-svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="currentColor" d="M5 5h14v3L9.5 19H5v-3L14.5 5H19v3L9.5 19H5z"/></svg>',
}
