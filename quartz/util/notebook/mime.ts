export const NOTEBOOK_OUTPUT_MIME_PRIORITY = [
  'text/html',
  'text/markdown',
  'text/latex',
  'image/svg+xml',
  'image/png',
  'image/jpeg',
  'image/gif',
  'application/json',
  'text/plain',
] as const

export type NotebookOutputMimeType = (typeof NOTEBOOK_OUTPUT_MIME_PRIORITY)[number]

export const NOTEBOOK_ATTACHMENT_MIME_TYPES = [
  'image/png',
  'image/jpeg',
  'image/gif',
  'image/webp',
  'image/svg+xml',
] as const

export type NotebookAttachmentMimeType = (typeof NOTEBOOK_ATTACHMENT_MIME_TYPES)[number]

export const NOTEBOOK_IMAGE_BINARY_MIME_TYPES = ['image/png', 'image/jpeg', 'image/gif'] as const

export type NotebookImageBinaryMimeType = (typeof NOTEBOOK_IMAGE_BINARY_MIME_TYPES)[number]
