export const NOTEBOOK_VIM_MODE_STORAGE_KEY = 'quartz:notebook-vim-mode'

export async function loadVimExtension(): Promise<unknown> {
  const mod = await import('@replit/codemirror-vim')
  return mod.vim()
}

export function readVimModeSetting(storage: Storage | undefined = pickStorage()): boolean {
  if (!storage) return false
  return storage.getItem(NOTEBOOK_VIM_MODE_STORAGE_KEY) === 'true'
}

export function writeVimModeSetting(
  value: boolean,
  storage: Storage | undefined = pickStorage(),
): void {
  if (!storage) return
  if (value) {
    storage.setItem(NOTEBOOK_VIM_MODE_STORAGE_KEY, 'true')
  } else {
    storage.removeItem(NOTEBOOK_VIM_MODE_STORAGE_KEY)
  }
}

function pickStorage(): Storage | undefined {
  if (typeof globalThis === 'undefined') return undefined
  return (globalThis as { localStorage?: Storage }).localStorage
}
