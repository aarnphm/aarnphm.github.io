type NotebookRuntimeModule = { mountNotebookRuntime(root: HTMLElement, text: string): void }

let notebookRuntimeModule: Promise<NotebookRuntimeModule> | undefined

function notebookRuntimeScriptUrl(name: string) {
  return new URL(name, import.meta.url).href
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const htmlEntityReplacements = new Map([
  ['&quot;', '"'],
  ['&#34;', '"'],
  ['&#x22;', '"'],
  ['&apos;', "'"],
  ['&#39;', "'"],
  ['&#x27;', "'"],
  ['&lt;', '<'],
  ['&#60;', '<'],
  ['&#x3c;', '<'],
  ['&#x3C;', '<'],
  ['&gt;', '>'],
  ['&#62;', '>'],
  ['&#x3e;', '>'],
  ['&#x3E;', '>'],
  ['&amp;', '&'],
  ['&#38;', '&'],
  ['&#x26;', '&'],
])

function decodeHtmlEntity(entity: string) {
  return htmlEntityReplacements.get(entity) ?? entity
}

function runtimeIdFromJson(text: string) {
  try {
    const parsed = JSON.parse(text)
    return isRecord(parsed) && typeof parsed.id === 'string' ? parsed.id : undefined
  } catch {}

  const decoded = text.replace(
    /&(quot|apos|lt|gt|amp|#34|#x22|#39|#x27|#60|#x3[cC]|#62|#x3[eE]|#38|#x26);/g,
    decodeHtmlEntity,
  )
  if (decoded === text) return undefined

  try {
    const parsed = JSON.parse(decoded)
    return isRecord(parsed) && typeof parsed.id === 'string' ? parsed.id : undefined
  } catch {
    return undefined
  }
}

function runtimeDataById() {
  const data = new Map()
  document.querySelectorAll('script[data-notebook-runtime-data]').forEach(script => {
    const text = script.textContent
    if (!text) return
    const id = runtimeIdFromJson(text)
    if (id) data.set(id, text)
  })
  return data
}

async function mountNotebookRuntime() {
  const roots = document.querySelectorAll<HTMLElement>('[data-notebook-runtime]')
  const data = runtimeDataById()
  const targets = Array.from(roots)
    .map(root => {
      const id = root.dataset.notebookRuntime
      const text = id ? data.get(id) : undefined
      return text && root.dataset.runtimeMounted !== 'true' ? { root, text } : undefined
    })
    .filter((target): target is { root: HTMLElement; text: string } => target !== undefined)
  if (targets.length === 0) return
  notebookRuntimeModule ??= import(notebookRuntimeScriptUrl('notebook-runtime.client.js'))
  const runtime = await notebookRuntimeModule
  for (const target of targets) {
    runtime.mountNotebookRuntime(target.root, target.text)
  }
}

function scheduleNotebookRuntimeMount() {
  window.setTimeout(() => {
    void mountNotebookRuntime().catch(error => {
      console.error('failed to mount notebook runtime', error)
    })
  }, 0)
}

document.addEventListener('nav', scheduleNotebookRuntimeMount)
scheduleNotebookRuntimeMount()
