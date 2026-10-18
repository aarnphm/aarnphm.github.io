import {
  createBrowserJavaScriptKernel,
  notebookCellActions,
  notebookCellControls,
  notebookCellFrameOpen,
  notebookCellRuntimeOutput,
  notebookSourceEditor,
  notebookRuntimeLocalSourceKey,
} from '/fixture-exports.js'
import { mountNotebookRuntime } from '/static/scripts/notebook-runtime.client.js'

const checks = []
const pause = () => new Promise(resolve => setTimeout(resolve, 20))
function check(name, condition, observed) {
  checks.push({ name, passed: Boolean(condition), observed })
  if (!condition) throw new Error(name + ': ' + JSON.stringify(observed))
}
async function until(predicate) {
  for (let attempt = 0; attempt < 300; attempt++) {
    if (predicate()) return
    await pause()
  }
  throw new Error('browser fixture timed out')
}

function mount(id, source, sourcePath = '/fixtures/notebook.js') {
  const root = document.createElement('article')
  const cell = { id: id + '-cell', source, language: 'javascript', executionIndex: null }
  root.innerHTML = [
    notebookCellFrameOpen(cell.id, cell.language),
    notebookCellActions(cell),
    ...notebookCellControls(cell),
    '<figure><pre><code></code></pre></figure>',
    notebookSourceEditor(cell.id),
    notebookCellRuntimeOutput(cell.id),
    '</div>',
  ].join('\n')
  document.body.append(root)
  mountNotebookRuntime(
    root,
    JSON.stringify({ id, sourcePath, language: 'javascript', cells: [cell] }),
    {
      javascriptWorkerUrl:
        location.origin + '/static/scripts/notebook-runtime.javascript.worker.js',
    },
  )
  return root
}
async function runMounted(root, marker) {
  root.querySelector('[data-notebook-run-cell]').click()
  await until(() => root.querySelector('[data-notebook-output]').textContent.includes(marker))
  await until(() => root.querySelector('[data-notebook-stop]')?.disabled === true)
  return root.querySelector('[data-notebook-output]')
}

try {
  const initial = mount(
    'published-source',
    `element.text('published-source-ran');
    element.html('<img src=x onerror="document.body.dataset.injected=1"><iframe srcdoc="bad"></iframe><script>document.body.dataset.injected=1</script>');`,
  )
  await pause()
  check(
    'mount and URL query do not execute cell source',
    !initial.querySelector('[data-notebook-output]').textContent,
    initial.textContent,
  )
  const output = await runMounted(initial, 'published-source-ran')
  await pause()
  check(
    'published source executes after Run',
    output.textContent.includes('published-source-ran'),
    output.textContent,
  )
  check(
    'HTML output is sanitized at the production renderer',
    !output.querySelector('script, iframe, [onerror]') && !document.body.dataset.injected,
    output.innerHTML,
  )

  localStorage.setItem(
    notebookRuntimeLocalSourceKey('/fixtures/local.js', 'local-source-cell'),
    "element.text('local-edit-ran')",
  )
  const local = mount(
    'local-source',
    "element.text('wrong-published-source')",
    '/fixtures/local.js',
  )
  const localOutput = await runMounted(local, 'local-edit-ran')
  check(
    'saved local source has explicit-run precedence',
    !localOutput.textContent.includes('wrong-published-source'),
    localOutput.textContent,
  )

  const broker = mount(
    'asset-broker',
    `(async () => {
    const sibling = await (await fetch('sibling.txt')).text();
    let blocked = '';
    try { await fetch('/private/probe') } catch (error) { blocked = error.message }
    element.text('broker-result:' + sibling + ':' + blocked);
  })()`,
  )
  const brokerOutput = await runMounted(broker, 'broker-result:')
  check(
    'production fetch broker accepts siblings and rejects non-siblings',
    brokerOutput.textContent.includes('sibling-fixture') &&
      brokerOutput.textContent.includes('blocked non-sibling'),
    brokerOutput.textContent,
  )

  const kernel = createBrowserJavaScriptKernel({
    runtimeId: 'boundary-fixture',
    sourcePath: '/fixtures/kernel.js',
    workerUrl: location.origin + '/static/scripts/notebook-runtime.javascript.worker.js',
  })
  const collect = async source => {
    const events = []
    for await (const event of kernel.execute('cell-probe', source)) events.push(event)
    return events
  }
  const numeric = await collect('6 * 7')
  check(
    'kernel executes actual JavaScript in a Worker',
    numeric.some(event => event.type === 'output' && event.output.text === '42'),
    numeric,
  )
  const capabilities = await collect(`(async () => {
    const read = await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest(); request.open('GET', '/private/probe');
      request.onload = () => resolve(request.responseText); request.onerror = reject; request.send();
    });
    const database = await new Promise((resolve, reject) => {
      const request = indexedDB.open('code-scanning-fixture');
      request.onsuccess = () => { request.result.close(); resolve(true) }; request.onerror = reject;
    });
    return { origin: location.origin, documentType: typeof document, localStorageType: typeof localStorage,
      sameOriginResponse: read, indexedDBAvailable: database };
  })()`)
  const capabilityOutput = capabilities.find(
    event => event.type === 'output' && event.output.type === 'json',
  )
  const observed = JSON.parse(capabilityOutput.output.text)
  check(
    'worker has no DOM or localStorage interface',
    observed.documentType === 'undefined' && observed.localStorageType === 'undefined',
    observed,
  )
  check(
    'same-origin XHR retains ambient cookie authority',
    observed.sameOriginResponse.includes('fixture-cookie-present'),
    observed,
  )
  check(
    'same-origin IndexedDB is available, so worker execution is trusted',
    observed.indexedDBAvailable === true,
    observed,
  )

  const pending = collect('while (true) {}').then(
    events => ({ events, error: null }),
    error => ({ events: [], error: String(error) }),
  )
  await pause()
  kernel.interrupt()
  const interrupted = await pending
  check(
    'interrupt terminates a running worker',
    interrupted.events.some(event => event.type === 'interrupted') || interrupted.error,
    interrupted,
  )
  const restarted = await collect('40 + 2')
  check(
    'kernel restarts after interruption',
    restarted.some(event => event.type === 'output' && event.output.text === '42'),
    restarted,
  )
  await kernel.dispose()

  const worker = new Worker('/static/scripts/notebook-runtime.javascript.worker.js', {
    type: 'module',
  })
  const messages = []
  worker.addEventListener('message', event => messages.push(event.data))
  worker.postMessage({
    source: 'quartz-notebook-runtime',
    type: 'init',
    runtimeId: 'message-fixture',
  })
  await until(() => messages.some(message => message.type === 'ready'))
  worker.postMessage({
    source: 'wrong-source',
    type: 'run',
    runtimeId: 'message-fixture',
    cellId: 'wrong-source',
    code: 'globalThis.badMessageRan = true',
  })
  worker.postMessage({
    source: 'quartz-notebook-runtime',
    type: 'run',
    runtimeId: 'wrong-runtime',
    cellId: 'wrong-runtime',
    code: 'globalThis.badMessageRan = true',
  })
  worker.postMessage({
    source: 'quartz-notebook-runtime',
    type: 'run',
    runtimeId: 'message-fixture',
    cellId: 'valid',
    code: 'typeof globalThis.badMessageRan',
  })
  await until(() => messages.some(message => message.type === 'done' && message.cellId === 'valid'))
  check(
    'worker rejects mismatched source and runtime IDs',
    messages.some(message => message.cellId === 'valid' && message.output?.text === 'undefined') &&
      !messages.some(message => message.cellId?.startsWith('wrong-')),
    messages,
  )
  worker.terminate()
  await fetch('/result', { method: 'POST', body: JSON.stringify({ passed: true, checks }) })
  document.body.textContent = 'runtime boundary checks passed'
} catch (error) {
  await fetch('/result', {
    method: 'POST',
    body: JSON.stringify({ passed: false, checks, error: String(error), stack: error.stack }),
  })
  document.body.textContent = 'runtime boundary checks failed: ' + error
}
