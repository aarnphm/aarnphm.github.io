import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const bridgePath = fileURLToPath(new URL('pyodide-bridge.py', import.meta.url))

function pythonCommand(): string {
  const candidates = [process.env.PYTHON, 'python3', 'python'].filter(
    (value): value is string => typeof value === 'string' && value.length > 0,
  )
  for (const command of candidates) {
    const result = spawnSync(command, ['--version'], { encoding: 'utf8' })
    if (result.status === 0) return command
  }
  throw new Error('python is required for pyodide bridge tests')
}

function runBridgeProbe(code: string): string {
  const script = `
import os
import runpy
import sys
import types

js = types.ModuleType('js')
js.quartz_notebook_ml_register_callbacks = lambda *args: None
js.quartz_notebook_display = lambda payload: None
js.quartz_notebook_python_error = lambda payload: None
sys.modules['js'] = js
os.makedirs = lambda *args, **kwargs: None
os.chdir = lambda *args, **kwargs: None
runpy.run_path(${JSON.stringify(bridgePath)}, run_name='__quartz_bridge_test__')
${code}
`
  const result = spawnSync(pythonCommand(), ['-c', script], { encoding: 'utf8' })
  if (result.error) throw result.error
  assert.equal(result.status, 0, result.stderr)
  return result.stdout.trim()
}

test('installs the IPython package branches used by notebooks', () => {
  const output = runBridgeProbe(`
import json
import os
import sys
from IPython import extract_module_locals as top_level_extract_module_locals
from IPython.core.display import HTML, Javascript
from IPython.display import display
from IPython.utils.frame import extract_module_locals

def probe():
  marker = 42
  module, local_ns = extract_module_locals()
  top_level_module, top_level_locals = top_level_extract_module_locals()
  return {
    'module': module.__name__,
    'marker': local_ns['marker'],
    'top_level_module': top_level_module.__name__,
    'top_level_marker': top_level_locals['marker'],
    'javascript': Javascript('1 + 1')._repr_mimebundle_()[0]['application/javascript'],
    'html': HTML('<b>x</b>')._repr_mimebundle_()[0]['text/html'],
    'ipython_package': hasattr(sys.modules['IPython'], '__path__'),
    'core_package': hasattr(sys.modules['IPython.core'], '__path__'),
    'utils_package': hasattr(sys.modules['IPython.utils'], '__path__'),
  }

try:
  working_dir = os.path.dirname(extract_module_locals()[1]['__session__'])
  working_dir_source = 'session'
except Exception:
  working_dir = os.getcwd()
  working_dir_source = 'cwd'

display('ok')
result = probe()
result['pca_working_dir_source'] = working_dir_source
result['pca_working_dir_is_string'] = isinstance(working_dir, str)
print(json.dumps(result, sort_keys=True))
`)

  const parsed: unknown = JSON.parse(output)
  assert.deepEqual(parsed, {
    core_package: true,
    html: '<b>x</b>',
    ipython_package: true,
    javascript: '1 + 1',
    marker: 42,
    module: '__main__',
    pca_working_dir_is_string: true,
    pca_working_dir_source: 'cwd',
    top_level_marker: 42,
    top_level_module: '__main__',
    utils_package: true,
  })
})
