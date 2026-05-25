import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  readRustPlaygroundExecuteResult,
  rustPlaygroundVisibleStderr,
  type RustPlaygroundExecuteResult,
} from './playground-kernel'

describe('Rust Playground kernel helpers', () => {
  test('reads playground execute responses', () => {
    assert.deepStrictEqual(
      readRustPlaygroundExecuteResult({
        success: true,
        exitDetail: 'Exited with status 0',
        stdout: 'hi\n',
        stderr: '',
      }),
      { success: true, exitDetail: 'Exited with status 0', stdout: 'hi\n', stderr: '' },
    )
    assert.strictEqual(readRustPlaygroundExecuteResult({ success: true }), undefined)
  })

  test('hides cargo progress noise after successful runs', () => {
    const result: RustPlaygroundExecuteResult = {
      success: true,
      exitDetail: 'Exited with status 0',
      stdout: 'hi\n',
      stderr:
        '   Compiling playground v0.0.1 (/playground)\n    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.71s\n     Running `target/debug/playground`\n',
    }
    assert.strictEqual(rustPlaygroundVisibleStderr(result), '')
  })

  test('keeps compiler diagnostics when a rust cell fails', () => {
    const result: RustPlaygroundExecuteResult = {
      success: false,
      exitDetail: 'Exited with status 101',
      stdout: '',
      stderr: 'error[E0601]: `main` function not found in crate `playground`\n',
    }
    assert.match(rustPlaygroundVisibleStderr(result), /main` function not found/)
  })
})
