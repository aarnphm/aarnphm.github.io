import assert from 'node:assert'
import test, { describe } from 'node:test'
import {
  haskellPlaygroundFailureText,
  readHaskellPlaygroundExecuteResult,
  type HaskellPlaygroundExecuteResult,
} from './playground-kernel'

describe('Haskell Playground kernel helpers', () => {
  test('reads playground execute responses', () => {
    assert.deepStrictEqual(
      readHaskellPlaygroundExecuteResult({
        ec: 0,
        ghcout: '',
        sout: 'hi\n',
        serr: '',
        timesecs: 0.1,
      }),
      { ec: 0, ghcout: '', sout: 'hi\n', serr: '', timesecs: 0.1 },
    )
    assert.strictEqual(readHaskellPlaygroundExecuteResult({ ec: 0 }), undefined)
  })

  test('combines compiler and process diagnostics for failures', () => {
    const result: HaskellPlaygroundExecuteResult = {
      ec: 1,
      ghcout: 'Main.hs:1:1: error: nope\n',
      sout: '',
      serr: '',
    }
    assert.match(haskellPlaygroundFailureText(result), /Main\.hs/)
    assert.match(haskellPlaygroundFailureText(result), /code 1/)
  })
})
