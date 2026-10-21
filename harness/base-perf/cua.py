"""Client for the Helium CUA broker (~/.local/share/helium-cua): newline JSON over its Unix socket.
Each `execute` runs JavaScript in the session's persistent Node REPL, where `cua` controls the
real Helium profile (TabFS discovery, exclusive tab leases, raw CDP per tab). The bench keeps one
tab in a named task group and talks to the page through it. Nothing here runs node itself."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid

STATE_DIR = os.environ.get('CUA_STATE_DIR') or os.path.expanduser(
  '~/.local/state/helium-cua'
)
SOCKET = os.path.join(STATE_DIR, 'broker.sock')
RUNTIME = os.path.expanduser('~/.local/share/helium-cua')


class CuaError(RuntimeError):
  pass


def _rpc(method: str, params: dict, timeout: float) -> dict:
  s = socket.socket(socket.AF_UNIX)
  s.settimeout(timeout)
  try:
    s.connect(SOCKET)
    s.sendall(
      (
        json.dumps({'id': 1, 'method': method, 'params': params}) + '\n'
      ).encode()
    )
    buf = b''
    while not buf.endswith(b'\n'):
      chunk = s.recv(1 << 20)
      if not chunk:
        raise CuaError('broker closed the connection before a response')
      buf += chunk
  finally:
    s.close()
  message = json.loads(buf)
  if message.get('error'):
    raise CuaError(message['error'])
  return message.get('result')


def ensure_broker() -> None:
  """The broker normally runs already (the extension keeps it up). Start it only when the socket
  is dead, the way the runtime's ensureBroker does."""
  try:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(2)
    s.connect(SOCKET)
    s.close()
    return
  except OSError:
    pass
  os.makedirs(STATE_DIR, mode=0o700, exist_ok=True)
  subprocess.Popen(
    ['node', os.path.join(RUNTIME, 'broker.mjs')],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
  )
  for _ in range(60):
    time.sleep(0.1)
    try:
      s = socket.socket(socket.AF_UNIX)
      s.settimeout(2)
      s.connect(SOCKET)
      s.close()
      return
    except OSError:
      continue
  raise CuaError('browser broker did not start')


class Session:
  """One REPL session. `js` returns the texts the code wrote with repl.write (and the images it
  emitted), raising on a REPL error. `value` runs code whose last repl.write is JSON."""

  def __init__(self, name: str):
    if not name or len(name) > 128:
      raise ValueError('session name')
    self.name = name
    ensure_broker()

  def js(self, code: str, timeout_ms: int = 30000) -> list[dict]:
    result = _rpc(
      'execute',
      {
        'session': self.name,
        'requestId': str(uuid.uuid4()),
        'code': code,
        'timeoutMs': timeout_ms,
      },
      timeout=timeout_ms / 1000 + 10,
    )
    content = result.get('content', [])
    if result.get('isError'):
      text = '\n'.join(
        c.get('text', '') for c in content if c.get('type') == 'text'
      )
      raise CuaError(text or 'REPL error')
    return content

  def value(self, code: str, timeout_ms: int = 30000):
    texts = [
      c['text'] for c in self.js(code, timeout_ms) if c.get('type') == 'text'
    ]
    if not texts:
      return None
    return json.loads(texts[-1])

  def image(self, code: str, timeout_ms: int = 30000) -> bytes:
    import base64

    for c in self.js(code, timeout_ms):
      if c.get('type') == 'image':
        return base64.b64decode(c['data'])
    raise CuaError('no image emitted')

  def reset(self, release: bool = False) -> None:
    _rpc('reset', {'session': self.name, 'release': release}, timeout=30)


class Tab:
  """A task tab owned by the session: `var tab` in the REPL. Page code goes through raw CDP
  (`tab.command`) so the bench reads Performance.getMetrics like a DevTools client would. Open it
  on a light page such as robots.txt (tabs.new takes HTTP(S) URLs only and waits for the load):
  the home page took 30 s and more on a loaded worktree server, which spent the whole creation
  timeout before the first goto."""

  def __init__(
    self,
    session: Session,
    url: str,
    group: str = 'base-perf',
    width: int = 1440,
    height: int = 900,
  ):
    self.s = session
    self.s.js(
      "var browser = await cua.getBrowser({id: 'helium'});"
      f'var tab = await browser.tabs.new({{url: {json.dumps(url)}, sessionName: {json.dumps(group)}}});'
      "await tab.command('Page.enable'); await tab.command('Runtime.enable'); await tab.command('Performance.enable');"
      f"await tab.command('Emulation.setDeviceMetricsOverride', {{width: {width}, height: {height}, deviceScaleFactor: 1, mobile: false}});"
      'repl.write(JSON.stringify({id: tab.id}))',
      60000,
    )

  def evaluate(self, expression: str, timeout_ms: int = 30000):
    code = (
      f"var __r = await tab.command('Runtime.evaluate', {{expression: {json.dumps(expression)}, returnByValue: true, awaitPromise: true}});"
      'if (__r.exceptionDetails) throw new Error(__r.exceptionDetails.exception?.description || __r.exceptionDetails.text);'
      'repl.write(JSON.stringify(__r.result.value === undefined ? null : __r.result.value))'
    )
    return self.s.value(code, timeout_ms)

  def command(
    self, method: str, params: dict | None = None, timeout_ms: int = 30000
  ):
    code = f'repl.write(JSON.stringify(await tab.command({json.dumps(method)}, {json.dumps(params or {})})))'
    return self.s.value(code, timeout_ms)

  def goto(self, url: str) -> None:
    self.s.js(f'await tab.goto({json.dumps(url)})', 30000)

  def bring_to_front(self) -> None:
    self.s.js("await tab.command('Page.bringToFront')", 10000)

  def wait_ready(
    self, ready: str, limit_ms: int, poll_ms: int = 100
  ) -> float | None:
    """Polls the ready expression; returns the wait in ms, or None at the limit."""
    t0 = time.monotonic()
    probe = (
      f'(() => {{ try {{ return !!({ready}) }} catch {{ return false }} }})()'
    )
    while (time.monotonic() - t0) * 1000 < limit_ms:
      try:
        # A loaded main thread answers late; a REPL timeout here would reset the session.
        if self.evaluate(probe, 30000):
          return (time.monotonic() - t0) * 1000
      except CuaError as e:
        if 'EXECUTION_TIMEOUT' in str(e) or 'tab is not defined' in str(e):
          raise
      time.sleep(poll_ms / 1000)
    return None

  def metrics(self) -> dict:
    keys = {
      'Nodes',
      'Documents',
      'JSEventListeners',
      'LayoutObjects',
      'JSHeapUsedSize',
      'JSHeapTotalSize',
      'LayoutCount',
      'RecalcStyleCount',
      'LayoutDuration',
      'RecalcStyleDuration',
      'ScriptDuration',
      'TaskDuration',
    }
    return {
      m['name']: m['value']
      for m in self.command('Performance.getMetrics')['metrics']
      if m['name'] in keys
    }

  def heap_usage(self) -> dict:
    """Runtime.getHeapUsage: usedSize/totalSize are V8, embedderHeapUsedSize is Blink's Oilpan
    heap (DOM, style, layout objects), which Performance.getMetrics does not report."""
    return self.command('Runtime.getHeapUsage')

  def screenshot(self) -> bytes:
    return self.s.image('await repl.emitImage(await tab.screenshot())', 30000)

  def finish(self) -> None:
    try:
      self.s.js('await cua.finish()', 40000)
    finally:
      self.s.reset(release=True)
