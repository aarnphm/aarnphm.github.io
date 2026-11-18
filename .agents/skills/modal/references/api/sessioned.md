# sessioned

```python
sessioned()
```

Decorator that enables sticky sessions on a Server.

Every request must carry a session token obtained from a session start request; requests with the same token are
routed to the same container until the session is idle for `idle_timeout` seconds or explicitly terminated. A
container won't be scaled down for as long as it holds a live session.

Only valid with `@app.server()`.

**Usage**

Define a Server with the `@modal.sessioned()` decorator:

```python
app = modal.App('my-app')


@app.server(port=8000)
@modal.sessioned()
class MyServer:
  @modal.enter()
  def start(self):
    self.proc = subprocess.Popen(['python3', '-m', 'http.server', '8000'])

  @modal.exit()
  def stop(self):
    self.proc.terminate()
```

After deploying the App, start a session from another script:

```python notest
server = modal.Server.from_name('my-app', 'MyServer')
server_url = server.get_url()
session = server.sessions.start(idle_timeout=600)
headers = {'Modal-Authorization': f'Bearer {session.token}'}

requests.get(server_url, headers=headers).raise_for_status()

server.sessions.terminate(session.token)
```
