# Function

```python
class Function(typing.Generic, modal.object.Object)
```

Functions are the basic units of serverless execution on Modal.

Generally, you will not construct a `Function` directly. Instead, use the
`App.function()` decorator to register your Python functions with your App.

## hydrate

```python
hydrate(self, client=None)
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

_Added in v0.72.39_: This method replaces the deprecated `.resolve()` method.

## metrics

```python
metrics: FunctionMetricsManager
```

Access metric samples with `export()` and series definitions with `schema()`.

### metrics.schema

```python
schema(self, groups=None)
```

Return metric definitions keyed by group name without fetching samples.

Omit `groups` to list all groups. Each group contains its exact individual
series definitions.

### metrics.export

```python
export(self, *, since=None, until=None, bucket_size=None, groups=None,
    all_variants=False)
```

Fetch metric samples for a Function.

Defaults to all groups over the last hour. `since` defaults to one hour
before `until`; naive datetimes use local time. The maximum range is 31
days, with at most 500 points per series. If `bucket_size` is omitted,
Modal selects a supported width targeting approximately 100 buckets.
Bounds round down to UTC bucket boundaries, so data before `since` may
be included and the partial bucket at `until` is excluded.

Set `all_variants=True` to include the root Function and its variants.
Missing observations have value None. Use `schema()` to discover
group names and series definitions without fetching samples.

```python notest
metrics = function.metrics.export(
  bucket_size=timedelta(minutes=5), groups=['calls']
)
for point in metrics['successful_calls_count'].points:
  print(point.time, point.value)
```

## logs

```python
logs: FunctionLogsManager
```

Access logs for a `Function`.

Use [`fetch()`](#logsfetch)
to read logs from a UTC time range, [`tail()`](#logstail)
to read the most recent logs, and [`stream()`](#logsstream)
to follow new logs as they arrive.

**See Also**

- [`modal app logs`](https://modal.com/docs/cli/latest/app#modal-app-logs):
  CLI access to logs for an App.

### logs.fetch

```python
fetch(self, *, since, until=None, source=None, search_text="")
```

Fetch Function logs corresponding to the date range and filters.

**Parameters**

<Parameter name="since" type="datetime" description="Start date to fetch logs from. Must be in UTC or timezone-naive, which is interpreted as local time." />
<Parameter name="until" type="datetime | None" defaultValue="None" description="Defaults to current date if None. Must be in UTC or timezone-naive, which is interpreted as local time." />
<Parameter name="source" type="LogSource | None" defaultValue="None" description="Filter by source: &#x27;stdout&#x27;, &#x27;stderr&#x27;, or &#x27;system&#x27;." />
<Parameter name="search_text" type="str" defaultValue="&quot;&quot;" description="Filter by search text." />

**Yields**

`LogEntry` objects in chronological order.

**Usage**

```python notest
function = modal.Function.from_name('my-app', 'train')

for entry in function.logs.fetch(
  since=datetime.now() - timedelta(hours=4), source='stdout'
):
  print(entry.message, end='')
```

### logs.tail

```python
tail(self, entries=100, *, source=None)
```

Fetch the most recent Function logs.

**Parameters**

<Parameter name="entries" type="int" defaultValue="100" description="The number of log entries to return." />
<Parameter name="source" type="LogSource | None" defaultValue="None" description="Filter by source: &#x27;stdout&#x27;, &#x27;stderr&#x27;, or &#x27;system&#x27;." />

**Yields**

`LogEntry` objects in chronological order.

**Usage**

```python notest
function = modal.Function.from_name('my-app', 'train')

for entry in function.logs.tail(20):
  print(entry.message, end='')
```

### logs.stream

```python
stream(self, timeout=None)
```

Stream new Function logs until the timeout is reached.

**Parameters**

<Parameter name="timeout" type="float | None" defaultValue="None" description="Number of seconds to wait between log entries before terminating the stream. By default, this will block until it is interrupted." />

**Yields**

`LogEntry` objects as they arrive.

**Usage**

```python notest
function = modal.Function.from_name('my-app', 'train')

for entry in function.logs.stream(timeout=60):
  print(entry.message, end='')
```

## update_autoscaler

```python
update_autoscaler(self, *, min_containers=None, max_containers=None,
    buffer_containers=None, scaledown_window=None)
```

Override the current autoscaler behavior for this Function.

Unspecified parameters will retain their current value, i.e. either the static value
from the function decorator, or an override value from a previous call to this method.

Subsequent deployments of the App containing this Function will reset the autoscaler back to
its static configuration.

**Parameters**

<Parameter name="min_containers" type="int | None" defaultValue="None" description="Minimum number of containers to keep running." />
<Parameter name="max_containers" type="int | None" defaultValue="None" description="Maximum concurrent containers." />
<Parameter name="buffer_containers" type="int | None" defaultValue="None" description="Extra containers to keep warm beyond current demand." />
<Parameter name="scaledown_window" type="int | None" defaultValue="None" description="Maximum duration (in seconds) idle containers wait before scaling down." />

**Returns**

A `FunctionAutoscalerSettings` dataclass which contains the current autoscaler settings
of this Function after the call.

**Usage**

```python notest
f = modal.Function.from_name('my-app', 'function')

# Always have at least 2 containers running, with an extra buffer when the Function is active
f.update_autoscaler(min_containers=2, buffer_containers=1)

# Limit this Function to avoid spinning up more than 5 containers
f.update_autoscaler(max_containers=5)

# Extend the scaledown window to increase the amount of time that idle containers stay alive
f.update_autoscaler(scaledown_window=300)
```

## from_id

```python
from_id(cls, function_id, *, client=None)
```

Reference a Function from a deployed or running App by its ID.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time it is actually used.

**Parameters**

<Parameter name="function_id" type="str" description="ID of the function." />
<Parameter name="client" type="_Client | None" defaultValue="None" description="Modal client to use; defaults to `Client.from_env()` when omitted." />

**Returns**

A lazy `Function` handle.

**Usage**

```python
f = modal.Function.from_id('fu-123')
```

## from_name

```python
from_name(cls, app_name, name, *, version=None, environment_name=None,
    client=None)
```

Reference a Function from a deployed App by its name.

This is a lazy method that defers hydrating the local
object with metadata from Modal servers until the first
time it is actually used.

**Parameters**

<Parameter name="app_name" type="str" description="Name of the deployed App." />
<Parameter name="name" type="str" description="Name of the Function within that App. For class methods, use `Cls.from_name` instead." />
<Parameter name="environment_name" type="str | None" defaultValue="None" description="Environment to look up the App in; defaults to the active environment." />
<Parameter name="client" type="_Client | None" defaultValue="None" description="Modal client to use; defaults to `Client.from_env()` when omitted." />

**Returns**

A lazy `Function` handle.

**Usage**

```python
f = modal.Function.from_name('other-app', 'function')
```

The `version` parameter allows you to invoke a version-pinned function:

```python
f_v3 = modal.Function.from_name('other-app', 'function', version=3)
```

## info

```python
info(self, *, refresh=False)
```

Get an overview of a Function's resource requests, associated mounts, etc.

This method performs a network request to populate this information if the Function handle is
a remote lookup whose information has not yet been fetched (e.g. from `Function.from_name(...)`),
or if `refresh=True`.

**Parameters**

<Parameter name="refresh" type="bool" defaultValue="False" description="Always perform a network request. Pass `refresh=True` to ensure that this method returns the most up to date information." />

**Returns**

This returns a [`modal.types.FunctionInfo`](https://modal.com/docs/sdk/py/latest/types#FunctionInfo)
dataclass.

## get_web_url

```python
get_web_url(self)
```

URL for addressing a Web Function via HTTP.

**Returns**

The HTTPS URL for the web endpoint, or `None` if this Function is not a web endpoint.

## with_options

```python
with_options(self, *, cpu=None, memory=None, gpu=None, env=None, secrets=None,
    volumes={}, retries=None, max_containers=None, buffer_containers=None,
    scaledown_window=None, timeout=None, region=None, cloud=None,
    routing_region=None)
```

Dynamically override the static Function configuration with invocation-specific values.

This method returns a new Function instance with the dynamic configuration. Invocations of
the new Function will run in a distinct container pool and autoscale independently from the
base Function (and from other dynamic configurations).

Note that options cannot be "unset" with this method (i.e., if a GPU is configured in the
`@app.cls()` decorator, passing `gpu=None` here will not create a CPU-only instance).
Additionally, container arguments like `volumes` and `secrets` will _replace_ the base
configuration or any previous use of this method rather than extending it.

**Usage:**

You can use this method after looking up a deployed Function:

```python notest
fn = modal.Function.from_name('my_app', 'fn').with_options(gpu='H100')
fn.remote()  # will run on a H100 GPU
```

Or by referencing another Function defined in the same App:

```python notest
@app.function()
def fn(): ...


# From a local entrypoint or another Function
fn.with_options(gpu='H100').remote()  # Uses an H100 GPU
fn.remote()  # Uses the static configuration with no GPU
```

## with_concurrency

```python
with_concurrency(self, *, max_inputs, target_inputs=None)
```

Override the static Function configuration with invocation-specific input concurrency.

Returns a new Function instance that is dynamically configured to behave like a Function using
the `@modal.concurrent` decorator. This instance will autoscale independently from the base Function.

## with_batching

```python
with_batching(self, *, max_batch_size, wait_ms)
```

Override the static Function configuration with invocation-specific dynamic batching.

Returns a new Function instance that is dynamically configured to behave like a Function using
the `@modal.batched` decorator. This instance will autoscale independently from the base Function.

## remote

```python
remote(self, *args, **kwargs)
```

Calls the function remotely, executing it with the given arguments and returning the execution's result.

**Parameters**

<Parameter name="*args" type="P.args" description="Positional arguments forwarded to the deployed function." />
<Parameter name="**kwargs" type="P.kwargs" description="Keyword arguments forwarded to the deployed function." />

**Returns**

The value returned by the remote function.

## remote_gen

```python
remote_gen(self, *args, **kwargs)
```

Calls the generator remotely, executing it with the given arguments.

**Parameters**

<Parameter name="*args" type="" description="Positional arguments forwarded to the deployed generator function." />
<Parameter name="**kwargs" type="" description="Keyword arguments forwarded to the deployed generator function." />

**Yields**

Values produced by the remote generator.

## local

```python
local(self, *args, **kwargs)
```

Calls the function locally, executing it with the given arguments and returning the execution's result.

The function will execute in the same environment as the caller, just like calling the underlying function
directly in Python. In particular, only secrets available in the caller environment will be available
through environment variables.

**Parameters**

<Parameter name="*args" type="P.args" description="Positional arguments passed to the underlying Python callable." />
<Parameter name="**kwargs" type="P.kwargs" description="Keyword arguments passed to the underlying Python callable." />

**Returns**

The return value of the local call (or a coroutine for async functions).

## spawn

```python
spawn(self, *args, **kwargs)
```

Calls the function with the given arguments, without waiting for the results.

Conceptually similar to `multiprocessing.pool.apply_async`, or a Future/Promise in other contexts.

**Parameters**

<Parameter name="*args" type="P.args" description="Positional arguments forwarded to the remote function." />
<Parameter name="**kwargs" type="P.kwargs" description="Keyword arguments forwarded to the remote function." />

**Returns**

A [`modal.FunctionCall`](https://modal.com/docs/sdk/py/latest/FunctionCall) object
that can later be polled or waited for using
[`.get(timeout=...)`](https://modal.com/docs/sdk/py/latest/FunctionCall#get).

## get_current_stats

```python
get_current_stats(self)
```

Return a snapshot of the Function's current input and container state.

**Returns**

A `FunctionCurrentStats` object containing live input and container counts.

## stats

```python
stats(self, *, since=None, until=None, container=None, all_variants=False)
```

Return statistics for a modal Function.

The default time range is the most recent hour. The maximum time range is 7 days.

**Parameters**

<Parameter name="since" type="datetime | None" defaultValue="None" description="The beginning of the time range, inclusive. If omitted, this defaults to an hour before `until`. Values without a timezone are interpeted as local time." />
<Parameter name="until" type="datetime | None" defaultValue="None" description="The end of the time range, exclusive. If omitted, this defaults to current time. Values without a timezone are interpeted as local time." />
<Parameter name="container" type="str | None" defaultValue="None" description="If passed in, the stats are computed for only this container. Default None." />
<Parameter name="all_variants" type="bool" defaultValue="False" description="If True, aggregate the base Function and its variants." />

**Returns**

A `FunctionStats` object

## autoscaler

```python
autoscaler: FunctionAutoscalerManager
```

Namespace for Function autoscaler APIs.

### autoscaler.settings

```python
settings(self)
```

View the current Autoscaler settings for this Function.

**Returns**

A `FunctionAutoscalerSettings` dataclass.

**Usage**

```python notest
fn = modal.Function.from_name('my-app', 'my_function')
settings = fn.autoscaler.settings()
```

### autoscaler.update

```python
update(self, *, min_containers=UNSET_PARAMETER, max_containers=UNSET_PARAMETER,
    buffer_containers=UNSET_PARAMETER, scaledown_window=UNSET_PARAMETER)
```

Override the current Autoscaler settings for this Function.

All parameters to this function are optional; not passing an argument for a given parameter
will not update its value.

Note that re-deploying the App containing this Function will wipe any overrides and revert
its Autoscaler settings back to those defined on the Function at creation time.

**Parameters**

<Parameter name="min_containers" type="int | None" defaultValue="UNSET_PARAMETER" description="Minimum number of containers to keep running. Must be a nonnegative integer, or `None`. Passing `None` is equivalent to setting the minimum to 0." />
<Parameter name="max_containers" type="int | None" defaultValue="UNSET_PARAMETER" description="Maximum concurrent containers. Must be a positive integer, or `None`. Passing `None` is equivalent to disabling the upper bound on container concurrency at the Function level (note that Environment and Workspace level container upper limits still apply.)" />
<Parameter name="buffer_containers" type="int | None" defaultValue="UNSET_PARAMETER" description="Extra containers to keep warm beyond current demand. Must be a nonnegative integer, or `None`. Passing `None` is equivalent to setting the buffer to 0." />
<Parameter name="scaledown_window" type="int" defaultValue="UNSET_PARAMETER" description="Maximum duration (in seconds) idle containers wait before scaling down. Must be a positive integer between 2 and 3600, inclusive." />

**Returns**

A `FunctionAutoscalerSettings` dataclass.

**Usage**

Update `my_function` so that 2 containers are always active.

```python notest
fn = modal.Function.from_name('my-app', 'my_function')
fn.autoscaler.update(min_containers=2)
```

Update `my_function` so that there is no Function-level limit on `max_containers`:

```python notest
fn = modal.Function.from_name('my-app', 'my_function')
fn.autoscaler.update(max_containers=None)
```

## map

```python
map(
  self,
  *input_iterators,
  kwargs={},
  order_outputs=True,
  return_exceptions=False,
  wrap_returned_exceptions=None,
)
```

Parallel map over a set of inputs.

Pass one iterable per positional argument of the underlying function. Results are yielded as an
iterable (sync) or async iterator (`map.aio`).

If applied to an `@app.function`, `map()` returns one result per input and output order matches
input order by default. Set `order_outputs=False` to emit results in completion order.

`return_exceptions` can aggregate failures into the result stream instead of raising.

**Parameters**

<Parameter name="*input_iterators" type="typing.Iterable[Any]" description="One iterator per mapped positional parameter on the function." />
<Parameter name="kwargs" type="" defaultValue="&#123;&#125;" description="Extra keyword arguments forwarded to every invocation." />
<Parameter name="order_outputs" type="bool" defaultValue="True" description="If True, preserve input order in outputs; if False, use completion order." />
<Parameter name="return_exceptions" type="bool" defaultValue="False" description="If True, failed inputs appear as exceptions in the result stream instead of raising." />
<Parameter name="wrap_returned_exceptions" type="bool | None" defaultValue="None" description="Deprecated; no longer has any effect." />

**Usage**

```python
@app.function()
def my_func(a):
  return a**2


@app.local_entrypoint()
def main():
  assert list(my_func.map([1, 2, 3, 4])) == [1, 4, 9, 16]
```

```python
@app.function()
def my_func(a):
  if a == 2:
    raise Exception('ohno')
  return a**2


@app.local_entrypoint()
def main():
  print(list(my_func.map(range(3), return_exceptions=True)))
```

## starmap

```python
starmap(self, input_iterator, *, kwargs={}, order_outputs=True,
    return_exceptions=False, wrap_returned_exceptions=None)
```

Like `map`, but each input item is unpacked into multiple positional arguments.

Every element of `input_iterator` should be a sequence (for example a tuple) with length equal to the
arity of the function.

**Parameters**

<Parameter name="input_iterator" type="typing.Iterable[typing.Sequence[Any]]" description="Iterable of argument tuples to unpack into each call." />
<Parameter name="kwargs" type="" defaultValue="&#123;&#125;" description="Extra keyword arguments forwarded to every invocation." />
<Parameter name="order_outputs" type="bool" defaultValue="True" description="If True, preserve input order in outputs; if False, use completion order." />
<Parameter name="return_exceptions" type="bool" defaultValue="False" description="If True, failed inputs appear as exceptions in the result stream instead of raising." />
<Parameter name="wrap_returned_exceptions" type="bool | None" defaultValue="None" description="Deprecated; no longer has any effect." />

**Usage**

```python
@app.function()
def my_func(a, b):
  return a + b


@app.local_entrypoint()
def main():
  assert list(my_func.starmap([(1, 2), (3, 4)])) == [3, 7]
```

## for_each

```python
for_each(self, *input_iterators, kwargs={}, ignore_exceptions=False)
```

Execute the function for all inputs and wait for completion, discarding return values.

Like `.map()` but you do not need to iterate the result to drive work—Modal processes every input.

**Parameters**

<Parameter name="*input_iterators" type="" description="One iterator per mapped positional parameter on the function." />
<Parameter name="kwargs" type="" defaultValue="&#123;&#125;" description="Extra keyword arguments forwarded to every invocation." />
<Parameter name="ignore_exceptions" type="bool" defaultValue="False" description="If True, failures are swallowed instead of propagating." />

## spawn_map

```python
spawn_map(self, *input_iterators, kwargs={})
```

Spawn parallel execution over a set of inputs, exiting as soon as the inputs are created (without waiting
for the map to complete).

Takes one iterator argument per argument in the function being mapped over.

Programmatic retrieval of results will be supported in a future update.

**Parameters**

<Parameter name="*input_iterators" type="" description="One iterator per mapped positional parameter on the function." />
<Parameter name="kwargs" type="" defaultValue="&#123;&#125;" description="Extra keyword arguments forwarded to every invocation." />

**Usage**

```python
@app.function()
def my_func(a):
  return a**2


@app.local_entrypoint()
def main():
  my_func.spawn_map([1, 2, 3, 4])
```
