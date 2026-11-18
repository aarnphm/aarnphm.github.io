# Workspace

```python
class Workspace(modal.object.Object)
```

## hydrate

```python
hydrate(self, client=None)
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

_Added in v0.72.39_: This method replaces the deprecated `.resolve()` method.

## name

```python
name(self)
```

## members

```python
members: WorkspaceMembersManager
```

Namespace with methods for managing the membership of a Workspace.

### members.list

```python
list(self)
```

Return the members of the Workspace.

**Examples:**

```python notest
members = modal.Workspace.from_context().members.list()
print([m.name for m in members])
```

## from_context

```python
from_context(*, client=None)
```

Look up the Workspace associated with the current context.

This returns the Workspace that the active Modal credentials authenticate against
(i.e., your active profile or the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment
variables). If called inside a Modal container, it returns the Workspace that the
container is running in.

## billing

```python
billing: WorkspaceBillingManager
```

Namespace for Workspace billing APIs.

### billing.rates

```python
rates(self)
```

Return current pricing rates for the given workspace.

**Returns**

A single mapping containing cost values. All values are reported as `decimal.Decimal`s.

### billing.report

```python
report(self, *, start, end=None, resolution="d", tag_names=None)
```

Return a cost report for all Workspace usage, broken down by object and time.

**Parameters**

<Parameter name="start" type="datetime" description="Start of the report, inclusive and rounded to the beginning of the interval. Must be in UTC or timezone-naive (interpreted as UTC)." />
<Parameter name="end" type="datetime | None" defaultValue="None" description="End of the report, exclusive. Must be in UTC or timezone-naive. Partial final intervals will be excluded from the report." />
<Parameter name="resolution" type="str" defaultValue="&quot;d&quot;" description="Resolution, e.g. &quot;d&quot; for daily or &quot;h&quot; for hourly." />
<Parameter name="tag_names" type="list[str] | None" defaultValue="None" description="List of tag names; each row will include the tag name and value in use for that object during the relevant time interval. Pass `[&quot;*&quot;]` to include all tags in the report." />

**Returns**

A list of `BillingReportItem` dataclasses. Each item reports the cost attributed to
a specific Modal object during a given time interval. Cost is further broken down by
the resource type that generated it (e.g. CPU, Memory, specific GPU usage). Note that
the specific resource types included in the breakdown are subject to change as Modal's
billing model evolves.

**See Also**

- [`modal billing report`](https://modal.com/docs/cli/latest/billing#modal-billing-report):
  A workspace report CLI that has convenience features around relative time range queries
  and JSON/CSV output.
- [`Environment.billing.report()`](https://modal.com/docs/sdk/py/latest/Environment#billingreport):
  An analogous report API that is scoped to a specific Environment.

### billing.summary

```python
summary(self, cycle=None)
```

Return a summary of workspace cost over a single billing cycle determined by `cycle`

**Parameters**

<Parameter name="cycle" type="str | datetime | None" defaultValue="None" description="Start of the summary, inclusive. Must be the first of a month, and must be in UTC or timezone-naive (interpreted as UTC). If provided as a string, it must either be formatted as an ISO 8601 month (YYYY-MM), or must be one of the convenience spellings &quot;this month&quot; or &quot;last month&quot;. If not provided, `cycle` defaults to the first of the current month (in which case a summary is generated for the current billing cycle)." />

**Returns**

A single `WorkspaceBillingSummary` dataclass containing the following fields:

- `metered_cost` representing cost before any adjustments,
- `billed_cost` representing the cost actually invoiced, including all adjustments,
- `adjustments` containing a breakdown of the adjustments that make up the difference
  between `metered_cost` and `billed_cost`. This can include discounts for free volume
  storage, adjustments due to plan credits, etc. The exact keys of this are subject to
  change as Modal's billing model evolves.
- `metered_cost_breakdown` containing a breakdown of that cost by the Modal resources
  that generated it. The exact keys of this are subject to change as Modal's billing
  model evolves.

All values are reported as `decimal.Decimal`s.

**See Also**

- [`modal billing summary`](https://modal.com/docs/cli/latest/billing#modal-billing-summary):
  A workspace summary CLI that has convenience features around relative time range queries.
- [`Environment.billing.summary()`](https://modal.com/docs/sdk/py/latest/Environment#billingsummary):
  An analogous summary API that is scoped to a specific Environment.

## proxy_tokens

```python
proxy_tokens: WorkspaceProxyTokenManager
```

Namespace with methods for managing the proxy tokens in a Workspace.

See [the guide](https://modal.com/docs/guide/webhook-proxy-auth) for more information on proxy tokens.

### proxy_tokens.create

```python
create(self, name='')
```

Create a new proxy token for the Workspace.

**Parameters**

<Parameter name="name" type="str" defaultValue="&quot;&quot;" description="An optional name to help identify the token." />

**Usage**

```python notest
token = modal.Workspace.from_context().proxy_tokens.create(
  name='production-webhooks'
)
print(token.token_id, token.token_secret)
```

### proxy_tokens.update

```python
update(self, proxy_token_id, *, name)
```

Update a proxy token in the Workspace.

An empty name removes the token's name.

**Parameters**

<Parameter name="proxy_token_id" type="str" description="The token ID (`wk-...`) to update." />
<Parameter name="name" type="str" description="The updated name for the token." />

**Usage**

```python notest
ws = modal.Workspace.from_context()
ws.proxy_tokens.update(token_id, name='production-webhooks')
```

### proxy_tokens.list

```python
list(self, environment_name=None)
```

List proxy tokens in the Workspace.

**Parameters**

<Parameter name="environment_name" type="Optional[str]" defaultValue="None" description="When provided, list only the tokens associated with this environment." />

**Usage**

```python notest
ws = modal.Workspace.from_context()

# List all proxy tokens in the Workspace
tokens = ws.proxy_tokens.list()
print([t.token_id for t in tokens])

# List only the proxy tokens associated with a specific Environment
env_tokens = ws.proxy_tokens.list(environment_name='prod')
```

### proxy_tokens.allow

```python
allow(self, proxy_token_id, environment_name)
```

Allow a proxy token to authenticate requests to a given Environment.

**Parameters**

<Parameter name="proxy_token_id" type="str" description="The token ID (`wk-...`) to operate on." />
<Parameter name="environment_name" type="str" description="The name of the environment to allow access to." />

**Usage**

```python notest
ws = modal.Workspace.from_context()
token = ws.proxy_tokens.create()
ws.proxy_tokens.allow(token.token_id, 'prod')
```

### proxy_tokens.revoke

```python
revoke(self, proxy_token_id, environment_name)
```

Revoke a proxy token's access to a given Environment.

The proxy token is not deleted, and it will continue to authenticate requests to any
other Environments it is associated with.

**Parameters**

<Parameter name="proxy_token_id" type="str" description="The token ID (`wk-...`) to operate on." />
<Parameter name="environment_name" type="str" description="The name of the environment to revoke access from." />

**Usage**

```python notest
ws = modal.Workspace.from_context()
ws.proxy_tokens.revoke(token_id, 'prod')
```

### proxy_tokens.delete

```python
delete(self, proxy_token_id)
```

Delete a proxy token from the Workspace.

This cannot be reverted. Any clients currently using the token will immediately
lose access to associated resources.

**Parameters**

<Parameter name="proxy_token_id" type="str" description="The token ID (`wk-...`) to delete." />

**Usage**

```python notest
modal.Workspace.from_context().proxy_tokens.delete(token_id)
```

## settings

```python
settings: WorkspaceSettingsManager
```

Namespace for Workspace settings APIs.

### settings.list

```python
list(self)
```

Return a the current workspace settings.

**Returns**

A `WorkspaceSettings` dataclass.

### settings.set

```python
set(self, name, value)
```

Set a workspace setting to a new value. Must be workspace manager or owner.

The following settings can be updated:

- image-builder-version: The image builder version determines the software included in our base images.
- default-environment: The default environment when the environment is omitted from SDK or CLI methods.

**Parameters**

<Parameter name="name" type="str" description="The name of the setting." />
<Parameter name="value" type="str" description="The new value of the setting." />

**Usage**

```python notest
modal.Workspace.from_context().settings.set('default-environment', 'dev')
```

## service_users

```python
service_users: WorkspaceServiceUserManager
```

Namespace for managing Service Users in a Workspace.

See [the guide](https://modal.com/docs/guide/service-users) for more information about Service
Users.

### service_users.create

```python
create(self, name, *, expires_in)
```

Create a new Service User in the current workspace.

To call this function, you are required to have Manager permissions in the Workspace.

**Parameters**

<Parameter name="name" type="str" description="The name of the new Service User. Must consistent only of alphanumeric, &#x27;-&#x27;, or &#x27;_&#x27; characters." />
<Parameter name="expires_in" type="int | None" description="An expiration time, in seconds, after which the token will stop working. If `None`, the token will never expire." />

**Returns**

A `TokenData` dataclass containing the credentials for the Service User. Note that this
is the only time you will be able to view these credentials.

**Usage**

```python notest
ws = modal.Workspace.from_context()

# Creates a Service User named "test-service-user", whose credentials expire after one year.
service_user_token_data = ws.service_users.create(
  'test-service-user', expires_in=datetime.timedelta(days=365).total_seconds()
)
```

### service_users.set_role

```python
set_role(self, service_user_name, environment_name, *, role)
```

Set the role of a specified Service User within an Environment.

To call this function, you are required to have Manager permissions in the Workspace.

**Parameters**

<Parameter name="service_user_name" type="str" description="The name of the Service User." />
<Parameter name="environment_name" type="str" description="The name of the Environment to which the Service User&#x27;s new role will apply." />
<Parameter name="role" type="str" description="The access level to grant this Service User." />

**Usage**

```python notest
ws = modal.Workspace.from_context()
# Grants Service User "test-service-user" the Viewer role in Environment "production"
ws.service_users.set_role('test-service-user', 'production', role='viewer')

# Revokes access to "production" from Service User "test-service-user"
ws.service_users.set_role('test-service-user', 'production', role='no-access')
```

### service_users.info

```python
info(self, name)
```

Get information about a specific Service User.

**Parameters**

<Parameter name="name" type="str" description="The name of the Service User to introspect." />

**Returns**

A `ServiceUserInfo` dataclass containing information about the Service User.

**Usage**

```python notest
ws = modal.Workspace.from_context()
service_user = ws.service_users.info('test-service-user')
```

### service_users.list

```python
list(self)
```

List all Service Users in the current workspace.

**Returns**

A list of `ServiceUserInfo` dataclasses.

**Usage**

```python notest
ws = modal.Workspace.from_context()
for service_user in ws.service_users.list():
  print(service_user.name)
```

### service_users.delete

```python
delete(self, name, *, allow_missing=False)
```

Permanently delete a Service User.

**Parameters**

<Parameter name="name" type="str" description="The name of the Service User to delete." />
<Parameter name="allow_missing" type="bool" defaultValue="False" description="If `True`, do not throw an exception if the Service User does not exist." />

**Usage**

```python notest
ws = modal.Workspace.from_context()
ws.service_users.delete('test-service-user')
```

## member_tokens

```python
member_tokens: WorkspaceMemberTokenManager
```

Namespace for managing Member API Tokens in a Workspace.

```python
__init__(self, workspace)
```

### member_tokens.list

```python
list(self)
```

List Member Tokens in the specified Workspace.

If you are a Manager or an Owner of this Workspace, this method will return a list of all the
Member Tokens in the Workspace. Otherwise, this method will only return the Tokens you have
created.

**Returns**

A list of `MemberTokenInfo` dataclasses.

**Usage**

```python notest
ws = modal.Workspace.from_context()
for member_token in ws.member_tokens.list():
  print(member_token)
```

### member_tokens.delete

```python
delete(self, token_id, *, allow_missing=False)
```

Delete a Member Token in the specified Workspace.

If you are a Workspace Manager or Owner, you can use this method to delete any Token in the
Workspace. Otherwise, you can only delete Member Tokens that you have created.

**Parameters**

<Parameter name="token_id" type="str" description="The ID of the Member Token you would like to delete." />
<Parameter name="allow_missing" type="bool" defaultValue="False" description="If `True`, do not throw an exception if the Member Token does not exist." />

**Usage**

```python notest
ws = modal.Workspace.from_context()
ws.member_tokens.delete(token_id='ak-0123456789abcdefghijkl')
```
