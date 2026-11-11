# Cluster

```python
class Cluster(modal.object.Object)
```

A group of containers scheduled together for a clustered Function or Server.

Use `Cluster.from_context()` inside a cluster, or `Cluster.from_id()` to
inspect a cluster remotely. Containers are ordered by cluster rank.

## object_id

```python
object_id(self)
```

The cluster's unique `cu-` object ID.

## hydrate

```python
hydrate(self, client=None)
```

Synchronize the local object with its identity on the Modal server.

It is rarely necessary to call this method explicitly, as most operations
will lazily hydrate when needed. The main use case is when you need to
access object metadata, such as its ID.

_Added in v0.72.39_: This method replaces the deprecated `.resolve()` method.

## from_context

```python
from_context()
```

Reference a Cluster from within one of its containers.

Raises `InvalidError` outside an initialized clustered execution.

## from_id

```python
from_id(cluster_id, *, client=None)
```

Reference a Cluster by its ID.

**Parameters**

<Parameter name="cluster_id" type="str" description="ID of the cluster." />
<Parameter name="client" type="_Client | None" defaultValue="None" description="Modal client to use; defaults to `Client.from_env()` when omitted." />

**Usage**

```python notest
cluster = modal.Cluster.from_id('cu-123')
```

## container_ids

```python
container_ids(self)
```

Return container IDs ordered by cluster rank.

## container_ips

```python
container_ips(self, family='ipv6')
```

Return container IP addresses ordered by cluster rank.

Returns IPv6 addresses by default; pass `family="ipv4"` for IPv4.
These addresses are for intra-cluster communication.

Must be called from a container in this cluster; otherwise raises `InvalidError`.

## container_rank

```python
container_rank(self, container_id=None)
```

Return a container's rank within this cluster.

With no argument, return the executing container's rank without a network
request. Raises `InvalidError` if it is not a member of this cluster.

With an explicit container ID, look up its rank in the cluster's membership.
Raises `InvalidError` for nonmembers.
