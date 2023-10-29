---
id: Nagle and TCP Cork
tags:
  - seed
  - networking
title: Nagle's algorithm and TCP_CORK
---

### Nagle's algorithm and Delay ACK

- _small packets_ -> not for TCP
  -> Nagle algorithm: `Maximize ratio of packets - data content`
  -> Delay ACK: `silly window`

```text
if available_data & window_size > MSS
	send payload on wire
else
	if unconfirmed_data
		queue
	else
		send

```

### Cork algorithm
