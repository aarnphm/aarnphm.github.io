---
id: Nagle and TCP Cork
tags:
  - seed
  - networking
date: "2022-07-01"
modified: 2025-01-15 14:03:40 GMT-05:00
title: Nagle's algorithm and TCP_CORK
---

## Nagle's algorithm and Delay ACK

- _small packets_ -> not for TCP
  -> Nagle algorithm: `Maximize ratio of packets - data content`
  -> Delay ACK: `silly window`

```prolog
if available_data & window_size > MSS
	send payload on wire
else
	if unconfirmed_data
		queue
	else
		send
```

## CORK algorithm
