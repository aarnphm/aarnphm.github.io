---
date: '2022-07-01'
description: tcp optimization algorithms that maximize data-to-packet ratio by buffering small packets, including nagle's algorithm and delay ack mechanisms.
id: Nagle and TCP Cork
modified: 2025-10-29 02:15:30 GMT-04:00
tags:
  - seed
  - networking
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
