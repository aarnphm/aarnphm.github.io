---
id: "intern"
tags: []
---

Map to url to uuid

large uuid be? collision

tradeoff? actual mapping vs. hash

mapping:
- no collision (store the actual mapping)


> Optimized for reading, (some sort of cache)

> Build URL shortener for real world (1b/month)

> Peak traffic

## Design

hash? or random uuid

how many bit or hash space?

- 64 bit or 128 bit hash space

How many URL are you assuming it to be?

Read and write?

incremental counter?

- challenging because of global sequence
- RW Lock?
  - Collision?

- Security concern?
- guessable?

10e6 entries/month

## Storage

Store this url? do we need to store it? schema?

Not storing?

- two way hash
- don't need to store URL?
- can be collision (popular with same shorten URL) (nice if user can bring this
  up)

Storing?

- schema in db
- metadata, expiration
- estimate how big is the db? partition scheme or storage?

  8/16 bytes of uuid space, maxlen of the size (128/256 bytes)
  - not scalable in instance db, not availability
  - how partition? replicate db
  - partition based on url keyspace
  - hashing based on production

## Serving

- read/write ratio?
  - 1-to-10
  - read (caching layer)
  - only write to hit db

Bottleneck is DB

api server -> cache -> DB

Assumption:
- hot url
- stack (minimize db access) -> horizontal scale on web server
- write traffic? write in DB? reasonable db can handle this traffic?

Cache: Eviction, LRU

- Multi-region (handles copies)

## Follow up
- expire bit.ly links
- How to avoid collision (increase hash space)
    - same uuid -> what happened?

- Security (reverse engineer)


## Store the mapping, what is the db schema?

partition?

## Curious and demonstrate continuous learning

Q: talk about one thing you learned the most recently? assume college grad,
understand the details

- curiosity and continuous learning
- communication skill?
  - explain complex topics for a 5 year old

## Intro

distributed team, work on multiple position. remote, doesn't have to be sf, 40/h
week in cad, 500$ stipend in home setup

BentoML does, problem solution and value prop

gage commitment in terms of involvement into the space?J

rate out of 5

## Investigating 'process is overloaded'

- explain when asyncio.CancelledError
- something called on the future.cancel()

if they don't know about python, then they should ask.

raised when the wait time is exceeded.

tell them upfront to ask questions that they don't understand.

1. sata
2. logan
3. david
