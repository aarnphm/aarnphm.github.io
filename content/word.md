---
cssclasses:
  - nolist
date: '2024-10-10'
description: a bag of chips/words/vernacular
id: word
layout: technical
modified: 2026-06-28 00:55:13 GMT-04:00
tags:
  - evergreen
title: lists
---

- {{sidenotes[transitions]: in CSS or terminology}}
  - context transition
  - drill transition
  - continuity [transition](https://x.com/gabriell_lab/status/2058217175047901310)
  - tween/interpoloation
  - origin-aware animation
  - crossfade
  - morph
  - layout animation
  - direction-aware transition
  - rubber-banding
  - rippling
  - asymmetric easing
  - text morph and clip-path
  - number ticker
  - Layout thrashing
  - Spatial consistency
- Vim magic
  - `[&|?]curius=\v\d+(,\s*\d+)*//g`
  - `\v\((O[^|]*)\)(\s*\|)/$\1$\2/g`
  - `/^\s*-\s*tags:\s*\[/s/\(\[[^]]*\)\zs\([a-z0-9]\+\)-\([a-z0-9]\+\)\ze[^]]*\]/\2 \3/g`
- bash magic
  - {{sidenotes[`sudo pmset -b sleep 0; sudo pmset -b disablesleep 1`]: disable display sleeping MacOS}}
  - {{sidenotes[`sudo pmset -b sleep 5; sudo pmset -b disablesleep 0`]: revert sleeping setup on MacOS}}
- SAXPY: single-precision a \* x plus y
- BLAS: Basic Linear Algebra Subprograms
  - L1: scalar,vector,vector-vector ops
  - L2: matrix-vector ops
  - L3: matrix-matrix ops
  - ![[thoughts/images/BLAS.webp]]
- CPU operation costs ([ithare/6IT](https://6it.dev/blog/infographics-operation-costs-in-cpu-clock-cycles-741))
  - ballpark cycle counts for modern x86/x64 CPUs; the order of magnitude matters more than the exact number
  - at 3GHz, 1 cycle is ~0.33ns; 1,000,000 cycles is ~0.33ms, a 1,000,000:1 latency spread
  - arithmetic and predictable control
    | Operation | Cycles |
    | --- | --- |
    | simple register-register op (`ADD`, `OR`, etc.) | `<1` |
    | memory write | `~1` |
    | bypass delay between integer and floating-point units | `0-3` |
    | correctly predicted branch | `1-2` |
    | floating-point/vector addition | `1-3` |
    | multiplication, integer/float/vector | `1-7` |
    | return error code and check | `1-7` |
  - cache, TLB, memory
    | Operation | Cycles |
    | --- | --- |
    | L1 read | `3-4` |
    | TLB miss | `7-21` |
    | L2 read | `10-12` |
    | L3 read | `30-70` |
    | main RAM read | `100-150` |
    | NUMA different-socket L3 read | `100-300` |
    | NUMA different-socket main RAM read | `300-500` |
  - expensive scalar operations and calls
    | Operation | Cycles |
    | --- | --- |
    | branch misprediction | `10-20` |
    | floating-point division | `10-40` |
    | 128-bit vector division | `10-70` |
    | atomics/CAS | `15-30` |
    | C function direct call | `15-30` |
    | integer division | `15-40` |
    | C function indirect call | `20-50` |
    | C++ virtual function call | `30-60` |
  - allocation, NUMA, OS
    | Operation | Cycles |
    | --- | --- |
    | NUMA different-socket atomics/CAS | `100-300` |
    | allocation+deallocation pair, small objects | `200-500` |
    | kernel call | `1,000-1,500` |
    | thread context switch, direct costs | `2,000` |
    | C++ exception thrown+caught | `5,000-10,000` |
    | thread context switch, total costs including cache invalidation | `10,000-1,000,000` |
- latency numbers everyone should know ([Finbarr Timbers](https://finbarr.ca/numbers-you-should-know/))
  - latency table
    | Operation | Time (ns) | Time (ms) |
    | --- | ---: | ---: |
    | L1 cache reference | `1` | `0.000001` |
    | Branch misprediction | `3` | `0.000003` |
    | L2 cache reference | `4` | `0.000004` |
    | Mutex lock/unlock | `17` | `0.000017` |
    | Main memory reference | `100` | `0.0001` |
    | Compress 1 kB with Zippy | `2,000` | `0.002` |
    | Read 1 MB sequentially from memory | `10,000` | `0.010` |
    | Send 2 kB over 10 Gbps network | `1,600` | `0.0016` |
    | SSD 4 kB random read | `20,000` | `0.020` |
    | Read 1 MB sequentially from SSD | `1,000,000` | `1` |
    | Round trip within same datacenter | `500,000` | `0.5` |
    | Read 1 MB sequentially from disk | `5,000,000` | `5` |
    | Read 1 MB sequentially from 1 Gbps network | `10,000,000` | `10` |
    | Disk seek | `10,000,000` | `10` |
    | TCP packet round trip between continents | `150,000,000` | `150` |
  - derived throughput estimates
    | Path | Throughput |
    | --- | ---: |
    | Sequential read from HDD | `~200 MB/s` |
    | Sequential read from SSD | `~1 GB/s` |
    | Sequential read from main memory | `~100 GB/s` burst |
    | Sequential read from 10 Gbps Ethernet | `~1000 MB/s` |
  - additional observations
    | Path | Rate |
    | --- | ---: |
    | Europe to US round trips | `~6-7/s` |
    | Same-datacenter round trips | `~2000/s` |
  - sample calculation: retrieving 30 × 256 kB images from one server
    - reads required: `30 images / 2 disks per machine = 15 reads`
    - one HDD image read: `(256 kB / 1 MB) * 5ms + 10ms seek = 1.28ms + 10ms = 11.28ms`
    - total time: `15 reads * 11.28ms = 169.2ms`
    - throughput: `1000ms / 169.2ms ≈ 5 result pages/s`
- $$\begin{aligned} &\text{Big O(micron)}: O \text{ or } \mathcal{O} \\ &\text{Big Omega}: \Omega \\ &\text{Big Theta}: \Theta \\ &\text{Small O(micron)}: o \\ &\text{Small Omega}: \omega \\ &\text{On the order of}: \sim \end{aligned}$$
- time complexity of algorithm
  | Algorithm | Best | Average | Worst | Stable | In-place |
  | -------------- | ------------ | ------------ | ------------ | ------- | -------- |
  | Bubble Sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | Yes | Yes |
  | Selection Sort | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | No | Yes |
  | Insertion Sort | $O(n)$ | $O(n^2)$ | $O(n^2)$ | Yes | Yes |
  | Merge Sort | $O(n\log n)$ | $O(n\log n)$ | $O(n\log n)$ | Yes | No |
  | Quick Sort | $O(n\log n)$ | $O(n\log n)$ | $O(n^2)$ | No | Yes-ish |
  | Heap Sort | $O(n\log n)$ | $O(n\log n)$ | $O(n\log n)$ | No | Yes |
  | Counting Sort | $O(n+k)$ | $O(n+k)$ | $O(n+k)$ | Yes | No |
  | Radix Sort | $O(d(n+k))$ | $O(d(n+k))$ | $O(d(n+k))$ | Yes | No |
  | Bucket Sort | $O(n+k)$ | $O(n+k)$ | $O(n^2)$ | Depends | Depends |
  | TimSort | $O(n)$ | $O(n\log n)$ | $O(n\log n)$ | Yes | No-ish |
- tomfoolering
- ostreperous
- carthusian
- vaudeville
- {{sidenotes[saudade]: a feeling of longing, melancholy, or nostalgia that is supposedly characteristic of Portuguese temperament}}
- {{sidenotes[fin de siècle]: relating to or characteristic of the end of a century, especially the 19th century.}}
- firmament
- nipper
- aspersion
- eudaimonia
- vim
- sibilant
- {{sidenotes[quine]: is a computer program that takes no input and produces a copy of its own _source code_ as its only output.}}
- {{sidenotes[endogenous]: having an internal cause or origin}}
- nomenclature
- inundation
- {{sidenotes[joie de vivre]: exuberant enjoyment of life}}
- paratactic
- blogosphere
- {{sidenotes[ditering]: quantized color on the web}}
- {{sidenotes[gastronomy]: art of choosing, cooking, and eating good food.}}
- {{sidenotes[la cuisine nouvelle]: characterized by lighter, more delicate dishes and an increased emphasis on presentation}}
- smitheeren
- bloviating
- {{sidenotes[haute cuisine]: meticulous preparation, elaborate presentation, and the use of high quality ingredients}}
- [POSIWID](https://en.wikipedia.org/wiki/The_purpose_of_a_system_is_what_it_does)
- sacrament
- {{sidenotes[liminal space]:the in-between/transition state of being, see also [[thoughts/aesthetic value|aesthetics]]}}
- muggy
- {{sidenotes[polyphony]: a feature of narrative, which includes a diversity of simultaneous points of view and voices}}
- dasein
- [amatonormativity](https://elizabethbrake.com/amatonormativity)
- row-major versus column-{{sidenotes[major]: indicates how we order data in memory for flattened array}}
  - row-major memory locations:
    - $$x + N_x \cdot y$$
    - Access `A[y][x]`
  - col-major memory locations:
    - $$y + N_y \cdot (x-1)$$
    - Access `A(y, x)`
  - Note that for transpose, we should use col-major instead.
    - see also: https://en.wikipedia.org/wiki/Row-_and_column-major_order
- polysyllabic
  - [[thoughts/Epistemology]] versus [[thoughts/Ontology]]
- epistolary
- {{sidenotes[triptych]: a picture/relief carving on three panels, typically hinged together side by side and used as an altarpiece}}
- meliorism
- lex parsimoniae
  - also known as law of parsimony, circa Occam's razor
- {{sidenotes[kalsarikännit]: drink alone in your underwear with no intention of going out}}
- ample
- {{sidenotes[stork]: Baby-bringing bird}}
- pesky
  - annoying to deal with
- interlocutors
- lowkenuinely
  - i.e. lowkey + genuinely
- a priori
- {{sidenotes[a fortiori]: mostly used to lead from a less certain proposition to a more evident corollary, i.e. "even more so"}}
- passementerie
  - fancy decorations on clothes
- jovial
- {{sidenotes[claudification]: the [[library/Phenomenology of Perception|phenomena]] where Claude Code "harnesses" tasks by attributing a verb describing what it is doing.}}
- accomplishing
- actioning
- actualizing
- baking
- booping
- brewing
- calculating
- cerebrating
- churning
- clauding
- coalescing
- cogitating
- computing
- concocting
- considering
- contemplating
- cooking
- crafting
- creating
- crunching
- deciphering
- deliberating
- determining
- discombobulating
- doing
- effecting
- enchanting
- envisioning
- finagling
- flibbertigibbeting
- forging
- forming
- frolicking
- generating
- germinating
- hatching
- herding
- honking
- ideating
- imagining
- incubating
- inferring
- manifesting
- marinating
- meandering
- moseying
- mulling
- mustering
- musing
- noodling
- percolating
- perusing
- philosophising
- pontificating
- pondering
- processing
- puttering
- puzzling
- reticulating
- ruminating
- scheming
- schlepping
- simmering
- smooshing
- spelunking
- [[thoughts/sheafification|sheafification]]
- spinning
- stewing
- sussing
- synthesizing
- thinking
- tinkering
- transmuting
- unfurling
- unravelling
- vibing
- wandering
- whirring
- wibbling
- working
- wrangling
- iconoclastic
- quixotic
- asinine
- {{sidenotes[semiotic]: the study of signs}}
- {{sidenotes[chauvinist]: a person who displays an aggressive, unreasonable, and fanatical belief in the superiority of their own group}}
- peleton
- domestique
- fevor
- {{sidenotes[truculent]: someone who is aggressively self-assertive, defiant, and easily angered}}
- preclude
- Euler's polyhedron formula
  - $V-E+F$
- {{sidenotes[mnestics]: see also [lesswrong](https://www.lesswrong.com/s/YshDJ9ECCgkwoNsBc/p/ksatPnddyZjHwZWwG), but largely this term comes from the book [[library/There is No Antimemetics Division|There is No Antimemetics Division]] by qntm}}
- {{sidenotes[portmanteau]: In literature, also known in linguistics and lexicography as a blend word, lexical blend, or simply a blend, is a word formed by combining the meanings and parts of the sounds of two or more words.}}
- {{sidenotes[solipsistic]: the philosophical belief that only one's own mind is sure to exist. In every day language, it can be interpreted as self-centered, self-absorbed}}
- {{sidenotes[errata]: a list of corrections or errors discovered in a published work}}
- lingua franca
- Torschlusspanik
- Kairos
- pêche à pied
- revenant
- moratorium
- pervasive
- myopic
- subaqueous
- rollicking
