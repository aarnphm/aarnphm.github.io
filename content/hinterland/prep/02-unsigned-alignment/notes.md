# Unsigned semantics and data alignment

Module 02 of the encoding screen kit. Companion files: `problems.py` (stubs), `solutions.py` (reference), `test_problems.py` (harness).

## 1. Core mental model

**Unsigned arithmetic is arithmetic in $\mathbb{Z}/2^n\mathbb{Z}$.** An $n$-bit unsigned value is a residue mod $2^n$; add, sub, mul are exact integer operations followed by reduction mod $2^n$. Overflow is not an event, it is the reduction. So in u32: `0xFFFFFFFF + 1 == 0`, `0 - 1 == 0xFFFFFFFF` ($\equiv -1$), `0x10000 * 0x10000 == 0`.

**Two's complement is a relabeling of the same residues.** The signed value of pattern $x$ is $x$ if $x < 2^{n-1}$, else $x - 2^n$. Add, sub, mul produce identical bit patterns for signed and unsigned (one ALU adder serves both); comparison, division, right shift, and widening are where the interpretations diverge.

**C**: unsigned overflow is defined as mod-$2^n$ wrap (C11 6.2.5p9: "can never overflow ... reduced modulo"). Signed overflow is UB (6.5p5), and optimizers exploit it: `x + 1 > x` folds to true for `int x`; `for (int i = 0; i <= n; ++i)` with `n == INT_MAX` is a loop the compiler may "prove" terminates. `-fwrapv` buys wrapping semantics, `-fsanitize=signed-integer-overflow` traps.

**Python**: one `int`, arbitrary precision, no unsigned, no widths. Model negatives as infinite two's complement (unbounded leading 1s): `-1 & 0xFF == 255`, `-8 >> 1 == -4` (shifts are arithmetic and floor). Fixed-width unsigned is a discipline, not a type: mask with $2^n - 1$ at every step that can carry out.

Divergences interviewers poke at:

| operation                        | C `uint32_t` / `int32_t`                                     | Python `int`                                                     |
| -------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| overflow on `+ - *`              | unsigned wraps; signed UB                                    | value grows, never wraps                                         |
| `x >> 1`, x negative             | implementation-defined (arithmetic everywhere real)          | arithmetic, floors: `-5 >> 1 == -3`                              |
| shift count $\ge$ width or $< 0$ | UB (x86 masks count mod 32/64; ARM32 register shifts give 0) | `x << 200` fine; negative count raises                           |
| `-7 / 2`, `-7 % 2`               | `-3`, `-1` (truncate toward 0)                               | `-7 // 2 == -4`, `-7 % 2 == 1` (floor; `%` takes divisor's sign) |
| width                            | fixed by type                                                | unbounded; emulate with `& ((1 << n) - 1)`                       |

The `%` row bites in ring buffers: `(head - 1) % cap` is already correct in Python, needs `(head + cap - 1) % cap` in C. The shift row: C `-5 >> 1` (arithmetic) is $-3$ while C `-5 / 2` is $-2$; shift is not division for negatives.

## 2. Idioms

### 2.1 Masking discipline: Python as a fixed-width machine

```python
M32 = (1 << 32) - 1  # 2**32 - 1 == 0xFFFFFFFF
add32 = lambda a, b: (a + b) & M32
neg32 = lambda x: -x & M32  # two's complement negate == ~x + 1
shr32 = lambda x, n: (x & M32) >> n  # LOGICAL shift: mask BEFORE shifting
rotl32 = lambda x, r: ((x << r) | ((x & M32) >> (-r & 31))) & M32
```

Worked: `neg32(1)` is $-1 \bmod 2^{32}$ = `0xFFFFFFFF`. `rotl32(0x80000001, 1)`: `x << 1 = 0x1_0000_0002`, `x >> 31 = 1`, or + mask gives `0x00000003`. The `-r & 31` handles `r == 0` without ever writing the UB-shaped `x >> 32` (in C the same trick, `x << r | x >> (-r & 31)`, is the recognized-and-optimized rotate idiom).

C runs the same discipline in reverse: storing into a narrow unsigned type masks for you, but intermediates are computed in the promoted type and are NOT masked (see the ladder in section 3).

### 2.2 Sign bridges: to_signed / to_unsigned

`to_unsigned` is just the mask, and it is exactly C's signed-to-unsigned conversion rule (defined for all inputs):

```python
def to_unsigned(x, n):
  return x & ((1 << n) - 1)  # x mod 2**n
```

`to_signed`, the xor-and-subtract trick, with $s = 2^{n-1}$:

```python
def to_signed(u, n):
  s = 1 << (n - 1)
  return ((u & ((1 << n) - 1)) ^ s) - s
```

Why it works: xor with the sign bit adds $s$ mod $2^n$ (swaps the halves of $[0, 2^n)$), then subtracting $s$ re-centers into $[-s, s)$; the result is the unique value in that range congruent to $u$ mod $2^n$. Worked, $n=8$, $u=$ `0xFE`: `0xFE ^ 0x80 = 0x7E` (126), $126 - 128 = -2$.

Equivalents worth saying out loud:

```python
u - (1 << n) if u >= (1 << (n - 1)) else u  # branchy version
int.from_bytes((0xFE).to_bytes(1, 'little'), 'little', signed=True)  # -2
(-2).to_bytes(1, 'little', signed=True)  # b'\xfe' round-trips
struct.unpack('<i', struct.pack('<I', 0xFFFFFFFE))[
  0
]  # -2, same bridge at 32 bits
```

C side: unsigned-to-signed with an out-of-range value is implementation-defined (C11 6.3.1.3p3); every mainstream compiler wraps, and C++20 defines the wrap. Portable C spells it with the same xor-subtract or `memcpy`.

### 2.3 Alignment formulas

For power-of-two $a$: $a - 1$ is a mask of the low $\log_2 a$ bits, and $\lnot(a-1) = -a$ in two's complement.

```
align_down(x, a) = x & ~(a - 1)  ==  x & -a        clear low bits
align_up(x, a)   = (x + a - 1) & ~(a - 1)           bias, then clear
is_aligned(x, a) = (x & (a - 1)) == 0
```

Worked, $a=8$: `align_up(13, 8)`: $13 + 7 = 20 =$ `0b10100`, `& 0b...11000` gives 16. `align_up(16, 8)`: $16 + 7 = 23$, clears back to 16 (idempotent; writing `+ a` instead of `+ a - 1` is the classic bug that bumps aligned inputs to 24). `align_up(0, a) == 0`.

Properties to state unprompted in an interview (the tests property-check exactly these): result $\ge x$; $0 \le \mathrm{align\_up}(x,a) - x < a$; result $\bmod a = 0$; idempotent; monotone in $x$.

Power-of-two is load-bearing: for $a = 12$, $\lnot(a-1) = \lnot 11 = $ `...110100` is not a contiguous mask and clears garbage bits. General-$a$ version is `(x + a - 1) // a * a` (one division instead of an AND). The power-of-two test is `a > 0 and a & (a - 1) == 0`; `x & (x - 1)` clears the lowest set bit, `x & -x` isolates it.

Two C-only caveats: `x + a - 1` can wrap near the top of `uintptr_t` (kernel code checks first, or uses `r = x & (a - 1); if (r) x += a - r;`). And Python, uniquely, floors correctly for negative `x` through the same expression (`-5 & -4 == -8`) because its ints behave as infinite two's complement, where C unsigned would wrap first.

### 2.4 Why hardware cares about alignment

- **Load granularity.** L1 delivers data in aligned chunks. Modern x86 does any load _within_ a cache line at full speed; the costs are crossings. An 8-byte load at a uniformly random address crosses a 64 B line with probability $7/64 \approx 11\%$ and costs roughly $2\times$ (two line reads merged); crossing a 4 KiB page is far worse (two TLB lookups, historically 100+ cycles).
- **Atomics.** Lock-free guarantees hold for naturally aligned words only. On x86 an aligned load/store up to 8 B is atomic; a `lock` RMW spanning two cache lines is a split lock that locks the bus (recent CPUs can raise #AC; Linux `split_lock_detect` warns or kills). Misaligned atomics fault, tear, or stall the whole machine, depending on ISA.
- **SIMD.** SSE `movdqa` #GP-faults unless 16-B aligned (`movdqu` is the unaligned form); 64 B for the AVX-512 aligned moves. Even where unaligned is legal, a 32-B vector load at a random offset splits a line $31/64 \approx 48\%$ of the time.
- **ISA spread.** x86 and ARMv8 tolerate unaligned scalar access (except ARM exclusives/atomics); older ARM/MIPS/SPARC delivered SIGBUS, and kernel fixup handlers turned each access into a trap, about $10^3\times$ slower.
- **GPUs.** Vectorized loads (`float4`, `ld.global.v4.b32`) require 16-B alignment; misalignment degrades a warp's single coalesced transaction into several. This is why tensor strides and KV-cache block layouts get padded to 16/32-element boundaries.
- **Language rules.** In C/C++ merely dereferencing a pointer misaligned for its type is UB even on x86; autovectorizers emit `movdqa` on that assumption and crash real code. `-fsanitize=alignment` catches it. Rust: constructing a misaligned `&T` is instant UB; use `ptr::read_unaligned`.

Natural alignment: `alignof(T) == sizeof(T)` for scalars (1, 2, 4, 8, 16). ABI wrinkle: i386 SysV aligns `double` to 4 inside structs while Windows x86 uses 8; layout is ABI, not physics. Cache lines: 64 B on x86 and most ARM servers, 128 B on Apple M-series.

Unaligned access, the sanctioned idioms:

```c
uint32_t v;
memcpy(&v, p, sizeof v);      /* one mov on x86/arm64, zero UB; never *(uint32_t *)p */
```

```python
struct.unpack_from('<I', buf, off)  # any offset; standard modes never align
int.from_bytes(
  mv[off : off + 4], 'little'
)  # mv = memoryview(buf) avoids copies in loops
```

Allocation: `malloc` returns storage aligned for `max_align_t` (16 B on mainstream 64-bit ABIs). Stricter needs `aligned_alloc(64, n)` (C11 pedantically wants `n` a multiple of the alignment; later drafts and glibc relax) or `posix_memalign`. Over-aligned types allocate correctly via `new` only since C++17.

### 2.5 C struct layout: the algorithm

Rules (SysV-flavored; gcc/clang/msvc agree absent pragmas):

1. `cursor = 0`. For each member in declaration order: `offset = align_up(cursor, alignof(member))`, then `cursor = offset + sizeof(member)`. C never reorders members (Rust `repr(Rust)` may; `repr(C)` does not).
2. `alignof(struct) = max(alignof(member))`, 1 if empty.
3. `sizeof(struct) = align_up(cursor, alignof(struct))`: tail padding.

Worked, LP64, `struct { char a; int b; short c; }`:

| member | align | offset | bytes                  |
| ------ | ----- | ------ | ---------------------- |
| `a`    | 1     | 0      | `a . . .` (3 pad)      |
| `b`    | 4     | 4      | `b b b b`              |
| `c`    | 2     | 8      | `c c . .` (2 tail pad) |

sizeof 12, alignof 4. Byte map: `a...bbbbcc..`

Tail padding exists because of arrays: `&arr[1] == &arr[0] + sizeof(T)`, and `arr[1].b` must still be 4-aligned, so sizeof must be a multiple of the struct alignment; without the 2 tail bytes, `arr[1].b` would land at offset 10.

Reordering to shrink: `{char; double; char}` lays out at offsets 0, 8, 16, sizeof 24. Declared `{double; char; char}`: $8 + 1 + 1 = 10$, padded to 16. That is 8 bytes (a third) recovered by moving two declarations. The rule and the theorem behind it: sort members by descending alignment. When every member's size is a multiple of its alignment (true of every real C type), each prefix sum stays aligned for every remaining member, so internal padding is zero and $\mathrm{sizeof} = \mathrm{align\_up}(\sum \mathrm{sizes}, \max \mathrm{align})$, which no ordering beats ($\mathrm{sizeof} \ge \sum \mathrm{sizes}$ and must be a multiple of the max alignment). `-Wpadded` reports every inserted pad byte.

Follow-up ammo, one line each: `#pragma pack(1)` / `__attribute__((packed))` caps member alignment (then passing a pointer to a packed member into code expecting alignment is the UB everyone ships); `alignas(64)` over-aligns (standard false-sharing fix); bit-field packing is implementation-defined, do not claim exact layout, say "ABI-specific"; a flexible array member `T x[];` adds no size but its alignment counts.

### 2.6 Python `struct`: native vs standard, with receipts

| mode          | byte order             | sizes    | padding                      |
| ------------- | ---------------------- | -------- | ---------------------------- |
| `@` (default) | native                 | native   | native, between members only |
| `=`           | native                 | standard | none                         |
| `<` `>` `!`   | little / big / network | standard | none                         |

```python
>>> struct.calcsize('<ci')    # standard: packed
5
>>> struct.calcsize('@ci')    # native: 3 pad bytes before the int
8
>>> struct.calcsize('@ic')    # native, but NO tail padding
5
>>> struct.calcsize('@ic0i')  # trailing 0i forces tail pad to int alignment
8
>>> struct.calcsize('<l'), struct.calcsize('@l')   # standard 'l' is 4 bytes; native 8 on LP64
(4, 8)
```

So `@ic` is 5 while `sizeof(struct { int; char; })` is 8: the module pads between members and never at the end; emulate tail padding with a trailing `'0'+code`. For wire formats always use `<` or `>` (fixed sizes, zero padding, no host dependence). Use `@` only to match an in-memory C struct on the same host, and remember both the `0X` trick and that native `'l'`/`'L'` change width across platforms.

## 3. Gotchas and interviewer follow-ups

The integer promotion ladder (C11 6.3.1.1): rank order `_Bool < char < short < int < long < long long`. Any operand of rank below `int` (includes `uint8_t`, `uint16_t`, bit-fields) first converts to `int` if `int` represents all its values (on 32-bit-int platforms it does), otherwise `unsigned int`. `uint32_t` has rank equal to `int` and stays put. Then the usual arithmetic conversions reconcile the two operands; at equal rank, signed converts to unsigned; a higher-ranked signed type that can represent the whole unsigned range absorbs it instead. Every trap below is one of those two rules firing.

1. **`(uint8_t)x << 24 >> 24` sign-extends** (this is `c_promote_trap` in `problems.py`). The u8 promotes to _signed_ `int`; `<< 24` parks its top bit in the sign bit (formally UB for values $\ge 128$, C11 6.5.7p4; defined in C++20; two's complement pattern in practice everywhere); `>> 24` on the now-negative int is arithmetic. `0xFF` comes back as $-1$, not 255. Fix: `(uint32_t)x << 24`.
2. **`-1 < 1u` is false.** Equal rank, signed converts: $-1$ becomes $2^{32}-1$. Corollary: `for (int i = 0; i < v.size() - 1; ++i)` with `size() == 0` compares `i` against $2^{64}-1$ and runs until it faults. `-Wsign-compare` (via `-Wextra`) flags it. The trap is rank-sensitive, which is why it survives review: `-1L < 1u` on LP64 is _true_ (`long` outranks and absorbs `unsigned int`).
3. **`size_t` reverse loop:** `for (size_t i = n - 1; i >= 0; --i)` never terminates; `i >= 0` is a tautology and $0 - 1$ wraps to $2^{64}-1$. Idiomatic fixes: `for (size_t i = n; i-- > 0;)` (the "goes-to operator" `i --> 0`), or loop over a signed index.
4. **`~` promotes too.** With `uint8_t s = 1`, `~s` is `int` `0xFFFFFFFE`, so `if (~s == 0xFE)` is false. Fix: `(uint8_t)~s` or `~s & 0xFF`. Same for unary minus and for `u8a - u8b` (int math, fine, until compared against an unsigned).
5. **`uint16_t * uint16_t` is signed-overflow UB.** Both promote to `int`; $65535 \times 65535 = 4294836225 > 2^{31}-1$. The famous one. Fix: `(uint32_t)a * b`. Adjacent trap: `uint64_t p = a * b;` with `uint32_t` operands is defined but multiplies in 32 bits and truncates before widening; write `(uint64_t)a * b`.
6. **Wraparound comparison.** For unsigned `a, b`: `a - b > 0` just means `a != b`. Serial-number arithmetic (RFC 1982, TCP sequence space) does it on purpose: `(int32_t)(a - b) > 0` is wrap-aware "a is after b" and keeps working when sequence numbers roll over; the plain compare `a > b` breaks at the wrap.
7. **Shift counts.** `x << 32` (u32) and `x << -1` are UB: x86 masks the count mod 32/64 so `x << 32 == x` at runtime, ARM32 register shifts use the low byte and give 0, and the compiler may fold to yet a third answer. Separately `1 << 31` is UB by itself (signed `int` overflow): write `1u << 31` or `UINT32_C(1) << 31`.
8. **`char` signedness is platform-split.** Signed on x86 Linux/macOS, unsigned on ARM/PowerPC/s390x Linux. `char c = buf[i]; int v = c << 8;` sign-extends on one and not the other; decode paths use `uint8_t` exclusively. Related: `char c; while ((c = getchar()) != EOF)` either never exits (unsigned char) or exits early on byte `0xFF` (signed); `c` must be `int`.
9. **Midpoint overflow.** `(lo + hi) / 2` is UB for `int`, wraps for u32; `lo + (hi - lo) / 2` is the fix (this bug sat in Java's `Arrays.binarySearch` for nine years).
10. **Python-side traps.** `>>` of a negative never reaches 0 (`-1 >> k == -1` for all $k$, infinite sign bits), so `while x: x >>= 1` hangs for negative x; mask first. And one forgotten mask silently promotes your "u64" into a bignum whose value diverges from the C reference at that step and never comes back.

Follow-ups interviewers actually ask, capsule answers:

- "What does the standard guarantee about overflow?" Unsigned: reduction mod $2^n$, never UB. Signed: UB. Conversion _to_ unsigned: defined mod $2^n$. Conversion to signed of an out-of-range value: implementation-defined (wraps in practice; C++20 defines the wrap).
- "Why is signed overflow UB rather than wrapping?" It licenses optimization: induction-variable widening, `a + 1 > a` folding, trip-count proofs. `-fwrapv` trades those away.
- "Read a u32 from a maybe-unaligned pointer, portably?" `memcpy` into a local; compiles to one load; `*(uint32_t *)p` is UB and really breaks under autovectorization.
- "sizeof / alignof of `struct { char c; double d; }`?" 16 and 8.
- "When does misalignment cost on x86?" Cache-line splits (about $2\times$) and page splits (much worse); split-lock atomics (bus lock). Within one line, free on modern cores.
- "What alignment does malloc give?" `alignof(max_align_t)`, 16 B on mainstream 64-bit; use `aligned_alloc`/`posix_memalign` for cache-line or page alignment.
- "Why does tail padding exist?" Arrays: element stride is sizeof, and every element's members must stay aligned, so sizeof is a multiple of alignof.
- "Shrink this struct" leads to descending-alignment reordering plus the zero-internal-padding argument from section 2.5 (`reorder_fields` in `problems.py`).

## 4. Rapid-fire drills

1. `align_up(0, 16)`? $\to$ 0.
2. `align_up(17, 16)` / `align_down(17, 16)`? $\to$ 32 / 16.
3. Why power-of-two only for `x & ~(a - 1)`? $\to$ `a - 1` must be a contiguous low-bit mask; $a = 12$ gives `0b1011` and clears the wrong bits.
4. u32 `0 - 1`? $\to$ $2^{32}-1 = $ `0xFFFFFFFF`, defined. `int` `INT_MIN - 1`? $\to$ UB.
5. C: type and value of `(uint8_t)250 + (uint8_t)10`? $\to$ `int`, 260; narrows to 4 only when stored into a `uint8_t`.
6. C: `-1 < 1u`? $\to$ false. `-1L < 1u` on LP64? $\to$ true (rank).
7. Python `-5 >> 1`? $\to$ $-3$ (floor). C `-5 / 2`? $\to$ $-2$ (truncate). Same machine, different rounding.
8. `to_signed(0xFE, 8)`? $\to$ $-2$ (`0xFE ^ 0x80 = 0x7E`, $126 - 128$).
9. `to_unsigned(-1, 64)`? $\to$ $2^{64}-1$.
10. `sizeof(struct { char; int; short; })`? $\to$ 12 (offsets 0, 4, 8; 2 tail bytes).
11. Minimal sizeof after reordering `{char; double; char}`? $\to$ 16, down from 24.
12. `struct.calcsize('@ic')` vs `sizeof(struct { int; char; })`? $\to$ 5 vs 8; the module never tail-pads; append `'0i'`.
13. `x & (x - 1)` and `x & -x`? $\to$ clear lowest set bit; isolate lowest set bit. Power-of-two test: `x && !(x & (x - 1))`.
14. Read u16 LE at byte offset 3 of `buf` in Python? $\to$ `struct.unpack_from('<H', buf, 3)[0]`.
15. Read u64 at unaligned `p` in C? $\to$ `uint64_t v; memcpy(&v, p, 8);`.
16. Is 0 aligned to 4096? $\to$ yes; `is_aligned(0, a)` for every `a`.
17. What does u32 `a - b > 0` test? $\to$ `a != b`; wrap-aware ordering is `(int32_t)(a - b) > 0`.
18. Why is `1 << 31` UB but `1u << 31` fine? $\to$ left operand is `int` and overflows; unsigned reduces mod $2^{32}$.
