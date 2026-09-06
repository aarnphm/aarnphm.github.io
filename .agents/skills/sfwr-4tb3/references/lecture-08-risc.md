---
name: lecture-08-risc
description: risc-v (and mips) as target — registers, instruction formats, translation schemes, register allocation
---

# lecture 08 — a risc architecture as target

## risc-v at a glance

- 32 integer registers `x0..x31` of 32 bits (base RV32I). `x0` is hardwired to $0$, writes ignored.
- ABI names: `zero`, `ra` (return addr), `sp`, `gp`, `tp`, `t0..t6` (temps, caller-saved), `s0..s11` (saved, callee-saved), `a0..a7` (args/returns, caller-saved)
- three-address ops only on registers (no mem-mem). immediate versions suffix `i` (e.g. `addi`).
- memory access via `lw rd, offset(rs)` and `sw rs, offset(rd)`. offset is 12-bit signed.
- `la rd, label` loads a 32-bit address in two instructions (pseudo-op).

## ops cheat-sheet

| instr                             | effect                                            |
| :-------------------------------- | :------------------------------------------------ |
| `add rd, rs1, rs2`                | $R[rd] := R[rs_1] + R[rs_2]\ (\bmod\ 2^{32})$     |
| `sub`, `mul`, `div`, `rem`        | same shape                                        |
| `addi rd, rs, imm`                | immediate version                                 |
| `lw rd, n(rs)`                    | $R[rd] := M[R[rs] + n]$                           |
| `sw rs, n(rd)`                    | $M[R[rd] + n] := R[rs]$                           |
| `la rd, label`                    | load address (two-instr pseudo)                   |
| `beq/bne/blt/bge rs1, rs2, label` | conditional branch, $pc\mathrel{+}=\text{offset}$ |
| `jal rd, label`                   | $R[rd] := pc + 4;\ pc := \text{label}$            |
| `jalr rd, rs, imm`                | $R[rd] := pc + 4;\ pc := R[rs] + \text{imm}$      |

## P0 register convention

the P0 compiler keeps a set `regs` of in-use registers. $\text{GPregs} = \{t_0..t_6, s_0..s_{11}, \ldots\}$ (general-purpose). `obtainReg()` grabs a fresh one; `releaseReg(r)` frees it. running out marks an error: `mark('RISC-V: out of registers')`.

## translation scheme — expressions (risc-v)

| $E$              | $\text{code}(E)$                                                   | register                        |
| :--------------- | :----------------------------------------------------------------- | :------------------------------ |
| $x$ (local)      | `lw reg, offset(fp)`                                               | `reg`                           |
| $x$ (global)     | `la` $reg_1,$ `x_;` `lw` $reg_2,$ `0(`$reg_1$`)`                   | $reg_2$                         |
| $n$ (constant)   | `addi reg, zero, n`                                                | `reg` (only if $n$ fits in imm) |
| $E_1 \times E_2$ | $\text{code}(E_1);\ \text{code}(E_2);$ `mul reg,` $reg_1,$ $reg_2$ | `reg`                           |
| $E_1 + E_2$      | $\text{code}(E_1);\ \text{code}(E_2);$ `add reg,` $reg_1,$ $reg_2$ | `reg`                           |
| $-E$             | $\text{code}(E);$ `sub reg, zero, reg_E`                           | `reg`                           |

when the compiler issues `mul reg,` $reg_1,$ $reg_2$, it also releases $reg_1$ and $reg_2$ back to `regs` ($reg_1$ is reused as `reg` in practice).

## translation scheme — statements

| $S$                                                | $\text{code}(S)$                                                                                             |
| :------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| $x := E$ (local)                                   | $\text{code}(E);$ `sw reg_E, offset(fp)`                                                                     |
| $x := E$ (global)                                  | $\text{code}(E);$ `la` $reg_1,$ `x_;` `sw reg_E, 0(`$reg_1$`)`                                               |
| $\text{if}\ E\ \text{then}\ S_1\ \text{else}\ S_2$ | $\text{code}(E);$ `beq reg_E, zero, Lelse;` $\text{code}(S_1);$ `j Lend; Lelse:` $\text{code}(S_2);$ `Lend:` |
| $\text{while}\ E\ \text{do}\ S$                    | `Lloop:` $\text{code}(E);$ `beq reg_E, zero, Lend;` $\text{code}(S);$ `j Lloop; Lend:`                       |

relational comparisons usually fuse with the branch: $x < y \to \ldots;$ `blt reg_x, reg_y, Ltrue`.

## procedure calling convention (P0, risc-v)

prologue:

1. allocate frame on stack: `addi sp, sp, -framesize`
2. save return address and old fp: `sw ra, 0(sp); sw fp, 4(sp)`
3. set new fp: `addi fp, sp, 0`

epilogue:

1. restore `ra, fp`: `lw ra, 0(sp); lw fp, 4(sp)`
2. `addi sp, sp, framesize`
3. `jalr x0, ra, 0` (i.e. `ret`)

arguments go in `a0..a7`, overflow to the stack. return values in `a0`, `a1`.

## mips highlights (diff vs risc-v)

same philosophy, different names. `$zero, $t0..$t9, $s0..$s7, $sp, $fp, $ra, $v0, $a0..$a3`. slots like `lw $t0, offset($fp)`. multiplication uses `mult` with `HI/LO` registers on classic mips; many assemblers accept `mul $d, $s, $t` as pseudo.

jumps: `j label`, `jal label`, `jr $ra`. branches: `beq $s, $t, label`, `bne`, `slt` (set-less-than).

## annotating risc-v code (lab 10)

"annotating" means: next to each instruction, write which P0 expression/statement it came from and which register holds which value. the pattern:

```
lw t0, -4(fp)           # t0 = x
lw t1, -8(fp)           # t1 = y
add t0, t0, t1          # t0 = x + y
sw t0, -12(fp)          # z := x + y
```

questions often ask for this annotation given a P0 program, or for the matching P0 program given the risc-v.
