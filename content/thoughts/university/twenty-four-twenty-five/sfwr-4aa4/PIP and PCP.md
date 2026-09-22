---
date: '2024-12-18'
description: priority inheritance protocol and priority ceiling protocol for real-time scheduling with resource contention and deadlock prevention.
id: PIP and PCP
modified: 2026-09-22 09:10:18 GMT-04:00
tags:
  - sfwr4aa4
title: PIP and PCP
---

Assume one CPU, preemptive scheduling, fixed base priorities, and mutex-protected resources. Larger priority values mean higher priority. The classical analysis also requires bounded, properly nested critical sections, all locks released by job completion, and no self-suspension such as waiting for I/O. These assumptions matter for the blocking guarantees below. [Sha, Rajkumar, and Lehoczky (1990)](https://www.cse.iitb.ac.in/~cs431/papers/pi-pcp.pdf)

## Priority Inheritance Protocol (PIP)

A high-priority task waiting for a lock needs its owner to run. Otherwise, unrelated medium-priority work can keep preempting the owner and delaying the waiting task. PIP raises the owner's priority when contention occurs.

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/PIP.webp]]

Keep two priorities separate: a task's assigned **base priority** and its **effective priority** while inheriting from blocked tasks.

- When a task blocks on an occupied resource, its owner inherits the waiter's effective priority if that is higher.
- Inheritance is transitive. If $T_1$ waits for $T_2$, which waits for $T_3$, the priority inherited from $T_1$ reaches $T_3$.
- After an unlock, recompute the owner's priority from its base priority and the waiters on every resource it still holds. Releasing one lock can leave another donation active. The owner returns to its base priority when no remaining donation exceeds it. [POSIX priority-inheritance semantics](https://pubs.opengroup.org/onlinepubs/7908799/xsh/pthread_mutexattr_setprotocol.html)

The original paper describes restoring the priority held on entry to a critical section within its nested-locking model. Dropping straight to base priority on every unlock would lose inheritance from an outer lock.

PIP still permits circular waiting: two tasks can each hold a resource the other needs. Raising their priorities cannot make either lock available. Even with deadlock prevented separately, a job may encounter several lower-priority critical sections.

## Priority Ceiling Protocol (PCP)

The original PCP adds an admission rule to inheritance. Assign each resource a fixed ceiling from the base priorities of every task that may lock it:

$$
\operatorname{ceiling}(R)
= \max_{T\text{ may lock }R} p_{\mathrm{base}}(T).
$$

For a requesting task $T$, take the highest ceiling among resources **held by other tasks**. Call this the system ceiling seen by $T$, $C_{-T}$. The task may acquire a free resource only when

$$
p_{\mathrm{eff}}(T) > C_{-T}.
$$

If no other task holds a resource, the ceiling check passes. Excluding $T$'s own resources allows nested locking. A failed check blocks $T$ even if its requested resource is free; the owner responsible for the blocking ceiling inherits $T$'s priority. [Real-time scheduling text, §4.6](https://www.es.mdu.se/pdf_publications/166.pdf)

For example, $T_L$ holds $R_A$, whose ceiling equals the base priority of $T_M$. Suppose $T_M$ has inherited no higher priority. If it requests another free lock, the strict check fails: its effective priority equals the system ceiling. $T_L$ inherits $T_M$'s priority and can release $R_A$ before a circular wait forms. [Original protocol, §IV](https://www.cse.iitb.ac.in/~cs431/papers/pi-pcp.pdf)

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/PCP.webp]]

Under the stated model, with all shared resources following PCP and their users known in advance, PCP prevents lock deadlock. Each job's lower-priority blocking is bounded by the longest lower-priority critical section whose resource ceiling reaches that job's base priority. Higher-priority preemption still contributes to response time, so meeting deadlines requires schedulability analysis. [Real-time scheduling text, §4.6](https://www.es.mdu.se/pdf_publications/166.pdf)

An **immediate ceiling** protocol instead raises priority as soon as a resource is locked. Keep that rule separate from the original PCP described here. [POSIX priority-protection semantics](https://pubs.opengroup.org/onlinepubs/7908799/xsh/pthread_mutexattr_setprotocol.html)
