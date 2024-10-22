---
id: midterm
tags:
  - sfwr4aa4
date: "2024-10-22"
modified: "2024-10-22"
title: rt_system items
---

correctness: $|C(t) - Cs(t)| < \epsilon$

**drift** is RoC of the clock value from perfect clock. Given clock has bounded drift $\rho$ then

$$
\mid \frac{dC(t)}{dt} -1 \mid < \rho
$$

Monotonicity: $\forall t_{2} > t_{1}: C(t_{2}) > C(t_{1})$

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/rt-sys-failure.png]]

## kernels

`syscall` in kernel: User space and Kernel Space are in different spaces

```mermaid
graph LR

A[procedure] --[parameters]--> B[TRAP]
B --> C[Kernel]
C --> B --> A
```

> [!important]
>
> a user process becomes kernel process when *execute syscall*

Scheduling ensures fairness, min response time, max throughput

|  | OS | RTOS |
| --------------- | --------------- | --------------- |
| philos | time-sharing | event-driven |
| requirements | high-throughput | schedulablity (meet all hard deadlines) |
| metrics | fast avg-response | ensureed worst-case response |
| overload | fairness | meet critical deadlines |

> Kernel programs can always preempt user-space programs

Kernel program example:

 ```c
 #include <linux/init.h>   /* Required by macros*/
 #include <linux/kernel.h> /*KERN_INFO needs it*/
 #include <linux/module.h>

 static char *my_string __initdata = "dummy";
 static int my_int __initdata = 4;

 /* Init function with user defined name*/
 static int __init hello_4_init(void) {
   printk(KERN_INFO "Hello %s world, number %d\n", my_string, my_int);
   return 0;
 }

 /* Exit function with user defined name*/
 static void __exit hello_4_exit(void) {
   printf(KERN_INFO "Goodbye cruel world 4\n");
 }

 /*Macros to be used after defining init and exit functions*/
 module_init(hello_4_init);
 module_exit(hello_4_exit)
 ```

## **preemption** && `syscall`

> The act of temporarily interrupting a currently scheduled task for higher priority tasks.

> NOTE: `make` doesn't recompile if DAG is not changed.

## process

- independent execution, logical unit of work scheduled by OS
- in virtual memory:
  - Stack: store local variables and function arguments
  - Heaps: dyn located (think of `malloc`, `calloc`)
  - BSS segment: uninit data
  - Data segment: init data (global & static variables)
  - text: RO region containing program instructions

|  | stack | heap |
| --------------- | --------------- | --------------- |
| creation | `Member m` | `Member*m = new Member()` |
| lifetime | function runs to completion | delete, free is called |
| grow | fixed | dyn added by OS |
| err | stack overflow | heap fragmentation |
| when | size of memory is known, data size is small | large scale dyn mem |

## `fork()`

- create a `child` process that is identical to its parents, return `0` to child process and pid
- add a lot of overhead as duplicated. **Data space is not shared**

> variables init b4 `fork()` will be duplicated in both parent and child.

```c
#include <stdio.h>

int main(int argc, char** argv) {
  int child = fork();
  int c = 0;
  if (child)
    c += 5;
  else {
    child = fork();
    c += 5;
    if (child) c += 5;
  }
  printf("%d ", c);
}
```

## threads

- program-wide resources: global data & instruction
- execution state of control stream
- shared address space for faster context switching

> - Needs synchronisation (global variables are shared between threads)
> - lack robustness (one thread can crash the whole program)

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/mem-layout-threaded.png]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/single-vs-multithreaded.png]]

```c
#include <pthread.h>

void *foo(void *args) {}
pthread_attr_t attr;
pthread_attr_init(attr);

pthread_t thread;
// pthread_create(&thread, &attr, function, arg);
```

To solve race condition, uses semaphores.

## polling and interrupt

- polling: reading memloc to receive update of an event
  - think of

    ```prolog
    while (true) {
      if (event) {
        process_data()
        event = 0;
      }
    }
    ```

- interrupt: receieve interrupt signal
  - think of

    ```prolog
    signal(SIGNAL, handler)
    void handler(int sig) {
      process_data()
    }

    int main() {
      while (1) { do_work() }
    }
    ```

|              | interrupt | polling |
| ------------ | --------- | ------- |
| speed        | fast      | slow    |
| efficiency   | good      | poor    |
| cpu-waste    | low       | high    |
| multitasking | yes       | yes     |
| complexity   | high      | low     |
| debug        | difficult | easy    |

## process priority

`nice`: change process priority

- 0-99: RT tasks
- 100-139: Users

> lower the NICE value, higher priority

```c
#include <sys/resource.h>
int getpriority(int which, id_t who);
int setpriority(int which, id_t who, int value);
```

set scheduling policy: `sched_setscheduler(pid, SCHED_FIFO | SCHED_RR | SCHED_DEADLINE, &param)`

## scheduling

1. Priority-based preemptive scheduling

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/pbps.png]]

Temporal parameters:

Let the following be the scheduling parameters:

| desc                 | var                   |
| -------------------- | --------------------- |
| # of tasks           | $n$                   |
| release/arrival-time | $r_{i,j}$             |
| absolute deadline    | $d_i$                 |
| relative deadline    | $D_i = r_{i,j} - d_i$ |
| execution time       | $e_i$                 |
| response time        | $R_i$                 |

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/abs-rel-deadline.png]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/resp-time-exec-time.png]]

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/resp-time-preempted-exec.png]]
*response time when execution is preempted*

> Period $p_i$ of a periodic task $T_i$ is **min length** of all time intervales between release times of consecutive tasks.

> Phase of a Task $\phi_i$ is the release time $r_{i,1}$ of a task $T_i$, or $\phi_i = r_{i,1}$

> *in phase* are first instances of several tasks that are released simultaneously

> [!important] Representation
>
> a periodic task $T_i$ can be represented by:
>
> - 4-tuple $(\phi_i, P_i, e_i, D_i)$
> - 3-tuple $(P_i, e_i, D_i)$, or $(0, P_i, e_i, D_i)$
> - 2-tuple $(P_i, e_i)$, or $(0, P_i, e_i, P_i)$

> [!important] Utilisation factor $u_i$
>
> for a task $T_i$ with execution time $e_i$ and period $p_i$ is given by
>
> $$
> u_i = \frac{e_i}{p_i}
> $$

For system with $n$ tasks overall system utilisation is $U = \sum_{i=1}^{n}{u_i}$

## cyclic executive

assume tasks are non-preemptive, jobs parameters with hard deadlines known.

- no race condition, no deadlock, just function call
- however, very brittle, number of frame $F$ can be large, release times of tasks must be fixed

### *hyperperiod*

> is the least common multiple (lcm) of the periods.

> [!important] maximum num of arriving jobs
>
> $$
> N = \sum_{i=1}^{n} \frac{H}{p_i}
> $$

**Frames**: each task must fit within a single frame with size $f$ => number of frames $F = \frac{H}{f}$

C1: A job must fit in a frame, or $f \geq \text{max} \space e_i \forall \space 1\leq i \leq n$ for all tasks

C2: hyperperiod has an integer number of frames, or $\frac{H}{f} = \text{integer}$

C3: $2f - \text{gcd}(P_i, f) \leq D_i$ per task.

### task slices

idea: if framesize constraint doesn't met, then "slice" into smaller sub-tasks

$T_3=(20, 5)$ becomes $T_{3_{1}}=(20,1)$ and $T_{3_{2}}=(20,3)$ and $T_{3_{3}}=(20, 1)$

### Flow Graph for hyper-period

- Denote all jobs in hyperperiod of $F$ frames as $J_{1} \cdots J_{F}$
- Vertices:
  - $k$ job vertices $J_{1},J_{2},\cdots,J_{k}$
  - $F$ frame vertices $x,y,\cdots,z$
- Edges:
  - $(\text{source}, J_i)$ with capacity $C_i=e_i$
    - Encode jobs' compute requirements
  - $(J_i, x)$ with capacity $f$ iff $J_i$ can be scheduled in frame $x$
    - encode periods and deadlines
    - edge connected job node and frame node if the following are met:
      1. job arrives **before** or at the starting time of the frame
      2. job's absolute deadline **larger** or equal to ending time of frame
  - $(f, \text{sink})$ with capacity $f$
    - encodes limited computational capacity in each frame

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/flow-graph-hyperperiod.png]]

## static priority assignment

For higher priority:

- shorter period tasks (rate monotonic RM)
- tasks with shorter relative deadlines (deadline monotonic DM)

### rate-monotonic

- running on uniprocessor, tasks are preemptive, no OS overhead for preemption

> task $T_i$ has higher priority than task $T_j$ if $p_i < p_j$

> [!important] schedulability test for RM (Test 1)
>
> Given $n$ periodic processes, independent and preemptable, $D_i \geq p_i$ for all processes,
> **periods of all tasks are *integer* multiples of each other**
>
> a sufficient condition for tasks to be scheduled on uniprocessor: $U = \sum_{i=1}^{n}\frac{e_i}{p_i} \leq 1$

> [!important] schedulability test for RM (Test 2)
>
> A *sufficient* but not necessary condition is $U \leq n \cdot (2^{\frac{1}{n}} - 1)$ for $n$ periodic tasks
>
> for $n \to \infty$, we have $U < \ln(2) \approx 0.693$

> [!important] schedulability test for RM (Test 3)
>
> Consider a set of task $(T_{1},T_{2},\cdots,T_i)$ with $p_{1}<p_{2}<\cdots<p_i$. **Assume all tasks are in phase**.
> to ensure that $T_1$ can be feasibly scheduled, that $e_1 \leq p_1$ ($T_1$ has highest priority given lowest period)
>
> Supposed $T_2$ finishes at $t$. Total number of isntances of task $T_1$ released over time interval $[0; t)$ is $\lceil \frac{t}{p_{1}} \rceil$
>
> Thus the following condition must be met for every instance of task $T_1$ released during tim interval $(0;t)$:
>
> $$
> t = \lceil \frac{t}{p_{1}} \rceil \space e_1 + e_2
> $$

idea: find $k$ such that time $t = k \times p_1 \geq k * e_1 + e_2$ and $k\times p_1 \leq p_2$ for task 2

> [!important] general solution for RM-schedulability
>
> The time demand function for task $i; 1 \leq i \leq n$:
>
> $$
> \begin{aligned}
> \omega_i(t) &= \sum_{k=1}^{i} \lceil \frac{t}{p_k} \rceil \space e_k \leq t \\
> \\
> &\because 0 \leq t \leq p_i
> \end{aligned}
> $$
>
> holds a time instant $t$ chosen as $t=k_j p_j, (j=1,\cdots,i)$ and $k_j = 1, \cdots, \lfloor \frac{p_i}{p_j} \rfloor$

### deadline-monotonic

- if every task has period equal to relative deadline, same as RM
- arbitrary deadlines then DM performs better than RM
- **RM always fails if DM fails**

## dynamic priority assignment

### earliest-deadline first (EDF)

*depends on closeness of absolute deadlines*

> [!important] EDF schedulability test 1
>
> set of $n$ periodic tasks, each whose relative deadline is equal to or greater than its period
> iff $\sum_{i=1}^{n}(\frac{e_i}{p_i}) \leq 1$

> [!important] EDF schedulability test 2
>
> relative deadlines are not equal to or greater than their periods
> $$
> \sum_{i=1}^{n}(\frac{e_i}{\text{min}(D_i, p_i)}) \leq 1
> $$

## Priority Inversion

**critical sections** to avoid **race condition**

> Higher priority task can be blocked by a lower priority task due to resource contention

shows how resource contention can delay completion of higher priority tasks

- access shared resources guarded by Mutex or semaphores
- access non-preemptive subsystems (storage, networks)

Resource Access Control

### mutex

serially reusable: a resource cannot be interrupted

> If a job wants to use $k_i$ units of resources $R_i$, it executes a lock $L(R_i; k_i)$, and unlocks $U(R_i; k_i)$ once it finished

### Non-preemptive Critical Section Protocol (NPCS)

idea: schedule all critical sections non-preemptively

**While a task holds a resource it executes at a priority higher than the priorities of all tasks**

**a higher priority task is blocked only when some lower priority job is in critical section**

pros:
- zk about resource requirements of tasks
cons:
- task can be blocked by a lower priority task for a long time even without resource conflict

### Priority Inheritance Protocol (PIP)

idea: increase the priorites only upon resource contention

avoid NPCS drawback

would still run into deadlock (think of RR task resource access)

### Priority Ceiling Protocol (PCP)

idea: extends PIP to prevent deadlocks
- assigned priorities are fixed
- resource requirements of all the tasks that will request a resource $R$ is known

`ceiling(R)`: highest priority. Each resource has fixed priority ceiling
