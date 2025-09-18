---
title: "tracing garbage collection"
source: "https://en.wikipedia.org/wiki/Tracing_garbage_collection"
author:
  - "[[Contributors to Wikimedia projects]]"
published: 2004-01-17
created: 2025-09-18
description: "Overview of tracing garbage collection algorithms, strategies, and performance"
tags:
  - "seed"
  - "clippings"
---

Form of computer memory management

In computer programming, tracing garbage collection is a form of automatic memory management that consists of determining which objects should be deallocated ("garbage collected") by tracing which objects are reachable by a chain of references from certain "root" objects, and considering the rest as "garbage" and collecting them [1]. Tracing is the most common type of garbage collection — so much so that "garbage collection" often refers to the tracing method, rather than others such as reference counting — and there are a large number of algorithms used in implementation.

## Reachability of an object

Informally, an object is reachable if it is referenced by at least one variable in the program, either directly or through references from other reachable objects. More precisely, objects can be reachable in only two ways:

1. A distinguished set of roots: objects that are assumed to be reachable. Typically, these include all the objects referenced from anywhere in the call stack (that is, all local variables and parameters in the functions currently being invoked), and any global variables.

2. Anything referenced from a reachable object is itself reachable; more formally, reachability is a transitive closure.

The reachability definition of "garbage" is not optimal, insofar as the last time a program uses an object could be long before that object falls out of the environment scope. A distinction is sometimes drawn between syntactic garbage, those objects the program cannot possibly reach, and semantic garbage, those objects the program will in fact never again use. For example:

```java
Object x = new Foo();
Object y = new Bar();
x = new Quux();
/* At this point, we know that the Foo object
 * originally assigned to x will never be
 * accessed: it is syntactic garbage.
 */

/* In the following block, y could be semantic garbage;
 * but we won't know until x.check_something() returns
 * some value -- if it returns at all.
 */
if (x.check_something()) {
    x.do_something(y);
}
System.exit(0);
```

The problem of precisely identifying semantic garbage can easily be shown to be partially decidable: a program that allocates an object $X$, runs an arbitrary input program $P$, and uses $X$ if and only if $P$ finishes would require a semantic garbage collector to solve the halting problem. Although conservative heuristic methods for semantic garbage detection remain an active research area, essentially all practical garbage collectors focus on syntactic garbage. [citation needed]

Another complication with this approach is that, in languages with both reference types and unboxed value types, the garbage collector needs to somehow be able to distinguish which variables on the stack or fields in an object are regular values and which are references: in memory, an integer and a reference might look alike. The garbage collector then needs to know whether to treat the element as a reference and follow it, or whether it is a primitive value. One common solution is the use of tagged pointers.

## Strong and weak references

The garbage collector can reclaim only objects that have no references pointing to them either directly or indirectly from the root set. However, some programs require weak references, which should be usable for as long as the object exists but should not prolong its lifetime. In discussions about weak references, ordinary references are sometimes called strong references. An object is eligible for garbage collection if there are no strong (i.e., ordinary) references to it, even though there still might be some weak references to it.

A weak reference is not merely just any pointer to the object that a garbage collector does not care about. The term is usually reserved for a properly managed category of special reference objects which are safe to use even after the object disappears because they lapse to a safe value (usually null). An unsafe reference that is not known to the garbage collector will simply remain dangling by continuing to refer to the address where the object previously resided. This is not a weak reference.

In some implementations, weak references are divided into subcategories. For example, the Java Virtual Machine provides three forms of weak references, namely soft references [2], phantom references [3], and regular weak references [4]. A softly referenced object is only eligible for reclamation if the garbage collector decides that the program is low on memory. Unlike a soft reference or a regular weak reference, a phantom reference does not provide access to the object that it references. Instead, a phantom reference is a mechanism that allows the garbage collector to notify the program when the referenced object has become phantom reachable. An object is phantom reachable if it still resides in memory and it is referenced by a phantom reference, but its finalizer has already executed. Similarly, Microsoft .NET provides two subcategories of weak references [5], namely long weak references (tracks resurrection) and short weak references.

## Weak collections

Data structures can also be devised which have weak tracking features. For instance, weak hash tables are useful. Like a regular hash table, a weak hash table maintains an association between pairs of objects, where each pair is understood to be a key and value. However, the hash table does not actually maintain a strong reference on these objects. Special behavior takes place when either the key or value or both become garbage: the hash table entry is spontaneously deleted. There exist further refinements such as hash tables which have only weak keys (value references are ordinary, strong references) or only weak values (key references are strong).

Weak hash tables are important for maintaining associations between objects, such that the objects engaged in the association can still become garbage if nothing in the program refers to them any longer (other than the associating hash table).

The use of a regular hash table for such a purpose could lead to a "logical memory leak": the accumulation of reachable data which the program does not need and will not use.

## Basic algorithm

Tracing collectors are so called because they trace through the working set of memory. These garbage collectors perform collection in cycles. It is common for cycles to be triggered when there is not enough free memory for the memory manager to satisfy an allocation request. But cycles can often be requested by the mutator directly or run on a time schedule. The original method involves a naïve mark-and-sweep in which the entire memory set is touched several times.

### Naïve mark-and-sweep

In the naive mark-and-sweep method, each object in memory has a flag (typically a single bit) reserved for garbage collection use only. This flag is always cleared, except during the collection cycle.

- Mark stage: do a tree traversal of the root set and mark each object reachable from roots as in-use (recursively).
- Sweep stage: scan memory and free any object not marked in-use; clear the in-use flags on live objects to prepare for the next cycle.

Disadvantages include stop-the-world pauses (the system is suspended during collection) and scanning the entire memory, which can be problematic in paged memory systems.

### Tri-color marking

Because of these performance problems, most modern tracing garbage collectors implement some variant of the tri-color marking abstraction. Three disjoint sets are maintained:

- White (condemned set): candidates for reclamation.
- Grey: reachable from roots but not yet scanned for references to white.
- Black: reachable from roots and known to have no references to white.

Algorithm sketch:

1. Pick an object $o$ from the grey set.
2. Move each white object referenced by $o$ to the grey set.
3. Move $o$ to the black set.
4. Repeat until the grey set is empty.

When the grey set is empty, black objects are live; white objects are garbage. The tri-color invariant — no black object points to a white object — ensures safety and enables freeing white objects once grey is empty. Variations may relax the invariant while preserving essential properties. The method can be performed on-the-fly by maintaining sets during allocation and mutation, reducing pauses and avoiding touching the entire working set each cycle.

## Implementation strategies

### Moving vs. non-moving

After determining the unreachable set, collectors may:

- Non-moving: release unreachable objects; leave others in place.
- Moving (compacting): copy some/all reachable objects into a new area, updating references.

Moving collectors offer advantages:

- Reclaim free space in bulk (no per-object free pass).
- Fast allocation (bump pointer) due to large contiguous free regions.
- Improved locality by placing related objects near each other.

A disadvantage is that unmanaged/native pointers become invalid if objects move; interop often requires copying objects outside the GC heap or pinning them to prevent movement [6].

### Copying vs. mark-and-sweep vs. mark-and-don't-sweep

- Semi-space (stop-and-copy): split heap into equally sized from-space and to-space. On collection, swap roles and copy reachable objects to to-space, scanning as you go. Simple but requires ~2× memory; Cheney's algorithm optimizes this.

- Mark-and-sweep: maintain per-object bits for colors (white/black) and possibly grey via lists/bits. Traverse to mark; then sweep memory to free white objects. Works with both moving and non-moving strategies; small per-object overhead (bits) or use tagged pointers.

- Mark-and-don't-sweep: all reachable objects are always black. Objects are marked black on allocation and stay black even if unreachable; white denotes free memory. When allocation finds no white memory, invert the sense of the black/white bit globally, then perform a full marking phase to re-blacken reachable objects. No sweep phase is needed. Space-efficient (one bit per allocated pointer) but can make it hard to return memory to the system since large regions may be (temporarily) marked used.

### Generational GC (ephemeral GC)

Empirical hypothesis: most recently created objects die young (infant mortality/generational hypothesis). Generational collectors divide objects by age and often collect only young generations, while maintaining knowledge of cross-generation references (via write barriers/remembered sets). This allows proving some objects unreachable with limited traversal, yielding faster cycles.

Many implementations maintain regions (eden, survivor spaces, old space). When a young region fills, trace from remembered-set roots and copy survivors; promote (tenure) survivors if needed. Occasional full collections handle fragmentation or when the hypothesis does not hold. Systems like Java and .NET use hybrids: frequent minor cycles on young generations, occasional major mark-and-sweep, and rare full copying passes.

### Stop-the-world vs. incremental vs. concurrent

- Stop-the-world: pause the program entirely to collect. Simple and fast per cycle, but can cause noticeable pauses ("embarrassing pause") [7].

- Incremental: split GC work into phases and interleave with program execution.

- Concurrent: run GC alongside the program, possibly with brief pauses (e.g., stack scanning). These approaches reduce pause times but often decrease overall throughput due to barriers and coordination.

Careful design is required to avoid interference between mutator and collector (e.g., coordinating allocations during a cycle).

### Precise vs. conservative and internal pointers

- Precise collectors (exact/accurate): correctly identify all references in objects.
- Conservative collectors: treat any bit pattern that "looks like" a pointer into an allocated object as a reference; may yield false positives (unreclaimed memory). False positives are less problematic on 64-bit systems. False negatives can occur if pointers are hidden (e.g., XOR linked lists). Precise collection feasibility depends on language type safety; languages like C often require conservative approaches.

Internal pointers (references to fields within objects) complicate liveness, since multiple addresses can refer to parts of the same object (e.g., due to multiple inheritance in C++). Such pointers must be scanned.

## Performance

Performance — latency and throughput — depends on implementation, workload, and environment. Naive implementations or very memory-constrained environments (e.g., embedded) can perform poorly; sophisticated ones with ample memory can perform excellently. [citation needed]

Throughput: tracing implies runtime overhead, but amortized cost can be very low; in some cases, GC can be faster than stack allocation [8]. Manual management incurs free-time overhead; reference counting adds inc/dec and zero-check overhead.

Latency:

- Stop-the-world collectors introduce pauses at arbitrary times and durations, unsuitable for hard real-time and often undesirable for interactive systems.
- Incremental collectors can provide hard real-time guarantees. Scheduling GC during idle periods on systems with free memory reduces impact.
- Manual management and reference counting can also cause long pauses when freeing large structures, though at deterministic code points.

Manual heap allocation overhead:

- Search for best/first-fit block of sufficient size
- Free list maintenance

Garbage collection overhead:

- Locate reachable objects
- Copy reachable objects (moving collectors)
- Read/write barriers (incremental/concurrent collectors)
- Best/first-fit search and free list maintenance (non-moving collectors)

Trade-offs vary: bump-pointer allocation can be best-case for moving GC; segregated free lists can be best-case for manual allocation but may cause external fragmentation and poorer cache behavior. Some GCed languages still use heap allocators behind the scenes, reducing theoretical advantages. In embedded systems, one may avoid GC and general-purpose heap management by preallocating pools and using lightweight schemes [9]. Write barrier overheads are more noticeable in imperative programs that frequently mutate pointers than in functional programs that build data immutably.

## Determinism

- Tracing GC is not deterministic in finalization timing. An object becoming eligible for GC will usually be cleaned up eventually, but there is no guarantee when (or if). This matters when objects manage non-memory resources (files, sockets, devices). Reference counting provides determinism for such cleanup.

- GC can nondeterministically impact execution time by introducing pauses unrelated to the program’s algorithm. Allocation may be fast or may trigger a lengthy GC. Under reference counting, decrements can trigger cascaded deallocations unpredictably.

## Real-time garbage collection

While GC is generally nondeterministic, it can be used in hard real-time systems. A real-time collector should guarantee that even in the worst case it will dedicate sufficient computational resources to mutator threads.

Constraints are work-based or time-based. A time-based constraint: within each time window of duration $T$, mutator threads should be allowed to run at least for $T_m$ time. For work-based analysis, MMU (minimal mutator utilization) is commonly used as a real-time constraint [10].

Examples include the Metronome algorithm (commercialized in IBM WebSphere Real Time) [11][12] and Staccato in IBM’s J9 JVM, which targets parallel/concurrent real-time compacting GC on multiprocessors [13]. Designing non-blocking concurrent GC on modern multicore systems is a key challenge; see Pizlo et al. [14].

## See also

- Dead-code elimination
- Mark–compact algorithm

## References

1. "A Unified Theory of Garbage Collection" (2019-11-08). https://www.cs.cornell.edu/courses/cs6120/2019fa/blog/unified-theory-gc/
2. "Class SoftReference<T>" (Java SE 7). http://docs.oracle.com/javase/7/docs/api/java/lang/ref/SoftReference.html
3. "Class PhantomReference<T>" (Java SE 7). http://docs.oracle.com/javase/7/docs/api/java/lang/ref/PhantomReference.html
4. "Class WeakReference<T>" (Java SE 7). http://docs.oracle.com/javase/7/docs/api/java/lang/ref/WeakReference.html
5. ".NET Weak References" (.NET 4.5). http://msdn.microsoft.com/en-us/library/ms404247.aspx
6. "Copying and Pinning" (Microsoft Docs). https://docs.microsoft.com/en-us/dotnet/framework/interop/copying-and-pinning
7. Steele, G. L. (1975). "Multiprocessing Compactifying Garbage Collection". CACM 18(9):495–508. https://doi.org/10.1145/361002.361005
8. Appel, A. W. (1987). "Garbage collection can be faster than stack allocation". IPL 25(4):275–279. https://doi.org/10.1016/0020-0190(87)90175-X
9. Hopwood, D. (2007). "Memory allocation in embedded systems" (cap-talk). https://web.archive.org/web/20150924001857/http://www.eros-os.org/pipermail/cap-talk/2007-January/006795.html
10. Cheng, P.; Blelloch, G. E. (2001). "A Parallel, Real-Time Garbage Collector". https://doi.org/10.1145/381694.378823
11. Bacon, D. F.; Cheng, P.; Rajan, V. T. (2003). "The Metronome: A Simpler Approach to Garbage Collection in Real-Time Systems". https://doi.org/10.1007/978-3-540-39962-9_52
12. Biron, B.; Sciampacone, R. (2007). "Real-time Java, Part 4: Real-time garbage collection". https://web.archive.org/web/20201109040826/http://www.ibm.com/developerworks/java/library/j-rtj4/index.html
13. McCloskey, B.; Bacon, D. F.; Cheng, P.; Grove, D. (2008). "Staccato: A Parallel and Concurrent Real-time Compacting Garbage Collector for Multiprocessors". https://dominoweb.draco.res.ibm.com/reports/rc24504.pdf
14. Pizlo, P.; Petrank, E.; Steensgaard, B. (2008). PLDI. https://doi.org/10.1145/1375581.1375587
