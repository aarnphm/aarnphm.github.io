---
date: '2024-12-13'
description: Set and bag semantics, joins, projection, and aggregation in relational algebra.
id: Relational Algebra
modified: 2026-09-28 09:11:47 GMT-04:00
tags:
  - sfwr3db3
title: Relational Algebra
---

Classical relational algebra operates on **sets** of tuples, so each tuple appears at most once. SQL usually preserves duplicates, which requires the [[thoughts/bags|bag]] version of these operators. The distinction changes projection, joins, and set operations. Unless a bag is specified below, use set semantics.

| Operator    | Operation              | Example                         |
| ----------- | ---------------------- | ------------------------------- |
| $\sigma_C$  | Selection              | $\sigma_{A=10}(R)$              |
| $\pi_L$     | Projection             | $\pi_{A,B}(R)$                  |
| $\times$    | Cross-Product          | $R_1 \times R_2$                |
| $\bowtie$   | Natural Join           | $R_1 \bowtie R_2$               |
| $\bowtie_C$ | Theta Join             | $R_1 \bowtie_{R_1.A=R_2.A} R_2$ |
| $\rho_R$    | Rename                 | $\rho_S(R)$                     |
| $\delta$    | Eliminate Duplicates   | $\delta(R)$                     |
| $\tau$      | Sort Tuples            | $\tau(R)$                       |
| $\gamma_L$  | Grouping & Aggregation | $\gamma_{A,AVG(B)}(R)$          |

## selection

Selection keeps the rows that satisfy a condition.

$$
R_{1} \coloneqq \sigma_C(R_{2})
$$

$C$ is a predicate over the attributes of $R_2$. Selection leaves the attributes unchanged. For a bag, it keeps every occurrence of a matching tuple.

## projection

Projection keeps selected attributes. Under set semantics, tuples that become identical after projection collapse into one tuple.

$$
R_{1} \coloneqq  \pi_L(R_{2})
$$

$L$ lists the attributes to keep. For example, projecting $\{(1,2),(1,3)\}$ onto its first attribute gives $\{(1)\}$. Bag projection gives two occurrences of $(1)$ because it processes both input tuples.

Extended projection also allows expressions and renamed output attributes:

$$
\begin{aligned}
R &=
\begin{bmatrix}
A & B \\
1 & 2 \\
3 & 4
\end{bmatrix} \\[8pt]

\pi_{A+B \rightarrow C, A \rightarrow A_1, A \rightarrow A_2}(R) &=
\begin{bmatrix}
C & A_1 & A_2 \\
3 & 1 & 1 \\
7 & 3 & 3
\end{bmatrix}
\end{aligned}
$$

## products

$$
R_{3} \coloneqq  R_{1} \times R_{2}
$$

The product pairs every tuple of $R_1$ with every tuple of $R_2$. Rename overlapping attribute names so each output column is unambiguous. Its size is $|R_1||R_2|$, counting occurrences for bags.

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/products-relalg.webp]]

## theta-join

$$
R_{3} \coloneqq  R_{1} \bowtie_C R_{2}
$$

Apply a selection to the product:

$$
R_1 \bowtie_C R_2 = \sigma_C(R_1 \times R_2).
$$

The predicate can use comparisons such as $A=B$ or $A<B$. An equijoin is the equality-only case.

## natural join

$$
R_{3} \coloneqq  R_{1} \bowtie R_{2}
$$

- equating attributes of the same name
- projecting out one copy of each pair of equated attributes

If the schemas share no attribute names, natural join is the Cartesian product. With bags, each matching pair contributes the product of its input multiplicities; merging the shared columns does not deduplicate the result. See [relational operations on bags](https://csci3030u.science.ontariotechu.ca/chapter_5/chapter_5_section_1.html).

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/natural-join.webp]]

## renaming

$$
R_{1} \coloneqq  \rho_{R_{1}(A_{1},\ldots,A_n)}(R_{2})
$$

Rename the relation and its attributes without changing tuple values or multiplicities. This also makes two copies of the same relation usable in a self-join.

## set operators

> [!abstract] union compatible
>
> In the named-attribute notation used here, two relations are _union compatible_ when they have the same attribute names and matching domains.

`Student(sNumber, sName)` and `Course(cNumber, cName)` need corresponding attributes renamed before taking their union, even if the domains match. SQL instead matches columns by position and requires compatible types.

![[thoughts/bags]]

### Set Operations on Relations

Let $m$ and $n$ be the multiplicities of tuple $t$ in the two inputs. The set operators discard duplicate occurrences. The bag operators retain counts:

| Operation    | Symbol | Set result: multiplicity of $t$       | Bag result: multiplicity of $t$ |
| ------------ | ------ | ------------------------------------- | ------------------------------- |
| Union        | $\cup$ | $1$ if $m>0$ or $n>0$; otherwise $0$  | $m+n$                           |
| Intersection | $\cap$ | $1$ if $m>0$ and $n>0$; otherwise $0$ | $\min(m,n)$                     |
| Difference   | $-$    | $1$ if $m>0$ and $n=0$; otherwise $0$ | $\max(0,m-n)$                   |

Here bag union means **additive** union. In SQL, `UNION`, `INTERSECT`, and `EXCEPT` use the set column; their `ALL` variants use the bag column. Plain `SELECT` preserves duplicates, while `SELECT DISTINCT` removes them. These defaults are explicit in the [PostgreSQL SELECT reference](https://www.postgresql.org/docs/current/sql-select.html).

### sequence of assignments

The precedence convention used in these notes, from highest to lowest:

$$
\begin{aligned}
&\sigma \quad \pi \quad \rho \\[8pt]
& \times \quad \bowtie \\[9pt]
& \cap \\
&\cup \quad -
\end{aligned}
$$

Use parentheses or intermediate assignments when mixing operators; notation conventions can differ between texts.

### expression tree

![[thoughts/university/twenty-four-twenty-five/sfwr-3db3/expression-tree-relalg.webp]]

## extended algebra

$\delta$: eliminate duplicates from bags

$\tau$: sort tuples

$\gamma_{L}(R)$ grouping and aggregation

outer join: retain unmatched tuples from one or both inputs

### duplicate elimination

$$
\delta(R)
$$

Each distinct tuple has multiplicity $1$ in the result. Applied to a set, $\delta$ changes nothing.

### sorting

$$
\tau_L(R)
$$

$L$ specifies the sort keys and directions. Sorting produces an ordered sequence, so its result extends the unordered relation model.

Ascending is the default here; $\tau_{L,\text{DESC}}(R)$ requests descending order. Ties need further sort keys if their relative order matters.

### applying aggregation

For $\gamma_L(R)$, $L$ lists grouping attributes and aggregate expressions.

- Partition $R$ by the values of its grouping attributes.

- Compute each aggregate within its group.

- Return one tuple per group, containing the grouping values and aggregate results.

With no grouping attributes, aggregation treats the input as one group. Multiplicity affects results: the mean of bag $\{\!\{1,1,4\}\!\}$ is $2$; removing duplicates first gives a mean of $5/2$.

### outerjoin

Start with the matching tuples from a join. A left outer join also retains each unmatched left tuple, padding the right-side attributes with `NULL`. A right outer join retains unmatched right tuples; a full outer join retains unmatched tuples from both sides. The [extended-algebra definitions](https://csci3030u.science.ontariotechu.ca/chapter_5/chapter_5_section_2.html) distinguish these three cases.

## bag operations

Double braces below denote a bag, so repeated values count separately.

Set union is idempotent: $R \cup R=R$. Additive bag union doubles each multiplicity, so this equality fails for any nonempty bag.

bag union: $\{\!\{1,2,1\}\!\} \cup \{\!\{1,1,2,3,1\}\!\} = \{\!\{1,1,1,1,1,2,2,3\}\!\}$

bag intersection: $\{\!\{1,2,1,1\}\!\} \cap \{\!\{1,2,1,3\}\!\} = \{\!\{1,1,2\}\!\}$

bag difference: $\{\!\{1,2,1,1\}\!\} - \{\!\{1,2,3\}\!\} = \{\!\{1,1\}\!\}$
