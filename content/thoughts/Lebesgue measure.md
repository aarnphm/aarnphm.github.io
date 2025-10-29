---
created: "2025-10-29"
date: "2025-10-29"
description: Definition, properties, and examples of Lebesgue measure on $R^n$.
id: Lebesgue measure
modified: 2025-10-29 07:24:24 GMT-04:00
published: "2001-09-30"
source: https://en.wikipedia.org/wiki/Lebesgue_measure
tags:
  - math
  - clippings
title: Lebesgue measure
---

> Concept of area/volume in any dimension. In measure theory, the Lebesgue measure is the standard measure on $\mathbb{R}^n$.
>
> For $n=1,2,3$ it coincides with length, area, volume, respectively; also called $n$-dimensional volume, $n$-volume, hypervolume.

Used throughout real analysis, especially for Lebesgue integration. If $A$ is Lebesgue-measurable, its measure is denoted $\boxed{\lambda(A)}$.

Lebesgue described this measure in 1901 and the Lebesgue integral in 1902 (dissertation: "Intégrale, Longueur, Aire"). [@lebesgue1902integrale]

Lebesgue measure extends the classical notion of volume to every dimension while retaining the usual length, area, and volume interpretations in low dimensions. [@eomVolume]

## Definition

1. One dimension. For an interval $I=[a,b]$ or $I=(a,b)$ in $\mathbb{R}$, define its length $\ell(I)=b-a$. For $E\subseteq\mathbb{R}$, the Lebesgue outer measure is

$$
\lambda^{*}(E)=\inf\Big\{\sum_{k=1}^{\infty}\ell(I_k):\ (I_k)_{k\in\mathbb{N}}\text{ are open intervals with }E\subset\bigcup_{k=1}^{\infty}I_k\Big\}.
$$

2. Higher dimensions. For a rectangular cuboid (product of open intervals) $C=I_1\times\cdots\times I_n$, define

$$
\operatorname{vol}(C)=\ell(I_1)\cdots\ell(I_n).
$$

For $E\subseteq\mathbb{R}^n$, define

$$
\lambda^{*}(E)=\inf\Big\{\sum_{k=1}^{\infty}\operatorname{vol}(C_k):\ (C_k)_{k\in\mathbb{N}}\text{ are products of open intervals with }E\subset\bigcup_{k=1}^{\infty}C_k\Big\}.
$$

3. Carathéodory criterion. A set $E\subseteq\mathbb{R}^n$ is Lebesgue-measurable if for all $A\subseteq\mathbb{R}^n$,

$$
\lambda^{*}(A)=\lambda^{*}(A\cap E)+\lambda^{*}(A\cap E^{\complement}).
$$

Lebesgue-measurable sets form a $\sigma$-algebra. For such $E$, define $\lambda(E)=\lambda^{*}(E)$.

These constructions follow the standard presentation via outer measure and Carathéodory's criterion in graduate real analysis texts.[@royden1988realanalysis]

Non-measurable sets exist in {{sidenotes[ZFC]: In set theory, Zermelo–Fraenkel set theory, is an axiomatic system that was proposed in the early twentieth century in order to formulate a theory of sets free of paradoxes such as [[thoughts/Wittgenstein#Bertrand Russell|Russell]]'s paradox}} (e.g., Vitali sets).

### intuition

- Outer measure: cover $E$ by countably many open intervals (or $n$-rectangles) and minimize total length/volume. The infimum captures “tight” covers.
- Measurability: $E$ is “compatible” with the outer measure in the sense of Carathéodory: clipping any set $A$ by $E$ and its complement splits outer measure additively.

These heuristics match textbook discussions that stress coverings, regularity, and Carathéodory measurability. [@royden1988realanalysis]

## examples

- Any closed interval $[a,b]$ is measurable with $\lambda([a,b])=b-a$. The open interval $(a,b)$ has the same measure (endpoints have measure zero).
- Any Cartesian product $[a,b]\times[c,d]$ is measurable with measure $(b-a)(c-d)$ (area of the rectangle).
- Every Borel set is Lebesgue-measurable; there exist Lebesgue-measurable sets that are not Borel.
- Every countable subset of $\mathbb{R}$ has Lebesgue measure 0 (e.g., algebraic numbers are dense yet null).
- The Cantor set and the set of Liouville numbers are uncountable null sets.
- Under the axiom of determinacy, all sets of reals are Lebesgue-measurable (incompatible with the axiom of choice).
- Vitali sets are non-measurable (existence uses the axiom of choice).
- Osgood curves: simple plane curves with positive Lebesgue measure (cf. Peano/dragon curves). [@osgood1903jordancurve]
- Any line in $\mathbb{R}^n$ for $n\ge 2$ has measure 0; more generally, every proper hyperplane is null in its ambient space.
- The volume of an $n$-ball is expressible via Euler’s gamma function.

## properties

- Translation invariance: $\lambda(A)=\lambda(A+t)$ for any $t\in\mathbb{R}^n$.

1. If $A=I_1\times\cdots\times I_n$ (intervals), then $A$ is measurable and:
   $$
   \lambda(A)=|I_1|\cdot|I_2|\cdots|I_n|
   $$
2. If $A=\bigsqcup_{k=1}^{\infty}A_k$ with pairwise disjoint measurable $A_k$, then $A$ is measurable and:
   $$
   \lambda(A)=\sum_{k=1}^{\infty}\lambda(A_k)
   $$
3. If $A$ is measurable, then $A^{\complement}$ is measurable.
4. $\lambda(A)\ge 0$ for every measurable $A$.
5. If $A\subseteq B$ are measurable, then $\lambda(A)\le\lambda(B)$.
6. Countable unions and intersections of measurable sets are measurable.
7. Every open, closed, or Borel subset of $\mathbb{R}^n$ is measurable.
8. Measurable sets are “approximately open/closed” in the sense of Lebesgue measure.
9. Regularity: For every $\varepsilon>0$ and measurable $E\subset\mathbb{R}$, there exist closed $F\subset E\subset G$ open with $\lambda(G\setminus F)<\varepsilon$.
10. There exist a $G_\delta$ set $G$ and an $F_\sigma$ set $F$ with $F\subseteq A\subseteq G$ and $\lambda(G\setminus A)=\lambda(A\setminus F)=0$.
11. Lebesgue measure is a Radon measure (locally finite, inner regular).
12. Strictly positive on nonempty open sets; support is all of $\mathbb{R}^n$.
13. If $\lambda(A)=0$ and $B\subseteq A$, then $\lambda(B)=0$ (in particular, $B$ is measurable).
14. For any $x\in\mathbb{R}^n$, the translation $A+x=\{a+x:a\in A\}$ is measurable with $\lambda(A+x)=\lambda(A)$.
15. For any $\delta>0$, the dilation $\delta A=\{\delta x:x\in A\}$ is measurable with $\lambda(\delta A)=\delta^n\,\lambda(A)$.
16. More generally, for a linear map $T$ and measurable $A\subset\mathbb{R}^n$, $T(A)$ is measurable with:
    $$
    \lambda(T(A))=|\det T|\,\lambda(A)
    $$

> [!summary]
>
> Lebesgue-measurable sets form a $\sigma$-algebra containing all products of intervals, and $\lambda$ is the unique complete, translation-invariant measure on it with $\lambda([0,1]^n)=1$. It is $\sigma$-finite. [@carothers2000realanalysis]

## null sets

A subset of $\mathbb{R}^n$ is null if for every $\varepsilon>0$ it can be covered by countably many products of $n$ intervals with total volume at most $\varepsilon$. All countable sets are null.

If a set has Hausdorff dimension $<n$ (with respect to the Euclidean metric), then it is null for $n$-dimensional Lebesgue measure. Conversely, a set may have topological dimension $<n$ and positive $n$-dimensional Lebesgue measure (e.g., the Smith–Volterra–Cantor set has topological dimension 0 but positive 1D Lebesgue measure).

To show that $A$ is measurable, one often finds a “nicer” set $B$ with symmetric difference $(A\setminus B)\cup(B\setminus A)$ null, and then builds $B$ from open/closed sets via countable unions/intersections.

Because Lebesgue measure is complete, adjoining all subsets of null sets strictly enlarges the Borel $\sigma$-algebra, and numerous non-Borel sets become measurable.[@karagila2013lebesguemeasurable] There are also many intermediate $\sigma$-algebras between the Borel and Lebesgue completions, obtained by adjoining carefully chosen families of sets [@karagila2012sigmaalgebra].

## construction of the lebesgue measure

Using Carathéodory’s extension theorem.

- Fix $n\in\mathbb{N}$. A box in $\mathbb{R}^n$ is $B=\prod_{i=1}^{n}[a_i,b_i]$ with volume
  $$
  \operatorname{vol}(B)=\prod_{i=1}^{n}(b_i-a_i)
  $$
- For any $A\subseteq\mathbb{R}^n$, define the outer measure
  $$
  \lambda^{*}(A)=\inf\Big\{\sum_{B\in\mathcal{C}}\operatorname{vol}(B):\ \mathcal{C}\text{ is a countable collection of boxes covering }A\Big\}
  $$
- A set $A\subseteq\mathbb{R}^n$ is Lebesgue-measurable if for all $S\subseteq\mathbb{R}^n$:
  $$
  \lambda^{*}(S)=\lambda^{*}(S\cap A)+\lambda^{*}(S\setminus A)
  $$
- Then define $\lambda(A)=\lambda^{*}(A)$ for measurable $A$.

Existence of non-measurable sets follows from the axiom of choice (e.g., Vitali). [@solovay1970model] showed that without choice, non-measurable sets are not provable in ZF (Solovay’s model).

This Carathéodory-style construction and its regularity refinements align with modern treatments of measure theory.[@carothers2000realanalysis; @royden1988realanalysis]

## relation to other measures

- Borel measure agrees with Lebesgue measure where both are defined, but there are strictly more Lebesgue-measurable sets; Borel measure is translation-invariant but not complete.
- Haar measure generalizes Lebesgue measure to locally compact groups ($\mathbb{R}^n$ under addition is one).
- Hausdorff measure generalizes to lower-dimensional subsets (submanifolds, fractals); distinct from Hausdorff dimension.
- There is no infinite-dimensional analogue of Lebesgue measure.

## See also

- 4-volume
- Edison Farah
- Lebesgue's density theorem
- Lebesgue measure of the set of Liouville numbers
- Non-measurable set (Vitali set)
- Peano–Jordan measure
