---
author:
  - "[[Contributors to Wikimedia projects]]"
created: "2025-12-01"
date: "2025-12-01"
description: "Arrow’s theorem: no ranked rule satisfies Pareto, IIA, and non-dictatorship."
external: https://en.wikipedia.org/wiki/Arrow%27s_impossibility_theorem
id: Arrow's impossibility theorem
modified: 2025-12-01 22:09:09 GMT-05:00
published: "2002-09-22"
seealso:
  - Comparison of electoral systems
  - Condorcet paradox
  - Doctrinal paradox
  - Gibbard–Satterthwaite theorem; Gibbard's theorem
  - May's theorem
  - Market failure
tags:
  - seed
  - clippings
  - democracy
title: Arrow's impossibility theorem
---

Arrow's impossibility theorem is a key result in social choice theory showing that no ranked-choice procedure for group decision-making can satisfy the requirements of rational choice. Specifically, Arrow showed no such rule can satisfy independence of irrelevant alternatives (IIA), the principle that a choice between two alternatives $A$ and $B$ should not depend on the quality of some third, unrelated option, $C$.

The result is often cited in discussions of voting rules, where it shows no ranked voting rule can eliminate the spoiler effect. This result was foreshadowed by the Marquis de Condorcet's voting paradox (majority-rule intransitivity); Arrow's theorem generalizes Condorcet's findings to include non-majoritarian rules.

While the impossibility theorem shows all ranked voting rules must have spoilers, the frequency of spoilers differs by rule. Plurality-rule methods (e.g., choose-one and ranked-choice/IRV) are highly sensitive to spoilers, creating them even when not mathematically necessary (e.g., center squeeze). In contrast, majority-rule (Condorcet) methods uniquely minimize spoiled elections by restricting them to voting cycles, which are rare in ideologically-driven elections. Under some models of voter preferences (e.g., single-peaked on a left–right spectrum), spoilers disappear for these methods.

Rated voting rules (cardinal methods), where voters assign separate grades to each candidate, are not affected by Arrow's theorem. Arrow later acknowledged rules based on cardinal utilities (e.g., score and approval voting) are not subject to his theorem.

---

## Background

Main articles: Social welfare function; Voting systems; Social choice theory

When Kenneth Arrow proved his theorem in 1950, it inaugurated modern social choice theory (a branch of welfare economics) studying mechanisms to aggregate preferences and beliefs across a society (markets, voting systems, constitutions, moral/ethical frameworks).

### Axioms of voting systems

#### Preferences

In Arrow's framework, citizens have ordinal preferences (rankings). If $A$ and $B$ are different candidates, then $A \succ B$ means $A$ is preferred to $B$. Individual preferences (ballots) must be transitive: if $A \succeq B$ and $B \succeq C$, then $A \succeq C$. A social choice function maps the individual orderings to a new ordering representing society's preferences.

#### Basic assumptions

Arrow assumes any non-degenerate social choice rule will satisfy:

- Unrestricted domain — the social choice function is a total function over the domain of all possible orderings of outcomes.
  - The system must always make some choice; it cannot "give up" for unusual opinions.
  - Without this, majority rule can satisfy Arrow's axioms by refusing to decide in cycles.
- Non-dictatorship — the system does not depend on only one voter's ballot.
  - This weakens anonymity (one vote, one value) to allow unequal treatment of voters.
  - It defines social choices as depending on more than one person.
- Non-imposition (surjectivity on pairwise outcomes) — the system does not ignore voters entirely when choosing between some pairs.
  - It is possible for any candidate to defeat any other given some combination of votes.
  - Often replaced by Pareto efficiency; the weaker non-imposition suffices.

Arrow originally included positive responsiveness (monotonicity) but later removed it; it is not needed for the theorem (other than to imply Pareto efficiency).

#### Independence (IIA)

- Independence of irrelevant alternatives (IIA) — the social preference between $A$ and $B$ depends only on the individual preferences between $A$ and $B$.
  - The social preference should not switch from $A \succ B$ to $B \succ A$ due solely to changes in how voters rank $A$ or $B$ relative to $C$.
  - Under the standard construction of a placement function, this is equivalent to independence from spoilers.

A classic illustration (Morgenbesser): offered blueberry or apple, one picks apple; when cherry is added, one switches to blueberry. Arrow's theorem shows that to always avoid such contradictions using only rankings is impossible.

---

## Theorem

### Intuitive argument

Condorcet's example shows an impossibility of a fair ranked system under stronger fairness conditions. Suppose three candidates $A, B, C$ and three voters:

| Voter | 1st | 2nd | 3rd |
| ----- | --- | --- | --- |
| 1     | $A$ | $B$ | $C$ |
| 2     | $B$ | $C$ | $A$ |
| 3     | $C$ | $A$ | $B$ |

- Two voters (1,2) prefer $B$ to $C$, so one might argue $B$ should win over $C$.
- Similarly, two voters (2,3) prefer $C$ to $A$; two voters (1,3) prefer $A$ to $B$.
- Society's preferences cycle: $A \succ B$, $B \succ C$, $C \succ A$ (intransitive), though individual preferences are transitive.

Arrow's theorem is more general; it applies beyond one-person-one-vote elections (e.g., markets, weighted voting) if preferences are ranked.

### Formal statement

Let $A$ be a set of alternatives. A voter's preferences over $A$ are a complete and transitive binary relation $R \subseteq A \times A$ (a total preorder). Interpret $(\mathbf a, \mathbf b) \in R$ as $\mathbf a \succeq \mathbf b$. Define:

- Indifference (symmetric part): $\mathbf a \sim \mathbf b$ iff $(\mathbf a, \mathbf b) \in R$ and $(\mathbf b, \mathbf a) \in R$.
- Strict preference (asymmetric part): $\mathbf a \succ \mathbf b$ iff $(\mathbf a, \mathbf b) \in R$ and $(\mathbf b, \mathbf a)
otin R$.

Let $\Pi(A)$ be the set of all preferences (rankings, ties allowed) on $A$. For a positive integer $N$, an ordinal social welfare function (SWF) is
\[
\mathrm F : \Pi(A)^N \to \Pi(A),
\]
which aggregates voters' preferences into a single social preference. A profile $(R_1,\dots,R_N) \in \Pi(A)^N$ is a preference profile.

Arrow's impossibility theorem (for $|A| \ge 3$): There is no SWF satisfying all:

- Pareto efficiency: If for all voters $R_i$ one has $\mathbf a \succ \mathbf b$, then $\mathbf a \succ \mathbf b$ in $\mathrm F(R_1,\dots,R_N)$.
- Non-dictatorship: There is no $i \in \{1,\dots,N\}$ such that for all profiles and all $\mathbf a, \mathbf b$, whenever $\mathbf a \succ_i \mathbf b$, then $\mathbf a \succ \mathbf b$ in $\mathrm F(R_1,\dots,R_N)$.
- IIA: For two profiles $(R_1,\dots,R_N)$ and $(S_1,\dots,S_N)$, if for all $i$ the relative order of $\mathbf a, \mathbf b$ is the same in $R_i$ and $S_i$, then the social order of $\mathbf a, \mathbf b$ is the same in $\mathrm F(R_1,\dots,R_N)$ and $\mathrm F(S_1,\dots,S_N)$.

### Formal proof

Two standard proofs are sketched.

#### Proof by decisive coalitions

Definitions:

- A coalition is a subset of voters.
- A coalition is decisive over $(x,y)$ if whenever everyone in it ranks $x \succ_i y$, society has $x \succ y$.
- A coalition is decisive if it is decisive over all ordered pairs.

Goal: show some decisive coalition is of size 1 (a dictator).

Auxiliary notion: A coalition is weakly decisive over $(x,y)$ if whenever all inside rank $x \succ_i y$ and all outside rank $y \succ_j x$, then $x \succ y$ socially.

Assume unrestricted domain, Pareto, IIA, and at least 3 outcomes.

- Field expansion lemma: If a coalition $G$ is weakly decisive over some $(x,y)$ with $x
e y$, then it is decisive. Sketch: for any $z
otin \{x,y\}$, by constructing profiles and using IIA and Pareto to obtain $x \succ y \succ z$, infer $x \succ z$; iterate to all pairs.
- Group contraction lemma: If a decisive coalition has size $\ge 2$, it has a decisive proper subset. Sketch: partition decisive $G$ into nonempty $G_1,G_2$, design a cyclic profile on distinct $x,y,z$; deduce either $G_1$ or $G_2$ is weakly decisive over some pair, then apply field expansion.

By Pareto, the entire electorate is decisive; repeated contraction yields a size-1 decisive coalition (a dictator).

#### Proof by pivotal voter

Setup: $n$ voters labeled $1,\dots,n$; candidates $A,B,C$ (by IIA, adding more does not affect the argument). Any SWF respecting unanimity and IIA is a dictatorship.

1. Existence of a pivotal voter for $A$ vs $B$:

- Consider profile 0 where all prefer $A \succ B$ and $C \succ B$; by unanimity, society has $A \succ B$ and $C \succ B$.
- Successively move $B$ to the top for voters $1,2,\dots$. There exists $k$ at which $B$ first becomes socially above $A$. Voter $k$ is pivotal for $B$ over $A$. By IIA, this $k$ is robust to irrelevant changes.

2. Pivotal for $B$ over $A$ implies dictator for $B$ over $C$:

- Partition voters into segment one $(1,\dots,k-1)$, pivotal $k$, segment two $(k+1,\dots,n)$.
- Construct profiles so that (by part 1 and unanimity) $A \succ B \succ C$ socially.
- Move pivotal's ballot to put $B$ above $A$; allow others to move $B$ below $C$ without changing $A$'s position. By IIA and part 1, society ranks $B \succ A$ and $A \succ C$, hence $B \succ C$ even if only pivotal ranks $B \succ C$. Thus $k$ dictates $B$ vs $C$.

3. There exists a dictator:

- Compare positions of pivotal voters across pairs; from (2), the pivotal for $B$ over $C$ must be at or before $k$, and for $C$ over $B$ at or after $k$.
- Symmetry implies all such pivotal positions coincide: $k_{B/C} = k_{B/A} = k_{C/B}$. Hence the same voter dictates all pairwise contests.

### Stronger versions

Arrow's theorem holds if Pareto is weakened to non-imposition: for any $a,b$, there exists a profile under which $a \succ b$ socially.

---

## Interpretation and practical solutions

Arrow's theorem says no ranked rule can always satisfy IIA; it does not quantify failure frequency. Arrow noted: many systems do not work badly all the time; all can work badly at times.

Approaches: (i) accept Arrow and seek least spoiler-prone ranked methods; (ii) relax assumptions (e.g., use rated voting).

### Minimizing IIA failures: majority-rule (Condorcet) methods

Condorcet methods limit spoilers to Condorcet cycles; they uniquely minimize spoiler effects among ranked rules. Under domain restrictions where Arrow’s axioms are satisfiable, Condorcet adheres whenever any rule can. Unlike pluralitarian rules (e.g., IRV, first-preference plurality), Condorcet avoids spoilers when a majority winner exists. Empirical and spatial models suggest cycles are rare.

#### Left–right spectrum (single-peakedness)

Black’s median voter theorem: with single-peaked preferences on a line, Arrow’s conditions are compatible and satisfied by any Condorcet-consistent rule; cycles do not occur.

In higher dimensions, cycles may appear (McKelvey–Schofield), but with rotational symmetry or a unique omnidirectional median, a Condorcet winner can exist. In realistic low-dimensional electorates with roughly normal opinion distributions, cycles are infrequent.

#### Generalized stability theorems

- Campbell–Kelly: Condorcet variants are maximally spoiler-resistant among ranked methods (can prevent spoilers whenever any ranked rule can, and never create new ones).
- Kalai–Muller: full characterization of domains admitting nondictatorial and strategyproof SWFs corresponds to settings with Condorcet winners.
- Holliday–Pacuit: propose a rule minimizing the number of potential spoilers (occasionally fails monotonicity, but less than IRV).

### Rated social choice (cardinal methods)

Arrow’s proof relies on ranked information and does not apply to rated systems (e.g., score voting, majority judgment). Utility representation theorems (VNM, Harsanyi) support cardinal coherence; however, Gibbard’s theorem still implies no voting game is universally strategyproof.

#### Meaningfulness of cardinal information

Arrow originally rejected interpersonal cardinal utilities but later acknowledged their usefulness and inapplicability of his theorem to them. Sen similarly evolved toward allowing partial comparability. Others note interpersonal comparisons are implicit in any non-dictatorial procedure; cardinal methods make them explicit. Psychometrics finds ratings (e.g., Likert scales) provide more valid/reliable information than rankings for many judgments.

#### Nonstandard spoilers

Behavioral economics documents IIA violations (e.g., decoy effects), suggesting psychological spoilers may occur irrespective of the voting rule. Ballot design and instructions (e.g., verbal grades, evaluative framing) can mitigate such effects.

### Esoteric solutions

#### Supermajority rules

Requiring supermajorities can avoid paradoxes at the cost of indecisiveness. Thresholds such as $2/3$ for 3 outcomes, $3/4$ for 4, etc., eliminate cycles. In $n$-dimensional spatial models with quasiconcave voter distributions, a $1 - e^{-1} \approx 0.632$ threshold suffices to prevent cycles; common $2/3$ rules often suffice in practice.

#### Infinite populations

With uncountably infinite voters (and the axiom of choice), Arrow’s axioms can be satisfied, but only by effectively disenfranchising almost all voters (measure-zero electorates), an "invisible dictatorship."

## Common misconceptions

- Arrow’s theorem does not involve strategic voting (though it implies results like Gibbard’s theorem). Preferences are taken as given; the issue is aggregation.
- Monotonicity (positive association) is not a condition of Arrow’s theorem; Arrow’s initial inclusion was an error and later corrected.
- Arrow’s theorem applies to ranked-choice systems, not to voting systems as a whole; cardinal systems are outside its scope.
