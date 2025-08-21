---
id: Inner product space
tags:
  - seed
  - clippings
  - math
author:
  - "[[Contributors to Wikimedia projects]]"
description: Vector space with generalized dot product operation that defines lengths, angles, and orthogonality, generalizing Euclidean spaces to infinite dimensions.
source: https://en.wikipedia.org/wiki/Inner_product_space
date: "2025-08-20"
created: "2025-08-20"
modified: 2025-08-20 16:51:13 GMT-04:00
published: "2001-09-29"
title: Inner product space
---

> Geometric interpretation of the angle between two vectors defined using an inner product

![Scalar product spaces, over any field, have "scalar products" that are symmetrical and linear in the first argument. Hermitian product spaces are restricted to the field of complex numbers and have "Hermitian products" that are conjugate-symmetrical and linear in the first argument. Inner product spaces may be defined over any field, having "inner products" that are linear in the first argument, conjugate-symmetrical, and positive-definite. Unlike inner products, scalar products and Hermitian products need not be positive-definite.
](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Product_Spaces_Drawing_%281%29.webp/500px-Product_Spaces_Drawing_%281%29.webp)

> [!abstract] definition
>
> an **inner product space** (or, rarely, a **[Hausdorff](https://en.wikipedia.org/wiki/Hausdorff_space) pre-Hilbert space** [^2] [^3]) is a [real vector space](https://en.wikipedia.org/wiki/Real_vector_space) or a [complex vector space](https://en.wikipedia.org/wiki/Complex_vector_space) with an [operation](<https://en.wikipedia.org/wiki/Operation_(mathematics)> "Operation (mathematics)") called an **inner product**. The inner product of two vectors in the space is a [scalar](<https://en.wikipedia.org/wiki/Scalar_(mathematics)>), often denoted with as ${\displaystyle \langle a,b\rangle }$ .
>
> Inner products allow formal definitions of intuitive geometric notions, such as lengths, [angles](https://en.wikipedia.org/wiki/Angle "Angle"), and [orthogonality](https://en.wikipedia.org/wiki/Orthogonality "Orthogonality") (zero inner product) of vectors.

Inner product spaces generalize [Euclidean vector spaces](https://en.wikipedia.org/wiki/Euclidean_vector_space "Euclidean vector space"), in which the inner product is the [dot product](https://en.wikipedia.org/wiki/Dot_product "Dot product") or _scalar product_ of [Cartesian coordinates](https://en.wikipedia.org/wiki/Cartesian_coordinates "Cartesian coordinates").

An inner product naturally induces an associated [[thoughts/norm|norm]]; so, every inner product space is a [normed vector space](https://en.wikipedia.org/wiki/Normed_vector_space).

- If this normed space is also [complete](https://en.wikipedia.org/wiki/Complete_metric_space "Complete metric space") (that is, a [Banach space](https://en.wikipedia.org/wiki/Banach_space "Banach space")) then the inner product space is a [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space "Hilbert space").[^2]
- If an inner product space $H$ is not a Hilbert space, it can be _extended_ by [completion](https://en.wikipedia.org/wiki/Complete_topological_vector_space#Completions) to a Hilbert space ${\displaystyle {\overline {H}}.}$ [^dense-subspace]

[^dense-subspace]: This means that ${\displaystyle H}$ is a [linear subspace](https://en.wikipedia.org/wiki/Linear_subspace "Linear subspace") of ${\displaystyle {\overline {H}},}$ the inner product of ${\displaystyle H}$ is the [restriction](<https://en.wikipedia.org/wiki/Restriction_(mathematics)> "Restriction (mathematics)") of that of ${\displaystyle {\overline {H}},}$ and ${\displaystyle H}$ is [dense](https://en.wikipedia.org/wiki/Dense_subset "Dense subset") in ${\displaystyle {\overline {H}}}$ for the [topology](<https://en.wikipedia.org/wiki/Topology_(structure)> "Topology (structure)") defined by the norm.[^2] [^5]

## Definition

$F$ denotes a [field](<https://en.wikipedia.org/wiki/Field_(mathematics)> "Field (mathematics)") that is either the [real numbers](https://en.wikipedia.org/wiki/Real_number "Real number") ${\displaystyle \mathbb {R} ,}$ or the [complex numbers](https://en.wikipedia.org/wiki/Complex_number "Complex number") ${\displaystyle \mathbb {C} .}$ A scalar is thus an element of $F$.

A bar over an expression representing a scalar denotes the [complex conjugate](https://en.wikipedia.org/wiki/Complex_conjugate "Complex conjugate") of this scalar. A zero vector is denoted ${\displaystyle \mathbf {0} }$ for distinguishing it from the scalar 0.

An _inner product_ space is a [[thoughts/Vector space]] $V$ over the field $F$ together with an _inner product_, that is, a map

$$
{\displaystyle \langle \cdot ,\cdot \rangle :V\times V\to F}
$$

that satisfies the following three properties for all vectors ${\displaystyle x,y,z\in V}$ and all scalars [^6]

- _Conjugate symmetry_:
  $$
  {\displaystyle \langle x,y\rangle ={\overline {\langle y,x\rangle }}.}
  $$
  As ${\textstyle a={\overline {a}}}$ [if and only if](https://en.wikipedia.org/wiki/If_and_only_if "If and only if") ${\displaystyle a}$ is real, conjugate symmetry implies that ${\displaystyle \langle x,x\rangle }$ is always a real number. If _F_ is ${\displaystyle \mathbb {R} }$ , conjugate symmetry is just symmetry.
- [Linearity](https://en.wikipedia.org/wiki/Linear_map "Linear map") in the first argument:[^1]
  $$
  {\displaystyle \langle ax+by,z\rangle =a\langle x,z\rangle +b\langle y,z\rangle .}
  $$
- [Positive-definiteness](https://en.wikipedia.org/wiki/Definite_bilinear_form "Definite bilinear form"): if ${\displaystyle x}$ is not zero, then
  $$
  {\displaystyle \langle x,x\rangle >0}
  $$
  (conjugate symmetry implies that ${\displaystyle \langle x,x\rangle }$ is real).

If the positive-definiteness condition is replaced by merely requiring that ${\displaystyle \langle x,x\rangle \geq 0}$ for all ${\displaystyle x}$ , then one obtains the definition of _positive semi-definite Hermitian form_. A positive semi-definite Hermitian form ${\displaystyle \langle \cdot ,\cdot \rangle }$ is an inner product if and only if for all ${\displaystyle x}$ , if ${\displaystyle \langle x,x\rangle =0}$ then ${\displaystyle x=\mathbf {0} }$ .[^8]

### Basic properties

In the following properties, which result almost immediately from the definition of an inner product, _x_, _y_ and z are arbitrary vectors, and a and b are arbitrary scalars.

- ${\displaystyle \langle \mathbf {0} ,x\rangle =\langle x,\mathbf {0} \rangle =0.}$
- ${\displaystyle \langle x,x\rangle }$ is real and nonnegative.
- ${\displaystyle \langle x,x\rangle =0}$ if and only if ${\displaystyle x=\mathbf {0} .}$
- ${\displaystyle \langle x,ay+bz\rangle ={\overline {a}}\langle x,y\rangle +{\overline {b}}\langle x,z\rangle .}$
  This implies that an inner product is a [sesquilinear form](https://en.wikipedia.org/wiki/Sesquilinear_form "Sesquilinear form").
- ${\displaystyle \langle x+y,x+y\rangle =\langle x,x\rangle +2\operatorname {Re} (\langle x,y\rangle )+\langle y,y\rangle ,}$ where ${\displaystyle \operatorname {Re} }$
  denotes the [real part](https://en.wikipedia.org/wiki/Real_part "Real part") of its argument.

Over ${\displaystyle \mathbb {R} }$ , conjugate-symmetry reduces to symmetry, and sesquilinearity reduces to bilinearity. Hence an inner product on a real vector space is a _positive-definite symmetric [bilinear form](https://en.wikipedia.org/wiki/Bilinear_form "Bilinear form")_. The [binomial expansion](https://en.wikipedia.org/wiki/Binomial_expansion "Binomial expansion") of a square becomes

$$
{\displaystyle \langle x+y,x+y\rangle =\langle x,x\rangle +2\langle x,y\rangle +\langle y,y\rangle .}
$$

### Notation

Several notations are used for inner products, including ${\displaystyle \langle \cdot ,\cdot \rangle }$ , ${\displaystyle \left(\cdot ,\cdot \right)}$ , ${\displaystyle \langle \cdot |\cdot \rangle }$ and ${\displaystyle \left(\cdot |\cdot \right)}$ , as well as the usual dot product.

### Convention variant

Some authors, especially in [physics](https://en.wikipedia.org/wiki/Physics "Physics") and [matrix algebra](https://en.wikipedia.org/wiki/Matrix_algebra "Matrix algebra"), prefer to define inner products and sesquilinear forms with linearity in the second argument rather than the first. Then the first argument becomes conjugate linear, rather than the second. [Bra-ket notation](https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation "Bra–ket notation") in [quantum mechanics](https://en.wikipedia.org/wiki/Quantum_mechanics "Quantum mechanics") also uses slightly different notation, i.e. ${\displaystyle \langle \cdot |\cdot \rangle }$ , where ${\displaystyle \langle x|y\rangle :=\left(y,x\right)}$ .

## Examples

Among the simplest examples of inner product spaces are ${\displaystyle \mathbb {R} }$ and ${\displaystyle \mathbb {C} .}$ The [real numbers](https://en.wikipedia.org/wiki/Real_number "Real number") ${\displaystyle \mathbb {R} }$ are a vector space over ${\displaystyle \mathbb {R} }$ that becomes an inner product space with arithmetic multiplication as its inner product:

$$
{\displaystyle \langle x,y\rangle :=xy\quad {\text{ for }}x,y\in \mathbb {R} .}
$$

The [complex numbers](https://en.wikipedia.org/wiki/Complex_number "Complex number") ${\displaystyle \mathbb {C} }$ are a vector space over ${\displaystyle \mathbb {C} }$ that becomes an inner product space with the inner product

$$
{\displaystyle \langle x,y\rangle :=x{\overline {y}}\quad {\text{ for }}x,y\in \mathbb {C} .}
$$

Unlike with the real numbers, the assignment ${\displaystyle (x,y)\mapsto xy}$ does _not_ define a complex inner product on ${\displaystyle \mathbb {C} .}$

More generally, the [real ${\displaystyle n}$ -space](https://en.wikipedia.org/wiki/Real_coordinate_space "Real coordinate space") ${\displaystyle \mathbb {R} ^{n}}$ with the [dot product](https://en.wikipedia.org/wiki/Dot_product "Dot product") is an inner product space, an example of a [Euclidean vector space](https://en.wikipedia.org/wiki/Euclidean_vector_space "Euclidean vector space").

$$
{\displaystyle \left\langle {\begin{bmatrix}x_{1}\\\vdots \\x_{n}\end{bmatrix}},{\begin{bmatrix}y_{1}\\\vdots \\y_{n}\end{bmatrix}}\right\rangle =x^{\textsf {T}}y=\sum _{i=1}^{n}x_{i}y_{i}=x_{1}y_{1}+\cdots +x_{n}y_{n},}
$$

where ${\displaystyle x^{\operatorname {T} }}$ is the [transpose](https://en.wikipedia.org/wiki/Transpose "Transpose") of ${\displaystyle x.}$

A function ${\displaystyle \langle \,\cdot ,\cdot \,\rangle :\mathbb {R} ^{n}\times \mathbb {R} ^{n}\to \mathbb {R} }$ is an inner product on ${\displaystyle \mathbb {R} ^{n}}$ if and only if there exists a [symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix "Symmetric matrix") [positive-definite matrix](https://en.wikipedia.org/wiki/Positive-definite_matrix "Positive-definite matrix") ${\displaystyle \mathbf {M} }$ such that ${\displaystyle \langle x,y\rangle =x^{\operatorname {T} }\mathbf {M} y}$ for all ${\displaystyle x,y\in \mathbb {R} ^{n}.}$ If ${\displaystyle \mathbf {M} }$ is the [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix "Identity matrix") then ${\displaystyle \langle x,y\rangle =x^{\operatorname {T} }\mathbf {M} y}$ is the dot product. For another example, if ${\displaystyle n=2}$ and ${\displaystyle \mathbf {M} ={\begin{bmatrix}a&b\\b&d\end{bmatrix}}}$ is positive-definite (which happens if and only if ${\displaystyle \det \mathbf {M} =ad-b^{2}>0}$ and one/both diagonal elements are positive) then for any ${\displaystyle x:=\left[x_{1},x_{2}\right]^{\operatorname {T} },y:=\left[y_{1},y_{2}\right]^{\operatorname {T} }\in \mathbb {R} ^{2},}$

$$
{\displaystyle \langle x,y\rangle :=x^{\operatorname {T} }\mathbf {M} y=\left[x_{1},x_{2}\right]{\begin{bmatrix}a&b\\b&d\end{bmatrix}}{\begin{bmatrix}y_{1}\\y_{2}\end{bmatrix}}=ax_{1}y_{1}+bx_{1}y_{2}+bx_{2}y_{1}+dx_{2}y_{2}.}
$$

As mentioned earlier, every inner product on ${\displaystyle \mathbb {R} ^{2}}$ is of this form (where ${\displaystyle b\in \mathbb {R} ,a>0}$ and ${\displaystyle d>0}$ satisfy ${\displaystyle ad>b^{2}}$ ).

The general form of an inner product on ${\displaystyle \mathbb {C} ^{n}}$ is known as the [Hermitian form](https://en.wikipedia.org/wiki/Hermitian_form "Hermitian form") and is given by

$$
{\displaystyle \langle x,y\rangle =y^{\dagger }\mathbf {M} x={\overline {x^{\dagger }\mathbf {M} y}},}
$$

where ${\displaystyle M}$ is any [Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix "Hermitian matrix") [positive-definite matrix](https://en.wikipedia.org/wiki/Positive-definite_matrix "Positive-definite matrix") and ${\displaystyle y^{\dagger }}$ is the [conjugate transpose](https://en.wikipedia.org/wiki/Conjugate_transpose "Conjugate transpose") of ${\displaystyle y.}$ For the real case, this corresponds to the dot product of the results of directionally-different [scaling](<https://en.wikipedia.org/wiki/Scaling_(geometry)> "Scaling (geometry)") of the two vectors, with positive [scale factors](https://en.wikipedia.org/wiki/Scale_factor "Scale factor") and orthogonal directions of scaling. It is a [weighted-sum](https://en.wikipedia.org/wiki/Weight_function "Weight function") version of the dot product with positive weights—up to an orthogonal transformation.

### Hilbert space

The article on [Hilbert spaces](https://en.wikipedia.org/wiki/Hilbert_spaces "Hilbert spaces") has several examples of inner product spaces, wherein the metric induced by the inner product yields a [complete metric space](https://en.wikipedia.org/wiki/Complete_metric_space "Complete metric space"). An example of an inner product space which induces an incomplete metric is the space ${\displaystyle C([a,b])}$ of continuous complex valued functions ${\displaystyle f}$ and ${\displaystyle g}$ on the interval ${\displaystyle [a,b].}$ The inner product is

$$
{\displaystyle \langle f,g\rangle =\int _{a}^{b}f(t){\overline {g(t)}}\,\mathrm {d} t.}
$$

This space is not complete; consider for example, for the interval \[−1, 1\] the sequence of continuous "step" functions, ${\displaystyle \{f_{k}\}_{k},}$ defined by:

$$
{\displaystyle f_{k}(t)={\begin{cases}0&t\in [-1,0]\\1&t\in \left[{\tfrac {1}{k}},1\right]\\kt&t\in \left(0,{\tfrac {1}{k}}\right)\end{cases}}}
$$

This sequence is a [Cauchy sequence](https://en.wikipedia.org/wiki/Cauchy_sequence "Cauchy sequence") for the norm induced by the preceding inner product, which does not converge to a _continuous_ function.

### Random variables

For real [random variables](https://en.wikipedia.org/wiki/Random_variable "Random variable") ${\displaystyle X}$ and ${\displaystyle Y,}$ the [expected value](https://en.wikipedia.org/wiki/Expected_value "Expected value") of their product

$$
{\displaystyle \langle X,Y\rangle =\mathbb {E} [XY]}
$$

is an inner product.[^9] [^10] [^11] In this case, ${\displaystyle \langle X,X\rangle =0}$ if and only if ${\displaystyle \mathbb {P} [X=0]=1}$ (that is, ${\displaystyle X=0}$ [almost surely](https://en.wikipedia.org/wiki/Almost_surely "Almost surely")), where ${\displaystyle \mathbb {P} }$ denotes the [probability](https://en.wikipedia.org/wiki/Probability "Probability") of the event. This definition of expectation as inner product can be extended to [random vectors](https://en.wikipedia.org/wiki/Random_vector "Random vector") as well.

### Complex matrices

The inner product for complex square matrices of the same size is the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product "Frobenius inner product") ${\displaystyle \langle A,B\rangle :=\operatorname {tr} \left(AB^{\dagger }\right)}$ . Since trace and transposition are linear and the conjugation is on the second matrix, it is a sesquilinear operator. We further get Hermitian symmetry by,

$$
{\displaystyle \langle A,B\rangle =\operatorname {tr} \left(AB^{\dagger }\right)={\overline {\operatorname {tr} \left(BA^{\dagger }\right)}}={\overline {\left\langle B,A\right\rangle }}}
$$

Finally, since for ${\displaystyle A}$ nonzero, ${\displaystyle \langle A,A\rangle =\sum _{ij}\left|A_{ij}\right|^{2}>0}$ , we get that the Frobenius inner product is positive definite too, and so is an inner product.

On an inner product space, or more generally a vector space with a [nondegenerate form](https://en.wikipedia.org/wiki/Nondegenerate_form "Nondegenerate form") (hence an isomorphism ${\displaystyle V\to V^{*}}$ ), vectors can be sent to covectors (in coordinates, via transpose), so that one can take the inner product and outer product of two vectors—not simply of a vector and a covector.

Every inner product space induces a [norm](<https://en.wikipedia.org/wiki/Norm_(mathematics)> "Norm (mathematics)"), called its _canonical norm_, that is defined by

$$
{\displaystyle \|x\|={\sqrt {\langle x,x\rangle }}.}
$$

With this norm, every inner product space becomes a [normed vector space](https://en.wikipedia.org/wiki/Normed_vector_space "Normed vector space").

So, every general property of normed vector spaces applies to inner product spaces. In particular, one has the following properties:

[Absolute homogeneity](https://en.wikipedia.org/wiki/Absolute_homogeneity "Absolute homogeneity")

$$
{\displaystyle \|ax\|=|a|\,\|x\|}
$$

for every ${\displaystyle x\in V}$ and ${\displaystyle a\in F}$ (this results from ${\displaystyle \langle ax,ax\rangle =a{\overline {a}}\langle x,x\rangle }$ ).

[Triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality "Triangle inequality")

$$
{\displaystyle \|x+y\|\leq \|x\|+\|y\|}
$$

for ${\displaystyle x,y\in V.}$ These two properties show that one has indeed a norm.

[Cauchy–Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality "Cauchy–Schwarz inequality")

$$
{\displaystyle |\langle x,y\rangle |\leq \|x\|\,\|y\|}
$$

for every ${\displaystyle x,y\in V,}$ with equality if and only if ${\displaystyle x}$ and ${\displaystyle y}$ are [linearly dependent](https://en.wikipedia.org/wiki/Linearly_independent "Linearly independent").

[Parallelogram law](https://en.wikipedia.org/wiki/Parallelogram_law "Parallelogram law")

$$
{\displaystyle \|x+y\|^{2}+\|x-y\|^{2}=2\|x\|^{2}+2\|y\|^{2}}
$$

for every ${\displaystyle x,y\in V.}$ The parallelogram law is a necessary and sufficient condition for a norm to be defined by an inner product.

[Polarization identity](https://en.wikipedia.org/wiki/Polarization_identity "Polarization identity")

$$
{\displaystyle \|x+y\|^{2}=\|x\|^{2}+\|y\|^{2}+2\operatorname {Re} \langle x,y\rangle }
$$

for every ${\displaystyle x,y\in V.}$ The inner product can be retrieved from the norm by the polarization identity, since its imaginary part is the real part of ${\displaystyle \langle x,iy\rangle .}$

[Ptolemy's inequality](https://en.wikipedia.org/wiki/Ptolemy%27s_inequality "Ptolemy's inequality")

$$
{\displaystyle \|x-y\|\,\|z\|~+~\|y-z\|\,\|x\|~\geq ~\|x-z\|\,\|y\|}
$$

for every ${\displaystyle x,y,z\in V.}$ Ptolemy's inequality is a necessary and sufficient condition for a [seminorm](https://en.wikipedia.org/wiki/Seminorm "Seminorm") to be the norm defined by an inner product.[^12]

### Orthogonality

[Orthogonality](<https://en.wikipedia.org/wiki/Orthogonality_(mathematics)> "Orthogonality (mathematics)")

Two vectors ${\displaystyle x}$ and ${\displaystyle y}$ are said to be _orthogonal_, often written ${\displaystyle x\perp y,}$ if their inner product is zero, that is, if ${\displaystyle \langle x,y\rangle =0.}$
This happens if and only if ${\displaystyle \|x\|\leq \|x+sy\|}$ for all scalars ${\displaystyle s,}$ [^13] and if and only if the real-valued function ${\displaystyle f(s):=\|x+sy\|^{2}-\|x\|^{2}}$ is non-negative. (This is a consequence of the fact that, if ${\displaystyle y\neq 0}$ then the scalar ${\displaystyle s_{0}=-{\tfrac {\overline {\langle x,y\rangle }}{\|y\|^{2}}}}$ minimizes ${\displaystyle f}$ with value ${\displaystyle f\left(s_{0}\right)=-{\tfrac {|\langle x,y\rangle |^{2}}{\|y\|^{2}}},}$ which is always non positive).
For a _complex_ inner product space ${\displaystyle H,}$ a linear operator ${\displaystyle T:V\to V}$ is identically ${\displaystyle 0}$ if and only if ${\displaystyle x\perp Tx}$ for every ${\displaystyle x\in V.}$ [^13] This is not true in general for real inner product spaces, as it is a consequence of conjugate symmetry being distinct from symmetry for complex inner products. A counterexample in a real inner product space is ${\displaystyle T}$ a 90° rotation in ${\displaystyle \mathbb {R} ^{2}}$ , which maps every vector to an orthogonal vector but is not identically ${\displaystyle 0}$ .

[Orthogonal complement](https://en.wikipedia.org/wiki/Orthogonal_complement "Orthogonal complement")

The _orthogonal complement_ of a subset ${\displaystyle C\subseteq V}$ is the set ${\displaystyle C^{\bot }}$ of the vectors that are orthogonal to all elements of C; that is,

$$
{\displaystyle C^{\bot }:=\{\,y\in V:\langle y,c\rangle =0{\text{ for all }}c\in C\,\}.}
$$

This set ${\displaystyle C^{\bot }}$ is always a closed vector subspace of ${\displaystyle V}$ and if the [closure](<https://en.wikipedia.org/wiki/Closure_(topology)> "Closure (topology)") ${\displaystyle \operatorname {cl} _{V}C}$ of ${\displaystyle C}$ in ${\displaystyle V}$ is a vector subspace then ${\displaystyle \operatorname {cl} _{V}C=\left(C^{\bot }\right)^{\bot }.}$

[Pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem "Pythagorean theorem")

If ${\displaystyle x}$ and ${\displaystyle y}$ are orthogonal, then

$$
{\displaystyle \|x\|^{2}+\|y\|^{2}=\|x+y\|^{2}.}
$$

This may be proved by expressing the squared norms in terms of the inner products, using additivity for expanding the right-hand side of the equation.
The name _Pythagorean theorem_ arises from the geometric interpretation in [Euclidean geometry](https://en.wikipedia.org/wiki/Euclidean_geometry "Euclidean geometry").

[Parseval's identity](https://en.wikipedia.org/wiki/Parseval%27s_identity "Parseval's identity")

An [induction](https://en.wikipedia.org/wiki/Mathematical_induction "Mathematical induction") on the Pythagorean theorem yields: if ${\displaystyle x_{1},\ldots ,x_{n}}$ are pairwise orthogonal, then

$$
{\displaystyle \sum _{i=1}^{n}\|x_{i}\|^{2}=\left\|\sum _{i=1}^{n}x_{i}\right\|^{2}.}
$$

[Angle](https://en.wikipedia.org/wiki/Angle "Angle")

When ${\displaystyle \langle x,y\rangle }$ is a real number then the Cauchy–Schwarz inequality implies that ${\textstyle {\frac {\langle x,y\rangle }{\|x\|\,\|y\|}}\in [-1,1],}$ and thus that

$$
{\displaystyle \angle (x,y)=\arccos {\frac {\langle x,y\rangle }{\|x\|\,\|y\|}},}
$$

is a real number. This allows defining the (non oriented) _angle_ of two vectors in modern definitions of [Euclidean geometry](https://en.wikipedia.org/wiki/Euclidean_geometry "Euclidean geometry") in terms of [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra "Linear algebra"). This is also used in [data analysis](https://en.wikipedia.org/wiki/Data_analysis "Data analysis"), under the name " [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity "Cosine similarity") ", for comparing two vectors of data. Furthermore, if ${\displaystyle \langle x,y\rangle }$ is negative, the angle ${\displaystyle \angle (x,y)}$ is larger than 90 degrees. This property is often used in computer graphics (e.g., in [back-face culling](https://en.wikipedia.org/wiki/Back-face_culling "Back-face culling")) to analyze a direction without having to evaluate [trigonometric functions](https://en.wikipedia.org/wiki/Trigonometric_functions "Trigonometric functions").

Suppose that ${\displaystyle \langle \cdot ,\cdot \rangle }$ is an inner product on ${\displaystyle V}$ (so it is antilinear in its second argument). The [polarization identity](https://en.wikipedia.org/wiki/Polarization_identity "Polarization identity") shows that the [real part](https://en.wikipedia.org/wiki/Real_part "Real part") of the inner product is

$$
{\displaystyle \operatorname {Re} \langle x,y\rangle ={\frac {1}{4}}\left(\|x+y\|^{2}-\|x-y\|^{2}\right).}
$$

If ${\displaystyle V}$ is a real vector space then

$$
{\displaystyle \langle x,y\rangle =\operatorname {Re} \langle x,y\rangle ={\frac {1}{4}}\left(\|x+y\|^{2}-\|x-y\|^{2}\right)}
$$

and the [imaginary part](https://en.wikipedia.org/wiki/Imaginary_part "Imaginary part") (also called the _complex part_) of ${\displaystyle \langle \cdot ,\cdot \rangle }$ is always ${\displaystyle 0.}$

Assume for the rest of this section that ${\displaystyle V}$ is a complex vector space. The [polarization identity](https://en.wikipedia.org/wiki/Polarization_identity "Polarization identity") for complex vector spaces shows that

$$
{\displaystyle {\begin{alignedat}{4}\langle x,\ y\rangle &={\frac {1}{4}}\left(\|x+y\|^{2}-\|x-y\|^{2}+i\|x+iy\|^{2}-i\|x-iy\|^{2}\right)\\&=\operatorname {Re} \langle x,y\rangle +i\operatorname {Re} \langle x,iy\rangle .\\\end{alignedat}}}
$$

The map defined by ${\displaystyle \langle x\mid y\rangle =\langle y,x\rangle }$ for all ${\displaystyle x,y\in V}$ satisfies the axioms of the inner product except that it is antilinear in its _first_, rather than its second, argument. The real part of both ${\displaystyle \langle x\mid y\rangle }$ and ${\displaystyle \langle x,y\rangle }$ are equal to ${\displaystyle \operatorname {Re} \langle x,y\rangle }$ but the inner products differ in their complex part:

$$
{\displaystyle {\begin{alignedat}{4}\langle x\mid y\rangle &={\frac {1}{4}}\left(\|x+y\|^{2}-\|x-y\|^{2}-i\|x+iy\|^{2}+i\|x-iy\|^{2}\right)\\&=\operatorname {Re} \langle x,y\rangle -i\operatorname {Re} \langle x,iy\rangle .\\\end{alignedat}}}
$$

The last equality is similar to the formula [expressing a linear functional](https://en.wikipedia.org/wiki/Real_and_imaginary_parts_of_a_linear_functional "Real and imaginary parts of a linear functional") in terms of its real part.

These formulas show that every complex inner product is completely determined by its real part. Moreover, this real part defines an inner product on ${\displaystyle V,}$ considered as a real vector space. There is thus a one-to-one correspondence between complex inner products on a complex vector space ${\displaystyle V,}$ and real inner products on ${\displaystyle V.}$

For example, suppose that ${\displaystyle V=\mathbb {C} ^{n}}$ for some integer ${\displaystyle n>0.}$ When ${\displaystyle V}$ is considered as a real vector space in the usual way (meaning that it is identified with the ${\displaystyle 2n-}$ dimensional real vector space ${\displaystyle \mathbb {R} ^{2n},}$ with each ${\displaystyle \left(a_{1}+ib_{1},\ldots ,a_{n}+ib_{n}\right)\in \mathbb {C} ^{n}}$ identified with ${\displaystyle \left(a_{1},b_{1},\ldots ,a_{n},b_{n}\right)\in \mathbb {R} ^{2n}}$ ), then the [dot product](https://en.wikipedia.org/wiki/Dot_product "Dot product") ${\displaystyle x\,\cdot \,y=\left(x_{1},\ldots ,x_{2n}\right)\,\cdot \,\left(y_{1},\ldots ,y_{2n}\right):=x_{1}y_{1}+\cdots +x_{2n}y_{2n}}$ defines a real inner product on this space. The unique complex inner product ${\displaystyle \langle \,\cdot ,\cdot \,\rangle }$ on ${\displaystyle V=\mathbb {C} ^{n}}$ induced by the dot product is the map that sends ${\displaystyle c=\left(c_{1},\ldots ,c_{n}\right),d=\left(d_{1},\ldots ,d_{n}\right)\in \mathbb {C} ^{n}}$ to ${\displaystyle \langle c,d\rangle :=c_{1}{\overline {d_{1}}}+\cdots +c_{n}{\overline {d_{n}}}}$ (because the real part of this map ${\displaystyle \langle \,\cdot ,\cdot \,\rangle }$ is equal to the dot product).

Let ${\displaystyle V_{\mathbb {R} }}$ denote ${\displaystyle V}$ considered as a vector space over the real numbers rather than complex numbers. The [real part](https://en.wikipedia.org/wiki/Real_part "Real part") of the complex inner product ${\displaystyle \langle x,y\rangle }$ is the map ${\displaystyle \langle x,y\rangle _{\mathbb {R} }=\operatorname {Re} \langle x,y\rangle ~:~V_{\mathbb {R} }\times V_{\mathbb {R} }\to \mathbb {R} ,}$ which necessarily forms a real inner product on the real vector space ${\displaystyle V_{\mathbb {R} }.}$ Every inner product on a real vector space is a [bilinear](https://en.wikipedia.org/wiki/Bilinear_map "Bilinear map") and [symmetric map](https://en.wikipedia.org/wiki/Symmetric_map "Symmetric map").

For example, if ${\displaystyle V=\mathbb {C} }$ with inner product ${\displaystyle \langle x,y\rangle =x{\overline {y}},}$ where ${\displaystyle V}$ is a vector space over the field ${\displaystyle \mathbb {C} ,}$ then ${\displaystyle V_{\mathbb {R} }=\mathbb {R} ^{2}}$ is a vector space over ${\displaystyle \mathbb {R} }$ and ${\displaystyle \langle x,y\rangle _{\mathbb {R} }}$ is the [dot product](https://en.wikipedia.org/wiki/Dot_product "Dot product") ${\displaystyle x\cdot y,}$ where ${\displaystyle x=a+ib\in V=\mathbb {C} }$ is identified with the point ${\displaystyle (a,b)\in V_{\mathbb {R} }=\mathbb {R} ^{2}}$ (and similarly for ${\displaystyle y}$ ); thus the standard inner product ${\displaystyle \langle x,y\rangle =x{\overline {y}},}$ on ${\displaystyle \mathbb {C} }$ is an "extension" the dot product. Also, had ${\displaystyle \langle x,y\rangle }$ been instead defined to be the **[symmetric map](https://en.wikipedia.org/wiki/#math_Symmetry)** ${\displaystyle \langle x,y\rangle =xy}$ (rather than the usual **[conjugate symmetric map](https://en.wikipedia.org/wiki/#math_Conjugate_symmetry)** ${\displaystyle \langle x,y\rangle =x{\overline {y}}}$ ) then its real part ${\displaystyle \langle x,y\rangle _{\mathbb {R} }}$ would _not_ be the dot product; furthermore, without the complex conjugate, if ${\displaystyle x\in \mathbb {C} }$ but ${\displaystyle x\not \in \mathbb {R} }$ then ${\displaystyle \langle x,x\rangle =xx=x^{2}\not \in [0,\infty )}$ so the assignment ${\textstyle x\mapsto {\sqrt {\langle x,x\rangle }}}$ would not define a norm.

The next examples show that although real and complex inner products have many properties and results in common, they are not entirely interchangeable. For instance, if ${\displaystyle \langle x,y\rangle =0}$ then ${\displaystyle \langle x,y\rangle _{\mathbb {R} }=0,}$ but the next example shows that the converse is in general _not_ true. Given any ${\displaystyle x\in V,}$ the vector ${\displaystyle ix}$ (which is the vector ${\displaystyle x}$ rotated by 90°) belongs to ${\displaystyle V}$ and so also belongs to ${\displaystyle V_{\mathbb {R} }}$ (although scalar multiplication of ${\displaystyle x}$ by ${\displaystyle i={\sqrt {-1}}}$ is not defined in ${\displaystyle V_{\mathbb {R} },}$ the vector in ${\displaystyle V}$ denoted by ${\displaystyle ix}$ is nevertheless still also an element of ${\displaystyle V_{\mathbb {R} }}$ ). For the complex inner product, ${\displaystyle \langle x,ix\rangle =-i\|x\|^{2},}$ whereas for the real inner product the value is always ${\displaystyle \langle x,ix\rangle _{\mathbb {R} }=0.}$

If ${\displaystyle \langle \,\cdot ,\cdot \,\rangle }$ is a complex inner product and ${\displaystyle A:V\to V}$ is a continuous linear operator that satisfies ${\displaystyle \langle x,Ax\rangle =0}$ for all ${\displaystyle x\in V,}$ then ${\displaystyle A=0.}$ This statement is no longer true if ${\displaystyle \langle \,\cdot ,\cdot \,\rangle }$ is instead a real inner product, as this next example shows. Suppose that ${\displaystyle V=\mathbb {C} }$ has the inner product ${\displaystyle \langle x,y\rangle :=x{\overline {y}}}$ mentioned above. Then the map ${\displaystyle A:V\to V}$ defined by ${\displaystyle Ax=ix}$ is a linear map (linear for both ${\displaystyle V}$ and ${\displaystyle V_{\mathbb {R} }}$ ) that denotes rotation by ${\displaystyle 90^{\circ }}$ in the plane. Because ${\displaystyle x}$ and ${\displaystyle Ax}$ are perpendicular vectors and ${\displaystyle \langle x,Ax\rangle _{\mathbb {R} }}$ is just the dot product, ${\displaystyle \langle x,Ax\rangle _{\mathbb {R} }=0}$ for all vectors ${\displaystyle x;}$ nevertheless, this rotation map ${\displaystyle A}$ is certainly not identically ${\displaystyle 0.}$ In contrast, using the complex inner product gives ${\displaystyle \langle x,Ax\rangle =-i\|x\|^{2},}$ which (as expected) is not identically zero.

## Orthonormal sequences

Let ${\displaystyle V}$ be a finite dimensional inner product space of dimension ${\displaystyle n.}$ Recall that every [basis](<https://en.wikipedia.org/wiki/Basis_(linear_algebra)> "Basis (linear algebra)") of ${\displaystyle V}$ consists of exactly ${\displaystyle n}$ linearly independent vectors. Using the [Gram–Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process "Gram–Schmidt process") we may start with an arbitrary basis and transform it into an orthonormal basis. That is, into a basis in which all the elements are orthogonal and have unit norm. In symbols, a basis ${\displaystyle \{e_{1},\ldots ,e_{n}\}}$ is orthonormal if ${\displaystyle \langle e_{i},e_{j}\rangle =0}$ for every ${\displaystyle i\neq j}$ and ${\displaystyle \langle e_{i},e_{i}\rangle =\|e_{a}\|^{2}=1}$ for each index ${\displaystyle i.}$

This definition of orthonormal basis generalizes to the case of infinite-dimensional inner product spaces in the following way. Let ${\displaystyle V}$ be any inner product space. Then a collection

$$
{\displaystyle E=\left\{e_{a}\right\}_{a\in A}}
$$

is a _basis_ for ${\displaystyle V}$ if the subspace of ${\displaystyle V}$ generated by finite linear combinations of elements of ${\displaystyle E}$ is dense in ${\displaystyle V}$ (in the norm induced by the inner product). Say that ${\displaystyle E}$ is an _[orthonormal basis](https://en.wikipedia.org/wiki/Orthonormal_basis "Orthonormal basis")_ for ${\displaystyle V}$ if it is a basis and

$$
{\displaystyle \left\langle e_{a},e_{b}\right\rangle =0}
$$

if ${\displaystyle a\neq b}$ and ${\displaystyle \langle e_{a},e_{a}\rangle =\|e_{a}\|^{2}=1}$ for all ${\displaystyle a,b\in A.}$

Using an infinite-dimensional analog of the Gram-Schmidt process one may show:

**Theorem.** Any [separable](https://en.wikipedia.org/wiki/Separable_space "Separable space") inner product space has an orthonormal basis.

Using the [Hausdorff maximal principle](https://en.wikipedia.org/wiki/Hausdorff_maximal_principle "Hausdorff maximal principle") and the fact that in a [complete inner product space](https://en.wikipedia.org/wiki/Hilbert_space "Hilbert space") orthogonal projection onto linear subspaces is well-defined, one may also show that

**Theorem.** Any [complete inner product space](https://en.wikipedia.org/wiki/Hilbert_space "Hilbert space") has an orthonormal basis.

The two previous theorems raise the question of whether all inner product spaces have an orthonormal basis. The answer, it turns out is negative. This is a non-trivial result, and is proved below. The following proof is taken from Halmos's _A Hilbert Space Problem Book_ (see the references).

| Proof |
| ----- |

[Parseval's identity](https://en.wikipedia.org/wiki/Parseval%27s_identity "Parseval's identity") leads immediately to the following theorem:

**Theorem.** Let ${\displaystyle V}$ be a separable inner product space and ${\displaystyle \left\{e_{k}\right\}_{k}}$ an orthonormal basis of ${\displaystyle V.}$ Then the map

$$
{\displaystyle x\mapsto {\bigl \{}\langle e_{k},x\rangle {\bigr \}}_{k\in \mathbb {N} }}
$$

is an isometric linear map ${\displaystyle V\rightarrow \ell ^{2}}$ with a dense image.

This theorem can be regarded as an abstract form of [Fourier series](https://en.wikipedia.org/wiki/Fourier_series "Fourier series"), in which an arbitrary orthonormal basis plays the role of the sequence of [trigonometric polynomials](https://en.wikipedia.org/wiki/Trigonometric_polynomial "Trigonometric polynomial"). Note that the underlying index set can be taken to be any countable set (and in fact any set whatsoever, provided ${\displaystyle \ell ^{2}}$ is defined appropriately, as is explained in the article [Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space "Hilbert space")). In particular, we obtain the following result in the theory of Fourier series:

**Theorem.** Let ${\displaystyle V}$ be the inner product space ${\displaystyle C[-\pi ,\pi ].}$ Then the sequence (indexed on set of all integers) of continuous functions

$$
{\displaystyle e_{k}(t)={\frac {e^{ikt}}{\sqrt {2\pi }}}}
$$

is an orthonormal basis of the space ${\displaystyle C[-\pi ,\pi ]}$ with the ${\displaystyle L^{2}}$ inner product. The mapping

$$
{\displaystyle f\mapsto {\frac {1}{\sqrt {2\pi }}}\left\{\int _{-\pi }^{\pi }f(t)e^{-ikt}\,\mathrm {d} t\right\}_{k\in \mathbb {Z} }}
$$

is an isometric linear map with dense image.

Orthogonality of the sequence ${\displaystyle \{e_{k}\}_{k}}$ follows immediately from the fact that if ${\displaystyle k\neq j,}$ then

$$
{\displaystyle \int _{-\pi }^{\pi }e^{-i(j-k)t}\,\mathrm {d} t=0.}
$$

Normality of the sequence is by design, that is, the coefficients are so chosen so that the norm comes out to 1. Finally the fact that the sequence has a dense algebraic span, in the _inner product norm_, follows from the fact that the sequence has a dense algebraic span, this time in the space of continuous periodic functions on ${\displaystyle [-\pi ,\pi ]}$ with the uniform norm. This is the content of the [Weierstrass theorem](https://en.wikipedia.org/wiki/Weierstrass_approximation_theorem "Weierstrass approximation theorem") on the uniform density of trigonometric polynomials.

Several types of [linear](https://en.wikipedia.org/wiki/Linear "Linear") maps ${\displaystyle A:V\to W}$ between inner product spaces ${\displaystyle V}$ and ${\displaystyle W}$ are of relevance:

- _[Continuous linear maps](https://en.wikipedia.org/wiki/Continuous_linear_operator "Continuous linear operator")_: ${\displaystyle A:V\to W}$ is linear and continuous with respect to the metric defined above, or equivalently, ${\displaystyle A}$ is linear and the set of non-negative reals ${\displaystyle \{\|Ax\|:\|x\|\leq 1\},}$ where ${\displaystyle x}$ ranges over the closed unit ball of ${\displaystyle V,}$ is bounded.
- _Symmetric linear operators_: ${\displaystyle A:V\to W}$ is linear and ${\displaystyle \langle Ax,y\rangle =\langle x,Ay\rangle }$ for all ${\displaystyle x,y\in V.}$
- _[Isometries](https://en.wikipedia.org/wiki/Isometry "Isometry")_: ${\displaystyle A:V\to W}$ satisfies ${\displaystyle \|Ax\|=\|x\|}$ for all ${\displaystyle x\in V.}$ A _linear isometry_ (resp. an _[antilinear](https://en.wikipedia.org/wiki/Antilinear_map "Antilinear map") isometry_) is an isometry that is also a linear map (resp. an [antilinear map](https://en.wikipedia.org/wiki/Antilinear_map "Antilinear map")). For inner product spaces, the [polarization identity](https://en.wikipedia.org/wiki/Polarization_identity "Polarization identity") can be used to show that ${\displaystyle A}$ is an isometry if and only if ${\displaystyle \langle Ax,Ay\rangle =\langle x,y\rangle }$ for all ${\displaystyle x,y\in V.}$ All isometries are [injective](https://en.wikipedia.org/wiki/Injective "Injective"). The [Mazur–Ulam theorem](https://en.wikipedia.org/wiki/Mazur%E2%80%93Ulam_theorem "Mazur–Ulam theorem") establishes that every surjective isometry between two _real_ normed spaces is an [affine transformation](https://en.wikipedia.org/wiki/Affine_transformation "Affine transformation"). Consequently, an isometry ${\displaystyle A}$ between real inner product spaces is a linear map if and only if ${\displaystyle A(0)=0.}$ Isometries are [morphisms](https://en.wikipedia.org/wiki/Morphism "Morphism") between inner product spaces, and morphisms of real inner product spaces are orthogonal transformations (compare with [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix "Orthogonal matrix")).
- _Isometrical isomorphisms_: ${\displaystyle A:V\to W}$ is an isometry which is [surjective](https://en.wikipedia.org/wiki/Surjective "Surjective") (and hence [bijective](https://en.wikipedia.org/wiki/Bijective "Bijective")). Isometrical isomorphisms are also known as unitary operators (compare with [unitary matrix](https://en.wikipedia.org/wiki/Unitary_matrix "Unitary matrix")).

From the point of view of inner product space theory, there is no need to distinguish between two spaces which are isometrically isomorphic. The [spectral theorem](https://en.wikipedia.org/wiki/Spectral_theorem "Spectral theorem") provides a canonical form for symmetric, unitary and more generally [normal operators](https://en.wikipedia.org/wiki/Normal_operator "Normal operator") on finite dimensional inner product spaces. A generalization of the spectral theorem holds for continuous normal operators in Hilbert spaces.[^14]

## Generalizations

Any of the axioms of an inner product may be weakened, yielding generalized notions. The generalizations that are closest to inner products occur where bilinearity and conjugate symmetry are retained, but positive-definiteness is weakened.

If ${\displaystyle V}$ is a vector space and ${\displaystyle \langle \,\cdot \,,\,\cdot \,\rangle }$ a semi-definite sesquilinear form, then the function:

$$
{\displaystyle \|x\|={\sqrt {\langle x,x\rangle }}}
$$

makes sense and satisfies all the properties of norm except that ${\displaystyle \|x\|=0}$ does not imply ${\displaystyle x=0}$ (such a functional is then called a [semi-norm](https://en.wikipedia.org/wiki/Semi-norm "Semi-norm")). We can produce an inner product space by considering the quotient ${\displaystyle W=V/\{x:\|x\|=0\}.}$ The sesquilinear form ${\displaystyle \langle \,\cdot \,,\,\cdot \,\rangle }$ factors through ${\displaystyle W.}$

This construction is used in numerous contexts. The [Gelfand–Naimark–Segal construction](https://en.wikipedia.org/wiki/Gelfand%E2%80%93Naimark%E2%80%93Segal_construction "Gelfand–Naimark–Segal construction") is a particularly important example of the use of this technique. Another example is the representation of [semi-definite kernels](https://en.wikipedia.org/wiki/Mercer%27s_theorem "Mercer's theorem") on arbitrary sets.

Alternatively, one may require that the pairing be a [nondegenerate form](https://en.wikipedia.org/wiki/Nondegenerate_form "Nondegenerate form"), meaning that for all non-zero ${\displaystyle x\neq 0}$ there exists some ${\displaystyle y}$ such that ${\displaystyle \langle x,y\rangle \neq 0,}$ though ${\displaystyle y}$ need not equal ${\displaystyle x}$ ; in other words, the induced map to the dual space ${\displaystyle V\to V^{*}}$ is injective. This generalization is important in [differential geometry](https://en.wikipedia.org/wiki/Differential_geometry "Differential geometry"): a manifold whose tangent spaces have an inner product is a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold "Riemannian manifold"), while if this is related to nondegenerate conjugate symmetric form the manifold is a [pseudo-Riemannian manifold](https://en.wikipedia.org/wiki/Pseudo-Riemannian_manifold "Pseudo-Riemannian manifold"). By [Sylvester's law of inertia](https://en.wikipedia.org/wiki/Sylvester%27s_law_of_inertia "Sylvester's law of inertia"), just as every inner product is similar to the dot product with positive weights on a set of vectors, every nondegenerate conjugate symmetric form is similar to the dot product with _nonzero_ weights on a set of vectors, and the number of positive and negative weights are called respectively the positive index and negative index. Product of vectors in [Minkowski space](https://en.wikipedia.org/wiki/Minkowski_space "Minkowski space") is an example of indefinite inner product, although, technically speaking, it is not an inner product according to the standard definition above. Minkowski space has four [dimensions](<https://en.wikipedia.org/wiki/Dimension_(mathematics)> "Dimension (mathematics)") and indices 3 and 1 (assignment of ["+" and "−"](<https://en.wikipedia.org/wiki/Sign_(mathematics)> "Sign (mathematics)") to them [differs depending on conventions](https://en.wikipedia.org/wiki/Sign_convention#Metric_signature "Sign convention")).

Purely algebraic statements (ones that do not use positivity) usually only rely on the nondegeneracy (the injective homomorphism ${\displaystyle V\to V^{*}}$ ) and thus hold more generally.

The term "inner product" is opposed to [outer product](https://en.wikipedia.org/wiki/Outer_product "Outer product") ([[thoughts/Tensor field#tensor product|tensor product]]), which is a slightly more general opposite. Simply, in coordinates, the inner product is the product of a ${\displaystyle 1\times n}$ _covector_ with an ${\displaystyle n\times 1}$ vector, yielding a ${\displaystyle 1\times 1}$ matrix (a scalar), while the outer product is the product of an ${\displaystyle m\times 1}$ vector with a ${\displaystyle 1\times n}$ covector, yielding an ${\displaystyle m\times n}$ matrix. The outer product is defined for different dimensions, while the inner product requires the same dimension. If the dimensions are the same, then the inner product is the _[trace](<https://en.wikipedia.org/wiki/Trace_(linear*algebra)> "Trace (linear algebra)")* of the outer product (trace only being properly defined for square matrices). In an informal summary: "inner is horizontal times vertical and shrinks down, outer is vertical times horizontal and expands out".

More abstractly, the outer product is the bilinear map ${\displaystyle W\times V^{*}\to \hom(V,W)}$ sending a vector and a covector to a rank 1 linear transformation ([simple tensor](https://en.wikipedia.org/wiki/Simple_tensor "Simple tensor") of type (1, 1)), while the inner product is the bilinear evaluation map ${\displaystyle V^{*}\times V\to F}$ given by evaluating a covector on a vector; the order of the domain vector spaces here reflects the covector/vector distinction.

The inner product and outer product should not be confused with the [interior product](https://en.wikipedia.org/wiki/Interior_product "Interior product") and [exterior product](https://en.wikipedia.org/wiki/Exterior_product "Exterior product"), which are instead operations on [vector fields](https://en.wikipedia.org/wiki/Vector_field "Vector field") and [differential forms](https://en.wikipedia.org/wiki/Differential_form "Differential form"), or more generally on the [exterior algebra](https://en.wikipedia.org/wiki/Exterior_algebra "Exterior algebra").

As a further complication, in [geometric algebra](https://en.wikipedia.org/wiki/Geometric_algebra "Geometric algebra") the inner product and the _exterior_ (Grassmann) product are combined in the geometric product (the Clifford product in a [Clifford algebra](https://en.wikipedia.org/wiki/Clifford_algebra "Clifford algebra")) – the inner product sends two vectors (1-vectors) to a scalar (a 0-vector), while the exterior product sends two vectors to a bivector (2-vector) – and in this context the exterior product is usually called the _outer product_ (alternatively, _[wedge product](https://en.wikipedia.org/wiki/Wedge_product "Wedge product")_). The inner product is more correctly called a _scalar_ product in this context, as the nondegenerate quadratic form in question need not be positive definite (need not be an inner product).

## See also

- [Bilinear form](https://en.wikipedia.org/wiki/Bilinear_form "Bilinear form") – Scalar-valued bilinear function
- [Biorthogonal system](https://en.wikipedia.org/wiki/Biorthogonal_system "Biorthogonal system")
- [Dual space](https://en.wikipedia.org/wiki/Dual_space "Dual space") – In mathematics, vector space of linear forms
- [Energetic space](https://en.wikipedia.org/wiki/Energetic_space "Energetic space")
- [L-semi-inner product](https://en.wikipedia.org/wiki/L-semi-inner_product "L-semi-inner product") – Generalization of inner products that applies to all normed spaces
- [Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance "Minkowski distance") – Vector distance using pth powers
- [Orthogonal basis](https://en.wikipedia.org/wiki/Orthogonal_basis "Orthogonal basis") – Basis for v whose vectors are mutually orthogonal
- [Orthogonal complement](https://en.wikipedia.org/wiki/Orthogonal_complement "Orthogonal complement") – Concept in linear algebra
- [Orthonormal basis](https://en.wikipedia.org/wiki/Orthonormal_basis "Orthonormal basis") – Specific linear basis (mathematics)
- [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold "Riemannian manifold")

## Bibliography

- Axler, Sheldon (1997). _Linear Algebra Done Right_ (2nd ed.). Berlin, New York: [Springer-Verlag](https://en.wikipedia.org/wiki/Springer-Verlag "Springer-Verlag"). [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-387-98258-8](https://en.wikipedia.org/wiki/Special:BookSources/978-0-387-98258-8 "Special:BookSources/978-0-387-98258-8").
- [Dieudonné, Jean](https://en.wikipedia.org/wiki/Jean_Dieudonn%C3%A9 "Jean Dieudonné") (1969). _Treatise on Analysis, Vol. I \[Foundations of Modern Analysis\]_ (2nd ed.). [Academic Press](https://en.wikipedia.org/wiki/Academic_Press "Academic Press"). [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-1-4067-2791-3](https://en.wikipedia.org/wiki/Special:BookSources/978-1-4067-2791-3 "Special:BookSources/978-1-4067-2791-3").
- Emch, Gerard G. (1972). _Algebraic Methods in Statistical Mechanics and Quantum Field Theory_. [Wiley-Interscience](https://en.wikipedia.org/wiki/Wiley-Interscience "Wiley-Interscience"). [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-471-23900-0](https://en.wikipedia.org/wiki/Special:BookSources/978-0-471-23900-0 "Special:BookSources/978-0-471-23900-0").
- [Halmos, Paul R.](https://en.wikipedia.org/wiki/Paul_Richard_Halmos "Paul Richard Halmos") (8 November 1982). _A Hilbert Space Problem Book_. [Graduate Texts in Mathematics](https://en.wikipedia.org/wiki/Graduate_Texts_in_Mathematics "Graduate Texts in Mathematics"). Vol. 19 (2nd ed.). New York: [Springer-Verlag](https://en.wikipedia.org/wiki/Springer_Publishing "Springer Publishing"). [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-387-90685-0](https://en.wikipedia.org/wiki/Special:BookSources/978-0-387-90685-0 "Special:BookSources/978-0-387-90685-0"). [OCLC](<https://en.wikipedia.org/wiki/OCLC_(identifier)> "OCLC (identifier)") [8169781](https://search.worldcat.org/oclc/8169781).
- [Lax, Peter D.](https://en.wikipedia.org/wiki/Peter_D._Lax "Peter D. Lax") (2002). [_Functional Analysis_](http://www.math.univ-metz.fr/~gnc/bibliographie/Functional%20Analysis/Lax,.Functional.Analysis,.Wiley,.2002,.603s.pdf) (PDF). Pure and Applied Mathematics. New York: Wiley-Interscience. [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-471-55604-6](https://en.wikipedia.org/wiki/Special:BookSources/978-0-471-55604-6 "Special:BookSources/978-0-471-55604-6"). [OCLC](<https://en.wikipedia.org/wiki/OCLC_(identifier)> "OCLC (identifier)") [47767143](https://search.worldcat.org/oclc/47767143). Retrieved July 22, 2020.
- [Rudin, Walter](https://en.wikipedia.org/wiki/Walter_Rudin "Walter Rudin") (1991). [_Functional Analysis_](https://archive.org/details/functionalanalys00rudi). International Series in Pure and Applied Mathematics. Vol. 8 (Second ed.). New York, NY: [McGraw-Hill Science/Engineering/Math](https://en.wikipedia.org/wiki/McGraw-Hill_Science/Engineering/Math "McGraw-Hill Science/Engineering/Math"). [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-07-054236-5](https://en.wikipedia.org/wiki/Special:BookSources/978-0-07-054236-5 "Special:BookSources/978-0-07-054236-5"). [OCLC](<https://en.wikipedia.org/wiki/OCLC_(identifier)> "OCLC (identifier)") [21163277](https://search.worldcat.org/oclc/21163277).
- [Schaefer, Helmut H.](https://en.wikipedia.org/wiki/Helmut_H._Schaefer "Helmut H. Schaefer"); Wolff, Manfred P. (1999). _Topological Vector Spaces_. [GTM](https://en.wikipedia.org/wiki/Graduate_Texts_in_Mathematics "Graduate Texts in Mathematics"). Vol. 8 (Second ed.). New York, NY: Springer New York Imprint Springer. [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-1-4612-7155-0](https://en.wikipedia.org/wiki/Special:BookSources/978-1-4612-7155-0 "Special:BookSources/978-1-4612-7155-0"). [OCLC](<https://en.wikipedia.org/wiki/OCLC_(identifier)> "OCLC (identifier)") [840278135](https://search.worldcat.org/oclc/840278135).
- [Schechter, Eric](https://en.wikipedia.org/wiki/Eric_Schechter "Eric Schechter") (1996). _Handbook of Analysis and Its Foundations_. San Diego, CA: Academic Press. [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-12-622760-4](https://en.wikipedia.org/wiki/Special:BookSources/978-0-12-622760-4 "Special:BookSources/978-0-12-622760-4"). [OCLC](<https://en.wikipedia.org/wiki/OCLC_(identifier)> "OCLC (identifier)") [175294365](https://search.worldcat.org/oclc/175294365).
- Swartz, Charles (1992). _An introduction to Functional Analysis_. New York: M. Dekker. [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-8247-8643-4](https://en.wikipedia.org/wiki/Special:BookSources/978-0-8247-8643-4 "Special:BookSources/978-0-8247-8643-4"). [OCLC](<https://en.wikipedia.org/wiki/OCLC_(identifier)> "OCLC (identifier)") [24909067](https://search.worldcat.org/oclc/24909067).
- [Trèves, François](https://en.wikipedia.org/wiki/Fran%C3%A7ois_Tr%C3%A8ves "François Trèves") (2006) \[1967\]. _Topological Vector Spaces, Distributions and Kernels_. Mineola, N.Y.: Dover Publications. [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-486-45352-1](https://en.wikipedia.org/wiki/Special:BookSources/978-0-486-45352-1 "Special:BookSources/978-0-486-45352-1"). [OCLC](<https://en.wikipedia.org/wiki/OCLC_(identifier)> "OCLC (identifier)") [853623322](https://search.worldcat.org/oclc/853623322).
- Young, Nicholas (1988). _An Introduction to Hilbert Space_. [Cambridge University Press](https://en.wikipedia.org/wiki/Cambridge_University_Press "Cambridge University Press"). [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [978-0-521-33717-5](https://en.wikipedia.org/wiki/Special:BookSources/978-0-521-33717-5 "Special:BookSources/978-0-521-33717-5").
- Zamani, A.; Moslehian, M.S.; & Frank, M. (2015) "Angle Preserving Mappings", _Journal of Analysis and Applications_ 34: 485 to 500 [doi](<https://en.wikipedia.org/wiki/Doi_(identifier)> "Doi (identifier)"):[10.4171/ZAA/1551](https://doi.org/10.4171%2FZAA%2F1551)

[^1]: By combining the _linear in the first argument_ property with the _conjugate symmetry_ property you get _conjugate-linear in the second argument_: ${\textstyle \langle x,by\rangle =\langle x,y\rangle {\overline {b}}}$ . This is how the inner product was originally defined and is used in most mathematical contexts. A different convention has been adopted in theoretical physics and quantum mechanics, originating in the [bra-ket](https://en.wikipedia.org/wiki/Bra-ket "Bra-ket") notation of [Paul Dirac](https://en.wikipedia.org/wiki/Paul_Dirac "Paul Dirac"), where the inner product is taken to be _linear in the second argument_ and _conjugate-linear in the first argument_; this convention is used in many other domains such as engineering and computer science.

[^2]: [Trèves 2006](https://en.wikipedia.org/wiki/#CITEREFTr%C3%A8ves2006), pp. 112–125.

[^3]: [Schaefer & Wolff 1999](https://en.wikipedia.org/wiki/#CITEREFSchaeferWolff1999), pp. 40–45.

[^4]: Moore, Gregory H. (1995). ["The axiomatization of linear algebra: 1875-1940"](https://doi.org/10.1006%2Fhmat.1995.1025). _Historia Mathematica_. **22** (3): 262– 303. [doi](<https://en.wikipedia.org/wiki/Doi_(identifier)> "Doi (identifier)"):[10.1006/hmat.1995.1025](https://doi.org/10.1006%2Fhmat.1995.1025).

[^5]: [Schaefer & Wolff 1999](https://en.wikipedia.org/wiki/#CITEREFSchaeferWolff1999), pp. 36–72.

[^6]: Jain, P. K.; Ahmad, Khalil (1995). ["5.1 Definitions and basic properties of inner product spaces and Hilbert spaces"](https://books.google.com/books?id=yZ68h97pnAkC&pg=PA203). _Functional Analysis_ (2nd ed.). New Age International. p. 203. [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [81-224-0801-X](https://en.wikipedia.org/wiki/Special:BookSources/81-224-0801-X "Special:BookSources/81-224-0801-X").

[^7]: Prugovečki, Eduard (1981). ["Definition 2.1"](https://books.google.com/books?id=GxmQxn2PF3IC&pg=PA18). _Quantum Mechanics in Hilbert Space_ (2nd ed.). Academic Press. pp. 18ff. [ISBN](<https://en.wikipedia.org/wiki/ISBN_(identifier)> "ISBN (identifier)") [0-12-566060-X](https://en.wikipedia.org/wiki/Special:BookSources/0-12-566060-X "Special:BookSources/0-12-566060-X").

[^8]: [Schaefer & Wolff 1999](https://en.wikipedia.org/wiki/#CITEREFSchaeferWolff1999), p. 44.

[^9]: Ouwehand, Peter (November 2010). ["Spaces of Random Variables"](https://web.archive.org/web/20170905225616/http://users.aims.ac.za/~pouw/Lectures/Lecture_Spaces_Random_Variables.pdf) (PDF). _AIMS_. Archived from [the original](http://users.aims.ac.za/~pouw/Lectures/Lecture_Spaces_Random_Variables.pdf) (PDF) on 2017-09-05. Retrieved 2017-09-05.

[^10]: Siegrist, Kyle (1997). ["Vector Spaces of Random Variables"](http://www.math.uah.edu/stat/expect/Spaces.html). _Random: Probability, Mathematical Statistics, Stochastic Processes_. Retrieved 2017-09-05.

[^11]: Bigoni, Daniele (2015). ["Appendix B: Probability theory and functional spaces"](http://orbit.dtu.dk/files/106969507/phd359_Bigoni_D.pdf) (PDF). _Uncertainty Quantification with Applications to Engineering Problems_ (PhD). Technical University of Denmark. Retrieved 2017-09-05.

[^12]: Apostol, Tom M. (1967). ["Ptolemy's Inequality and the Chordal Metric"](https://www.tandfonline.com/doi/pdf/10.1080/0025570X.1967.11975804). _Mathematics Magazine_. **40** (5): 233– 235. [doi](<https://en.wikipedia.org/wiki/Doi_(identifier)> "Doi (identifier)"):[10.2307/2688275](https://doi.org/10.2307%2F2688275). [JSTOR](<https://en.wikipedia.org/wiki/JSTOR_(identifier)> "JSTOR (identifier)") [2688275](https://www.jstor.org/stable/2688275).

[^13]: [Rudin 1991](https://en.wikipedia.org/wiki/#CITEREFRudin1991), pp. 306–312.

[^14]: [Rudin 1991](https://en.wikipedia.org/wiki/#CITEREFRudin1991)
