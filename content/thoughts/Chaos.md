---
date: '2024-01-08'
description: Chaos a la carte.
id: Chaos
modified: 2026-06-05 15:08:06 GMT-04:00
tags:
  - philosophy
title: Chaos
---

Full [[posts/Chaos|post]].

> Chaos: a looseness [collection](https://subconscious.substack.com/p/self-organizing-ideas) of one's [[thoughts/Will|will]] to life.

The word comes from Greek χάος (khaos), an [abyss or opening](https://www.merriam-webster.com/wordplay/chaos-meaning-and-history). The technical, philosophical and personal uses below ask different questions.

## as system.

See also [[thoughts/papers/Chaos as an intermittently forced linear system.pdf|Chaos as an intermittently forced linear system]].

A deterministic system specifies how its state changes. The Lorenz equations are one example:

$$
\begin{aligned}
\frac{dx}{dt} &= \sigma(y - x), \\
\frac{dy}{dt} &= x(\rho - z) - y, \\
\frac{dz}{dt} &= xy - \beta z.
\end{aligned}
$$

The state is $(x,y,z)$; $\sigma$, $\rho$ and $\beta$ are parameters. With $\sigma=10$, $\rho=28$ and $\beta=8/3$, typical trajectories approach the two-lobed [[thoughts/attractor|attractor]]. These are the parameters in Lorenz's [1963 numerical example](<https://doi.org/10.1175/1520-0469(1963)020%3C0130:DNF%3E2.0.CO;2>). The equations contain no random input. Starting from an exact state determines a trajectory, while an error in that state can grow until the forecast follows a different part of the attractor.

This sensitivity limits forecasts made from finite-precision measurements. During a period when a small error grows approximately exponentially,

$$
\delta(t) \approx \delta_0 e^{\lambda t}, \qquad \lambda>0.
$$

Here $\delta_0$ is the initial error, measured in a chosen state-space norm, and $\lambda$ is an effective growth rate. For a tolerated error $\varepsilon>\delta_0$, solving for the time when the error reaches that tolerance gives

$$
T \approx \frac{1}{\lambda}\ln\left(\frac{\varepsilon}{\delta_0}\right).
$$

The [[thoughts/Lyapunov time|Lyapunov time]] is $1/\lambda$. Reducing the initial error tenfold adds about $\ln(10)/\lambda$ to this estimated horizon. It does not multiply the horizon by ten. This estimate applies while exponential growth is a useful approximation, before separation saturates at the scale of the attractor. Model error and external disturbances can shorten the usable forecast further.

> Chaos: When the present determines the future, but the approximate present does not approximately determine the future.

This is the distinction between [[thoughts/Determinism|determinism]] and predictability. Apparent randomness alone does not establish chaos; a noisy measurement or a randomly forced system can also be hard to predict.

[[thoughts/Entropy|Entropy]] needs its own definition and a specified distribution or physical system. Chaotic sensitivity does not by itself establish an increase in thermodynamic entropy. In the Lorenz model, a small cloud of possible states stretches in some directions while its phase-space volume contracts. Directly differentiating the equations gives

$$
\nabla\!\cdot f=-\sigma-1-\beta=-\frac{41}{3}
$$

at the parameters above. Exponential growth of a perturbation can coexist with contraction of the cloud's volume. Calling both effects a loss of order hides that distinction.

## as scale.

See also [this tweet](https://twitter.com/eshear/status/1760755072571777412).

_How does Chaos influence the scale of human psyche?_

This is a question about experience. To turn it into a dynamical claim would require a measured state, a model of its evolution and evidence of sensitive dependence. The Lorenz equations alone say nothing about introversion, emotional turmoil or cognitive dissonance.

## fundamentals.

[[thoughts/Philosophy and Nietzsche|Nietzsche]], in the [prologue to _Thus Spoke Zarathustra_, §5](https://www.gutenberg.org/files/1998/1998-h/1998-h.htm), translated by Thomas Common:

> Alas! there cometh the time when man will no longer launch the arrow of his longing beyond man—and the string of his bow will have unlearned to whizz!
>
> I tell you: one must still have chaos in one, to give birth to a dancing star. I tell you: ye have still chaos in you.

The passage sets creative longing against the last man's settled comforts. Reading it as a claim about physical entropy would change its subject.

In [_The Gay Science_, §109](https://www.gutenberg.org/files/52881/52881-h/52881-h.htm), Nietzsche calls the world's character chaos while explicitly retaining necessity. His target is the purpose, beauty and organization people project onto the world. This use of chaos is also separate from the medieval [[thoughts/Transcendentals|transcendentals]].

The thought of eternal recurrence [^1] poses a further test in [_The Gay Science_, §341](https://www.gutenberg.org/files/52881/52881-h/52881-h.htm): could one affirm the return of every detail of one's life? The published passage presents a hypothetical challenge. It supplies no physical demonstration that recurrence occurs.

Philip J. Kain's [[thoughts/papers/Nietzsche the Kantian Self and Eternal Recurrence.pdf|Nietzsche, the Kantian Self, and Eternal Recurrence]] develops an interpretation in which recurrence gives continuity to a self composed of competing drives:

> The self, for Nietzsche, is not just a radically unstable postmodern self. It is such a self, but it is not simply such a self. It also has a stability, sameness, and unity that goes far beyond anything Kant ever imagined in his wildest dreams

This is [Kain's argument](https://scholarcommons.scu.edu/phi/52/), on manuscript page 17. It should be distinguished from Nietzsche's own words and from a theorem about chaotic systems. The question of creating values belongs with [[thoughts/Philosophy and Nietzsche|Nietzsche's criticism of inherited judgments]] and [[thoughts/Will to Power|will to power]]. Thermodynamic entropy does not provide a measure of those values.

## versus equanimity.

Equanimity should be one to seek, but yet chaos is all I desire. (moment of chaos, moment of equanimity)

The rule of a utilitarian is to maximize desire at all cost, therefore, does it mean I should always seek chaos?

That question contains a premise to examine. In [_Utilitarianism_, chapter II](https://www.gutenberg.org/files/11224/11224-h/11224-h.htm), Mill assesses happiness for everyone concerned. Utilitarian accounts require a measure of welfare and a way to assess the consequences. Satisfying one person's desire for chaos is insufficient to establish that an action is best. Its effects on other people and on later experience still matter.

[[thoughts/taste|Taste]] implies multiplicity of being. It is driven by inner chaos to explore and expands on our [[thoughts/Language|representation]] of the world.
Yet ignorance seems to overload chaos, and prevents the maximum utilization of ones potential. I wonder if chaos is just a collection of different entropic phenomena.

That last question remains a personal conjecture. A physical entropy would require a system, states and a measure; the attraction to uncertainty does not supply them. Equanimity describes how one responds to an experience. It leaves room for seeking unfamiliar experiences and for judging which disruptions are worth their consequences.

[^1]: See also [[thoughts/Gilles Deleuze#forces.|Giles Deleuze's]] interpretation.
