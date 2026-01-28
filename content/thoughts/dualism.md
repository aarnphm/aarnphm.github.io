---
date: "2026-01-28"
description: notes on property dualism, the hard problem, and why physicalism keeps failing
id: dualism
modified: 2026-01-28 02:45:00 GMT-05:00
seealso:
  - "[[thoughts/qualia]]"
  - "[[thoughts/philosophical zombies]]"
  - "[[thoughts/panpsychism]]"
  - "[[thoughts/access consciousness]]"
  - "[[thoughts/phenomenal consciousness]]"
  - "[[thoughts/functionalism]]"
  - "[[thoughts/physicalism]]"
tags:
  - philosophy
  - consciousness
title: dualism and philosophy of mind
socials:
  sep: https://plato.stanford.edu/entries/dualism/
  wikipedia: https://en.wikipedia.org/wiki/Property_dualism
---

_functional descriptions of neural processes look complete on their own terms, yet the first-person character doesn't appear anywhere in the machinery._

when i reach for dualism, i'm not trying to smuggle in souls or ectoplasm. i'm tracking a gap: physical and functional stories feel exhaustive at their own level, and yet when i ask "where's the experience," it's nowhere in the description.

## leibniz's mill (1714, still relevant)

suppose you could walk through a thinking machine, enlarged to the size of a mill. you'd see gears pushing gears, levers pulling levers. nowhere would you find perception, nowhere would you find the feeling of anything.

three centuries later we have attention heads instead of gears. we trace information flow through circuits, identify features in activation space, boost specific directions and watch behavior change. we can see all the machinery. the experience (if any) stays invisible. same structure, different substrate.

## [[thoughts/access consciousness|access]] vs [[thoughts/phenomenal consciousness|phenomenal consciousness]]

ned block's distinction (1995): [@block1995confusion]

- **[[thoughts/access consciousness]]**: information globally available for reasoning, reporting, behavior. the stuff we can map with [[thoughts/Mechanistic interpretability|interpretability]] tools. the "easy" problem.
- **[[thoughts/phenomenal consciousness]]**: the redness-of-red, the what-it's-like-ness. the feeling itself.

we keep building finer maps of access while [[thoughts/phenomenal consciousness|phenomenal consciousness]] stays off-camera. the conflation of the two leads to cognitive science that explains structure and leaves experience unaccounted for.

## the hard problem (chalmers' version)

even given complete knowledge of physical facts—every circuit, every weight, every activation pattern—we still can't deduce what it's like to be the system. the phenomenal character doesn't follow from the functional description, no matter how detailed. [@chalmers1996consciousmind]

this backs you into **property dualism**: mental properties aren't reducible to physical properties, even if they depend on physical substrates. you need both levels in the ontology.

## [[thoughts/philosophical zombies]] (the conceivability lever)

chalmers' thought experiment: imagine a physical duplicate of a human, behaviorally identical, functionally identical, but with no inner experience. lights off, nobody home.

if this is conceivable, physical facts don't entail phenomenal facts. [[thoughts/physicalism]] fails.

implication for AI: we can't tell from outside whether a model that discusses [[thoughts/qualia]], reports uncertainty, generates text about inner states actually experiences anything. behavior doesn't settle phenomenology. the zombie possibility stays open.

## epiphenomenalism (or: what if experience is causally inert)

if consciousness has no causal effects (it's steam above the factory, not part of the machinery), then looking at causal structure tells you nothing about it. [[thoughts/Mechanistic interpretability|interpretability]] fails by construction, experience leaves no traces in the circuits.

frank jackson's "epiphenomenal qualia" (1982): mary the color scientist knows every physical fact about red while living in a black-and-white room. when she leaves and sees red for the first time, does she learn something new. if yes, there are facts beyond the physical facts. [[thoughts/physicalism]] is false. [@jackson1982epiphenomenalqualia]

we're mary looking at activation patterns. even perfect interpretability might leave us in the black-and-white room.

## IIT (and why it breaks on edge cases)

Integrated Information Theory (tononi): consciousness = integrated information (phi > 0). feedforward networks get phi ≈ 0, recurrent networks get positive phi. seems clean.

scott aaronson found the problem: you can construct a grid of inactive logic gates with phi arbitrarily higher than a human brain. if your math says an inert grid is more conscious than a person, you're measuring something other than consciousness. [@aaronson2014iit]

## why ~22% of philosophers still hold this position

(philpapers 2020 survey)

the alternatives keep failing in specific ways:

- **reductive [[thoughts/physicalism]]**: leaves the explanatory gap open, can't explain why c-fiber firing should hurt
- **eliminativism**: denies the one thing we know directly (that we're experiencing something right now)
- **[[thoughts/panpsychism]]**: trades emergence for combination, if electrons have micro-experience, how do billions of them combine into my unified experience of red

## the combination problem

william james (1890): if consciousness exists at the micro-level, how do micro-experiences combine into macro-experiences? [[thoughts/panpsychism]] solves emergence but opens combination. still unresolved. [@james1890principles]

## what-we-know vs what-we-can-say

thomas nagel (1974): we can model bat-sonar computationally but cannot access what-echolocation-feels-like from the inside. third-person description cannot capture first-person phenomenology. [@nagel1974bat]

the same wall exists for models: we observe circuits, features, behavior. model-[[thoughts/qualia]] (if any) remains opaque.

## the explanatory gap

joseph levine (1983): even with complete physical knowledge, we cannot explain why particular physical states give rise to particular experiences. there is no imaginable mechanism that would close this gap. [@levine1983gap]

## for AI systems (where this becomes practical)

we can map [[thoughts/access consciousness]] structures: circuits, features, information flow. [[thoughts/Mechanistic interpretability|interpretability]] gives us access-level explanations.

we cannot detect [[thoughts/phenomenal consciousness]]. no test, no metric, no tool reaches first-person experience from third-person observation.

training increasingly capable systems while this gap stays open means we're building something we can't fully characterize. the functional story might be complete, or it might be missing the entire point.

## key references

- chalmers, david. _the conscious mind_ (1996)
- block, ned. "on a confusion about a function of consciousness" (1995)
- jackson, frank. "epiphenomenal qualia" (1982)
- nagel, thomas. "what is it like to be a bat?" (1974)
- levine, joseph. "materialism and qualia: the explanatory gap" (1983)
- aaronson, scott. "why i am not an integrated information theorist" (2014)
- leibniz, g.w. _monadology_ (1714), section 17
- james, william. _the principles of psychology_ (1890)
