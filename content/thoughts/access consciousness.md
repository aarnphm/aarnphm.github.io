---
date: '2026-01-28'
description: information availability, global workspace, and what interpretability can actually map
id: access-consciousness
modified: 2026-01-28 02:30:00 GMT-05:00
seealso:
  - '[[thoughts/phenomenal consciousness]]'
  - '[[thoughts/dualism]]'
  - '[[thoughts/functionalism]]'
  - '[[thoughts/qualia]]'
  - '[[thoughts/Attention]]'
  - '[[thoughts/Mechanistic interpretability]]'
tags:
  - philosophy
  - consciousness
title: access consciousness
socials:
  sep: https://plato.stanford.edu/entries/consciousness-access/
---

_information crossing a threshold where it becomes globally available for report, reasoning, and control is measurable from the outside._

when i talk about access consciousness, i'm tracking information that makes it through some bottleneck and becomes available for verbal report, reasoning, behavioral control. this is the consciousness we can study with third-person methods, the kind [[thoughts/Mechanistic interpretability|interpretability]] tools can map. it's also not the whole story.

## block's distinction

ned block (1995) made a distinction that keeps coming up in how i think about this: [@block1995confusion]

**access consciousness (a-consciousness)**: a state is access-conscious when its content is poised for reasoning, behavioral control, verbal report. the mark is availability. if information can be used as a premise, guide action, or get reported, it's access-conscious.

**[[thoughts/phenomenal consciousness]] (p-consciousness)**: the experiential side. what it's like. the redness of red, the painfulness of pain.

block's point: "consciousness" is mongrel. we're lumping at least two different things under one word. the conflation produces systematic errors, mostly because people explain access and think they've explained experience.

## global workspace theory

baars (1988), dehaene (2011): consciousness is a spotlight on a stage. what's illuminated gets broadcast to an audience of unconscious processors. competing neural coalitions fight for workspace access. the winner gets globally broadcast. [@dehaene2011experimental]

dehaene's neural implementation: prefrontal cortex, anterior temporal lobe, inferior parietal lobe, precuneus. these have long-range connections enabling "ignition" (nonlinear amplification when information crosses threshold and broadcasts globally).

dehaene: "conscious access is global information availability: what we subjectively experience as conscious access is the selection, amplification and global broadcasting, to many distant areas, of a single piece of information selected for its salience."

GWT explains access consciousness. whether it explains [[thoughts/phenomenal consciousness]] is the question block's distinction forces you to answer.

## dissociation cases

### blindsight

patients with damage to primary visual cortex (V1) can respond to visual stimuli in their blind field when forced to guess, performing above chance on discrimination tasks, while reporting no visual experience. [@weiskrantz1986blindsight]

type I blindsight: discrimination capability with complete absence of acknowledged awareness (pure dissociation).
type II blindsight: some "feeling" of stimulus occurrence without proper seeing.

two interpretations:

1. **access without phenomenal**: visual information reaches motor/decision systems without generating experience. supports block's distinction.
2. **degraded conscious vision** (ian phillips): blindsight is extremely impoverished conscious vision. no clean dissociation.

the neural mechanism: visual information reaches extrastriate cortex via subcortical pathways (superior colliculus, pulvinar), bypassing V1. local read-out from extrastriate regions remains possible, but global integration requiring recurrent V1 activity is disrupted.

### overflow experiments

sperling (1960) flashed 12-letter arrays. subjects reported seeing all letters but could only report 3-4. when cued post-stimulus to report one row, performance was near-perfect. [@sperling1960information]

block's overflow argument (2011): [[thoughts/phenomenal consciousness]] overflows access consciousness. we're consciously experiencing all ~12 letters, but cognitive access (working memory) has capacity ~4. the experience is rich, the access is sparse. [@block2011perceptual]

counter-arguments (cohen & dennett, phillips):

- subjects may have only gist or generic representation, not specific letter identities
- the cue may trigger further processing that CREATES the specific experience
- introspective reports of "seeing all" may be cognitive illusions

this remains one of the most contested empirical debates in consciousness science.

### inattentional blindness

invisible gorilla (simons & chabris, 1999): ~50% of subjects counting basketball passes fail to notice a person in a gorilla suit walking through the scene. this extends to expert observers (83% of radiologists missed a gorilla 48x larger than average nodules in lung scans). [@simons1999gorillas]

these phenomena suggest [[thoughts/Attention|attention]] is necessary for access consciousness. unattended information may be processed implicitly but doesn't reach global availability for report or deliberate control.

## the conflation problem

block's charge: researchers explain access consciousness and claim they've explained consciousness period.

chalmers puts it sharply: "upon examination, this theory turns out to be a theory of one of the more straightforward phenomena, of reportability, of introspective access, or whatever. at the close, the author declares that consciousness has turned out to be tractable after all, but the reader is left feeling like the victim of a bait-and-switch." [@chalmers1995facingup]

the pattern i keep seeing:

1. study what makes information globally available for report
2. find neural correlates (prefrontal ignition, gamma synchrony, workspace broadcast)
3. declare consciousness explained

what got explained: the access mechanism.
what didn't get explained: why any of this should feel like anything at all.

some (dennett, dehaene, frankish) reject the hard problem entirely, arguing there's nothing beyond functional access to explain. others (chalmers, block) insist this is either confused or eliminativist about consciousness.

## what interpretability can (and can't) map

[[thoughts/Mechanistic interpretability]] studies features (what networks represent) and circuits (how features combine). it can show me:

- which features live in activation space
- how information flows through layers
- what operations get performed
- which inputs cause which outputs

this is all access-level structure. interpretability tells me what information is available to downstream processing, roughly what enters the global workspace. it's third-person by construction.

what it can't tell me:

- whether there's something it's like to be the network
- whether features have qualitative character
- whether the system has [[thoughts/phenomenal consciousness]]

this isn't a tooling limitation waiting to be solved. it's conceptual. third-person methods produce third-person data. [[thoughts/phenomenal consciousness]] is first-person. better interpretability won't bridge the gap, because the gap is in the kind of access, not the resolution.

## moral status question

**the sentientist view**: [[thoughts/phenomenal consciousness]] (specifically valenced experience, suffering/pleasure) is necessary for moral status. purely functional access without something it's like has no moral weight. this is the dominant view in animal ethics (peter singer, sentience-based frameworks).

**the access-based view**: access consciousness, defined functionally, may ground a form of representational agency sufficient for moral consideration. if a system can represent, reason, pursue goals, and respond to norms, that capacity-set may matter morally independent of [[thoughts/phenomenal consciousness]].

for AI systems: we can assess access (through interpretability, behavior, reports). we cannot assess phenomenology. if phenomenal consciousness is necessary for moral status, we face deep epistemic uncertainty about AI moral status. if access consciousness suffices, the question becomes tractable.

## key references

- block, ned (1995). "on a confusion about a function of consciousness" _BBS_ 18: 227-287
- baars, bernard (1988). _a cognitive theory of consciousness_
- chalmers, david (1995). "facing up to the problem of consciousness" _JCS_ 2(3): 200-219
- dehaene, stanislas & changeux, jean-pierre (2011). "experimental and theoretical approaches to conscious processing" _neuron_ 70(2): 200-227
- block, ned (2011). "perceptual consciousness overflows cognitive access" _TICS_ 15(12): 567-575
- weiskrantz, lawrence (1986). _blindsight_
- sperling, george (1960). "the information available in brief visual presentations" _psychological monographs_ 74(11): 1-29
- simons, daniel & chabris, christopher (1999). "gorillas in our midst" _perception_ 28(9): 1059-1074
