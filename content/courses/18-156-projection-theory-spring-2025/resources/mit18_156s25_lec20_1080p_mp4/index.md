---
title: 'lecture 20: homogeneous dynamics, part 1'
description: 'We give a friendly introduction to the field of homogeneous dynamics and how projection theory can help understand it.'
id: 'mit-18-156-spring-2025-resource-resources-mit18-156s25-lec20-1080p-mp4'
tags:
  - 'course'
  - 'mit'
  - 'ocw'
  - 'mathematics'
  - 'math'
  - 'differential equations'
  - 'mathematical analysis'
  - 'lecture videos'
semantic: false
resource:
  {
    kind: 'video',
    n: 20,
    youtube: 'rChKr66KG-c',
    duration: 4833,
    cues: 1133,
    captions: 'static_resources/mit18_156s25_lec20_1080p_captions.vtt',
    captionsId: 'mit18_156s25_lec20_1080p_captions_vtt',
    transcript: 'static_resources/mit18_156s25_lec20_1080p_transcript.pdf',
    transcriptId: 'mit18_156s25_lec20_1080p_transcript_pdf',
  }
layout: 'resource'
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
---

[0:00:13](https://youtu.be/rChKr66KG-c?t=13) LAWRENCE GUTH: Hey, everyone! Today and this week, we are going to talk about the connection between homogeneous dynamics and projection theory. Today is, by the way, the first day of the Simons lectures. Maryna Viazovska is giving them this year. And she's going to be talking about sphere packing. And I got interested in this two years ago when Elon Lindenstrauss came to give the Simons lectures. And he was talking about homogeneous dynamics. And he explained a little bit this connection with projection theory.

[0:00:46](https://youtu.be/rChKr66KG-c?t=46) And I have been trying to understand it since then. And it's a pretty geometric visual thing. So I brought some tools with me. The goal of the class is to try to understand, visually, geometrically, how homogeneous dynamics is related to projection theory.

[0:01:09](https://youtu.be/rChKr66KG-c?t=69) So let's first say a little bit what is homogeneous dynamics.

[0:01:34](https://youtu.be/rChKr66KG-c?t=94) So we're going to be working with spaces that have a lot of symmetries. And so the spaces will come from a Lie group. So G will be a Lie group, for example, S ln(R) And inside of G, there'll be a discrete subgroup.

[0:01:58](https://youtu.be/rChKr66KG-c?t=118) So e.g. S ln(Z) And then we're going to be studying the quotient space, G mod gamma.

[0:02:09](https://youtu.be/rChKr66KG-c?t=129) So this is called a homogeneous space. So we'll see it has a lot of symmetries. And so the homogeneous and homogeneous dynamics is reacting on this homogeneous space. Now, what does dynamics mean? Suppose that H is a subgroup of G. And then if x is a point in G mod gamma, then we can look at H dot x. So H acts on G mod gamma. So we can talk about H dot x. That's called an orbit.

[0:02:40](https://youtu.be/rChKr66KG-c?t=160) And the question is to describe the orbits.

[0:02:53](https://youtu.be/rChKr66KG-c?t=173) So let's make a picture. So in actual Lie group G-- group G looks like this-- there might be some group element. And then you could take a coset H dot g, and h might be non-compact. So that might go on forever. But somehow, it's not that complicated. Yeah.

[0:03:18](https://youtu.be/rChKr66KG-c?t=198) AUDIENCE: If we're taking the quotient of G mod gamma, what are the conditions for gamma to be a normal subgroup to allow this question?

[0:03:27](https://youtu.be/rChKr66KG-c?t=207) LAWRENCE GUTH: So this object is not a group. It's just a space.

[0:03:32](https://youtu.be/rChKr66KG-c?t=212) AUDIENCE: A space, OK.

[0:03:33](https://youtu.be/rChKr66KG-c?t=213) LAWRENCE GUTH: So gamma is a-- so the question was about whether gamma is normal. And the answer is gamma doesn't have to be normal. So this is not a normal subgroup of this. That means that this is not a group, but it's still a space.

[0:03:51](https://youtu.be/rChKr66KG-c?t=231) So G mod gamma-- oh, and I'll put something else. So G mod gamma, we'll assume that the volume-- we'll be interested in cases where the volume of G mod gamma is finite. So the volume of G is infinite, but the volume of G mod gamma is finite. So G mod gamma looks sort of like this.

[0:04:13](https://youtu.be/rChKr66KG-c?t=253) And so then a point x in there, and the orbit goes through x, but then it wraps around.

[0:04:25](https://youtu.be/rChKr66KG-c?t=265) The picture looks a lot more complicated already. And we want to understand what does this do. Is it going to wrap back on itself and just go around over and over again, the same thing? Or is it going to evenly distribute all over G mod gamma or something else?

[0:04:46](https://youtu.be/rChKr66KG-c?t=286) So to make it concrete, I'd like to begin with a very simple warm up, which is with a commutative group. So it's simpler than that.

[0:05:07](https://youtu.be/rChKr66KG-c?t=307) So I think the simplest example of this is that G could be R2, and then gamma could be Z2. So G mod gamma is R2 mod Z2. It's a torus.

[0:05:21](https://youtu.be/rChKr66KG-c?t=321) And H is going to be a subgroup of G. And let's say it's like a one-parameter subgroup.

[0:05:30](https://youtu.be/rChKr66KG-c?t=330) We'll do a couple of examples. H could be the subgroup t, 2t. t in R. There's a subgroup of R2. And if we do this, what happens? So I'll visualize G mod gamma as a square with sides identified. And if I were to take-- so let's say x0 is just gamma, which is an element of G mod gamma. It's like you take the identity in G and push it down.

[0:06:04](https://youtu.be/rChKr66KG-c?t=364) So what is the orbit through x0 look like? Well, it's this line of slope 2 looks like that. And then it comes back where it started. And it just keeps going like that. And actually, it wouldn't be-- it's a little harder to draw accurately, but it wouldn't be different if you started at any other point. So x, if you start at any point x, then if I were to add 1 comma 2, that would be equal to x in R2 mod Z2.

[0:06:36](https://youtu.be/rChKr66KG-c?t=396) And so, actually, every orbit is periodic. So the conclusion in example one is that H dot x is periodic for every x. Example two, I'll take a different one-parameter subgroup, t comma square root of 2t t in R. So now what happens? Well, if I, again, start at x0, it's the identity.

[0:07:08](https://youtu.be/rChKr66KG-c?t=428) And now I'm going up at slope square root of 2 at the top. And keep going at slope square root of 2. And this time, this will never close up because the slope is irrational. And in fact, this orbit will be dense and evenly distributed. So now what happens is that H dot x is dense for every x.

[0:07:43](https://youtu.be/rChKr66KG-c?t=463) I won't do a proof right now, but I'll make a remark. So there are several proofs, but there's a nice proof using Fourier analysis that we might talk about later if we feel like it.

[0:08:03](https://youtu.be/rChKr66KG-c?t=483) Questions or comments so far?

[0:08:08](https://youtu.be/rChKr66KG-c?t=488) So we've seen two things that can happen. The orbit could be periodic. And the orbit could be dense. And we haven't yet seen other things, but we might wonder about whether other things could happen. And now we can shift to a more complicated noncommutative group, like SL2(R).

[0:08:43](https://youtu.be/rChKr66KG-c?t=523) So now let's say G is SL2(R) and gamma is SL2(Z).

[0:08:55](https://youtu.be/rChKr66KG-c?t=535) There are multiple reasons for looking at these quotients, G mod gamma. But one of them that comes up for this particular example is that these parametrize lattices. So G mod gamma is the space of lattices in R2 with the property lattices lambda in R2 with the property that the area of R2 mod lambda is 1.

[0:09:21](https://youtu.be/rChKr66KG-c?t=561) So let's do a proof sketch.

[0:09:27](https://youtu.be/rChKr66KG-c?t=567) Any such lattice, we can write as g times the lattice Z squared, where G is in SL2(R).

[0:09:37](https://youtu.be/rChKr66KG-c?t=577) Make any lattice by starting with Z squared, and applying a linear transformation. However, there is more than one G that gives you the same lambda. So if H is in SL2(Z), then hZ2 is Z2. And therefore, G times hZ2 is g times Z2. And these guys give you the same lattice. So the space of lattices is g modulo SL2(Z).

[0:10:20](https://youtu.be/rChKr66KG-c?t=620) Now, this space of lattices, G mod gamma, is not compact. And I will give you an example of a sequence of lattices that don't converge to anything-- don't have a subsequence that converge to anything. So for example, we could take lambda epsilon to be epsilon Z direct sum epsilon inverse. So this is a lattice, which is very skinny in one direction and very tall in the other direction. It has covolume one.

[0:10:51](https://youtu.be/rChKr66KG-c?t=651) And as epsilon goes to 0, this is not going to converge to anything. So the space of lattices is not compact. And this is the only reason that the space of lattices is not compact. So a definition, K epsilon is a subset of G mod gamma. K epsilon is defined to be the set of lattices so that the minimum over nonzero v in our lattice of the size of v is at least epsilon-- is greater than or equal to, let's say, epsilon.

[0:11:25](https://youtu.be/rChKr66KG-c?t=685) And there's a proposition, which is called the Mahler criterion, that says for every epsilon bigger than 0, K epsilon is compact.

[0:11:45](https://youtu.be/rChKr66KG-c?t=705) So a picture of G mod gamma is something like this. G mod gamma is three-dimensional, so it's not super easy to draw. But over here, so this would be G mod gamma. There's this noncompact part. And we'll make a cut off maybe here. And this stuff is K epsilon.

[0:12:17](https://youtu.be/rChKr66KG-c?t=737) So I won't prove this in detail. This is not super difficult. If you think about two vectors that generate lambda, if we're in K epsilon, the vectors definitely cannot be too close to 0. And you can also always find two vectors that are not too giant, because the area of the thing is one. And this gives you a compact set of choices of these two vectors.

[0:12:41](https://youtu.be/rChKr66KG-c?t=761) So that gives a little bit of intuition about what the space G mod gamma looks like.

[0:12:51](https://youtu.be/rChKr66KG-c?t=771) Now, inside of there, we have some interesting one-parameter subgroups of G of SL2(R).

[0:13:04](https://youtu.be/rChKr66KG-c?t=784) So one of them is the unipotent subgroup. So that's the set of 1, t, 0, 1, t in R. And the other one is the diagonal subgroup. So I'll call this-- so it's the set of UT. So this matrix is U sub t. And there's the diagonal subgroup, e to the r, e to the minus r, 0, 0, little r in the reals. And this I'll call the set of a sub r. So this matrix here is called a sub r. So those are one-parameter subgroups.

[0:13:36](https://youtu.be/rChKr66KG-c?t=816) And now we can ask our question about what the orbits look like.

[0:13:47](https://youtu.be/rChKr66KG-c?t=827) So there's a theorem that goes back to the 1930s, Hedlund in the 1930s. He said that a U orbit is either periodic or dense. But in contrast, there's a-- I guess I'll also call it a theorem-- I don't know who proved this one-- that D orbit does not have to be maybe neither periodic nor dense.

[0:14:29](https://youtu.be/rChKr66KG-c?t=869) Let me say something more than that about how these D orbits can be really quite complicated. So here's G mod gamma. And inside of G mod gamma, pick out three small balls that are not too close together. So B1, B2, B3 are small balls.

[0:14:51](https://youtu.be/rChKr66KG-c?t=891) Then for any sequence of 1's and 2's, so for example, the sequence 1121212221, any sequence, there exists an x in G mod gamma so that the D orbit of x never hits B3. And the D orbit of x hits B1 and B2 many times in this sequence.

[0:15:34](https://youtu.be/rChKr66KG-c?t=934) So what must this orbit look like?

[0:15:40](https://youtu.be/rChKr66KG-c?t=940) Let's see. So first, it hits B1. It goes through there. Then it goes around, and the next thing it hits is also B1. And it goes around. And the next thing it hits is B2. And it keeps going. And the next thing it hits is B1, and so on. And it never touches B3.

[0:16:04](https://youtu.be/rChKr66KG-c?t=964) So this orbit, apparently, is not dense, because it never intersects B3. And it is not periodic if this sequence is not periodic. So it's something else.

[0:16:20](https://youtu.be/rChKr66KG-c?t=980) So D orbits can do this somewhat crazy thing, somewhat chaotic thing. But U orbits can't. U orbits can be periodic. So if you were to take U of x0, this is periodic.

[0:16:45](https://youtu.be/rChKr66KG-c?t=1005) And the reason is-- so if you take-- so x0 corresponds to the matrix 1, 1 times SL2(Z). So now if I were to take 1, 1, t, x0, that's equal to x0 in G mod gamma if t is an integer.

[0:17:11](https://youtu.be/rChKr66KG-c?t=1031) So that's a periodic orbit. It makes a closed circle in G mod gamma. But if you were to start with a kind of weird-looking choice of x0, and you were to do this, then there won't be any obvious periodicity, so for most x. So let's suppose now x is root 2, 0, 0, 1 over root 2.

[0:17:43](https://youtu.be/rChKr66KG-c?t=1063) So now what happens when we-- and then times SL2(Z). So now what is UT of x? That's 1, 1, t, root 2, 0, 1 over root 2.

[0:18:01](https://youtu.be/rChKr66KG-c?t=1081) So that will be root 2. That will be root 2. That will be t over root 2. This will be 0. This will be 1 over root 2.

[0:18:25](https://youtu.be/rChKr66KG-c?t=1105) Oh, that one might be periodic. I think I picked the wrong example.

[0:18:32](https://youtu.be/rChKr66KG-c?t=1112) Yeah, yeah. OK, sorry. That was unhelpful.

[0:18:45](https://youtu.be/rChKr66KG-c?t=1125) So to see why they're not all periodic, let's instead try to visualize what is happening to the lattice. So a point x corresponds to a lattice lambda x. And if you think about what is happening if you take gx, that corresponds to the lattice g of lambda of x. So g is in SL2(R).

[0:19:16](https://youtu.be/rChKr66KG-c?t=1156) It acts on R2. It moves the lattice to another lattice. And that's what the action is. And so now g is going to be ut. And the way ut acts on R2 is by shearing. So ut acts on R2 by shearing. So I'll do it this way.

[0:19:45](https://youtu.be/rChKr66KG-c?t=1185) So I'll draw some points in R2. And perhaps, coincidentally, these points come from a lattice. And then I'll illustrate what ut does. So if this is the point p, then this over here is ut of p. It's a shearing motion. And so the shearing motion moves everybody horizontally. And it moves you horizontally by an amount proportional to your height.

[0:20:16](https://youtu.be/rChKr66KG-c?t=1216) So this one moves horizontally, but a bit less. And this one moves horizontally a bit more. Now, what would it mean? So you start with a lattice. And when we apply this unipotent orbit to the lattice, every point in the lattice is shearing. What would it mean if the orbit was periodic? Well, it would mean that this point, this lattice point is shearing along and eventually it reaches another lattice point? So if ux is periodic, that implies that each lattice point is at the same height as another lattice point.

[0:21:09](https://youtu.be/rChKr66KG-c?t=1269) So there must be another lattice point somewhere over here. But now, think about this. This difference is also in our lattice, because the lattice is a subgroup. And this distance is horizontal. So this is if and only if there exists-- if and only if lambda of x contains a vector of the form x1 comma 0.

[0:21:41](https://youtu.be/rChKr66KG-c?t=1301) So most lattices don't contain a vector on the x-axis. So u times x is usually not periodic. And then the difficult theorem is in that situation u times x is dense.

[0:22:02](https://youtu.be/rChKr66KG-c?t=1322) So this is the situation in SL2. The situation in other Lie groups, say SLn(R) for larger n, is more difficult and more recent, but there are similar theorems.

[0:22:20](https://youtu.be/rChKr66KG-c?t=1340) For SL3(R), Margulis and Dani in the 1980s proved an analog of this theorem. And then for all the SLn(R) , and all the groups, Ratner, around 1990 proved an analog of this theorem.

[0:22:43](https://youtu.be/rChKr66KG-c?t=1363) So that's a little summary of some kinds of theorems. There's some different motivations for this. There's some motivation from-- yeah.

[0:22:51](https://youtu.be/rChKr66KG-c?t=1371) AUDIENCE: I have just a question about the link between the lattices and the elements of the quotient group. So up there in the definition of K epsilon, there's kind of what looks like some sort of norm, the minimum size of v, or the nonzero vector v in the lattice. Does that generate the same topology as the quotient topology from SL2(R)?

[0:23:20](https://youtu.be/rChKr66KG-c?t=1400) LAWRENCE GUTH: So the question is about the topology of G mod gamma and how the topology of G mod gamma plays with this definition of K epsilon.

[0:23:32](https://youtu.be/rChKr66KG-c?t=1412) In a moment, we'll put a metric on G mod gamma. And that's what this topology is. It's a manifold. It has a topology manifold. And this thing here is a continuous function on our space of lattices. Another way to say what the topology is, if you describe a lattice as the span of two particular vectors, if you slightly perturb those particular vectors, that will be a slight change in the space of lattices.

[0:24:11](https://youtu.be/rChKr66KG-c?t=1451) AUDIENCE: Is that definition similar in any way to have a minimum? Yeah, I guess you're right. It's not quite-- thanks.

[0:24:27](https://youtu.be/rChKr66KG-c?t=1467) LAWRENCE GUTH: So when I introduced homogeneous dynamics, I mentioned some application in number theory connection to the Oppenheim-Davenport conjecture, which is about the values of quadratic forms.

[0:24:43](https://youtu.be/rChKr66KG-c?t=1483) That's a nice connection. I feel like we already talked about it a little bit. And if you wanted to read about it, it's also easy to read about it. So I wanted to spend the class time on--

[0:24:56](https://youtu.be/rChKr66KG-c?t=1496) so I wanted to spend the class time on trying to get intuition for why something like this would be true, and also maybe why something like this would be false. And there are many different ways of thinking about homogeneous dynamics. There's a big literature, but I want to highlight the way of connecting it to projection theory. And that's not one of the classical ways. It's actually not super easy to find in the literature.

[0:25:32](https://youtu.be/rChKr66KG-c?t=1532) So it appears in these recent works of Lindenstrauss and Mohammadi and their collaborators, and they're trying-- they're doing the cutting edge of homogeneous dynamics. And the particular problem they're trying to solve is to do, first of all, these higher dimensional things that Margulis and Ratner and Dani were doing. And second of all, instead of having a qualitative theorem like this, to have a quantitative theorem that says, if you're not approximately periodic for some length, then you're quantitatively dense.

[0:26:03](https://youtu.be/rChKr66KG-c?t=1563) And at that point, they ran into some things that many interesting classical techniques could not do, or could not obviously do. And they invented this new technique, this connection to projection theory. But in hindsight, you could also look back at the oldest theorems, the oldest theorems like this one, and you could approach it using projection theory. And it may not be the best approach to this theorem. It's certainly not the only approach to this theorem. But it's a way of looking at it and quite a visual way of looking at what's going on.

[0:26:35](https://youtu.be/rChKr66KG-c?t=1595) So the goal of the class is to try to explain how projection theory could help to understand something like this.

[0:26:51](https://youtu.be/rChKr66KG-c?t=1611) So we're going to do some geometry. So next, we need to think about the geometry on the space G mod gamma and how that geometry interacts with the action.

[0:27:31](https://youtu.be/rChKr66KG-c?t=1651) So geometry of G mod gamma, so first, we'll talk about the right action of G on itself. So if G is in G, there's a right action, which is a map from G to itself. And the right action G of h is h. And you multiply by G on the right side.

[0:28:02](https://youtu.be/rChKr66KG-c?t=1682) So these groups don't commute. So it matters which side you put it on. And actually, you have to be quite careful in this story. So that's a map from G to itself.

[0:28:17](https://youtu.be/rChKr66KG-c?t=1697) Now, we're going to use that to start to put a geometry on G. We're going to have a right invariant metric m.

[0:28:32](https://youtu.be/rChKr66KG-c?t=1712) So how does this work? So what does that mean? So m is going to be a Riemannian metric on G. And for every g in G, we have this right action G. So that's a map from our group with its metric to itself. And this is-- for every g in G, this is an isometry. That's what it means to have a right invariant metric.

[0:29:03](https://youtu.be/rChKr66KG-c?t=1743) Where do they come from? The idea is that we'll put a metric at the origin, and then we'll use R sub g to help define the metric at every other place.

[0:29:21](https://youtu.be/rChKr66KG-c?t=1761) So a lot of what we'll say here works for any group. But to be concrete, I'm going to stick to the group SL2(R).

[0:29:31](https://youtu.be/rChKr66KG-c?t=1771) The tangent space at the origin, so e is the identity. I was using the word origin. I should say identity. e is the identity in the group. The tangent space in the identity of this group is the space of matrices A, B, C, D with a plus d equal to 0.

[0:29:52](https://youtu.be/rChKr66KG-c?t=1792) So this is not a deep factor. So SL2(R) is matrices A, B, C, D, where ad minus bc equals 1. And the identity is a matrix that's sitting in there in R4. So the group is a 3-dimensional manifold in R4. And this is just its tangent space in the usual sense. So that's a tangent space. So now, m0 is going to be a metric on T e G.

[0:30:22](https://youtu.be/rChKr66KG-c?t=1822) And I can specify it by specifying who on an orthonormal basis. And this choice doesn't matter super a lot, but there's a convenient choice. So my orthonormal basis of m0 is going to be 0, 1, 0, 0, which I'll call u. 0, 0, 1, 0, which I'll call u tilde. And then 1 over root 2, 0, 0, minus 1 over 2, which I'll call d.

[0:30:56](https://youtu.be/rChKr66KG-c?t=1856) These root 2's are not actually important. But this orthonormal basis happens to be the orthonormal basis that corresponds to the metric where you just take the metric on R4 and restrict it to this subspace. So now how will we define the metric elsewhere? So now suppose g is in our group. R sub g of the identity is g. It's easy to check from the definition of R sub g.

[0:31:30](https://youtu.be/rChKr66KG-c?t=1890) And therefore, R sub g defines a map from tangent space e of G to the tangent space g of G. And this map is an isomorphism. If we have a metric over here, we use this map to define a metric over there. So now we define mg, which is a metric or inner product, mg on TgG so that Rg goes from TeG comma m0 to TgG comma mg so that that's an isometry.

[0:32:16](https://youtu.be/rChKr66KG-c?t=1936) And then it's an exercise to check that this thing we just defined is right invariant. So that all of these maps are isometries everywhere.

[0:32:53](https://youtu.be/rChKr66KG-c?t=1973) So now, because m is right invariant, we get a metric. It extends to a metric m on G mod gamma.

[0:33:16](https://youtu.be/rChKr66KG-c?t=1996) So how does that work? So we have all of G. We have a projection map pi to G mod gamma. We have a point x in G mod gamma. And x has many preimages. So x is equal to, say, little g times little g-- little h times gamma.

[0:33:53](https://youtu.be/rChKr66KG-c?t=2033) So the preimage of x is these many different points. And I have a metric upstairs. And I'd like to try to define a metric downstairs in a natural way. So I like to define a metric here. So for each of these lifts upstairs, I can take the metric there. And I can use the map pi. And I can produce a metric over here. There's no canonical choice of which of these purple points in the preimage I should use. But because our metric upstairs is right invariant, they all give the same answer. And so there's a coherent choice.

[0:34:26](https://youtu.be/rChKr66KG-c?t=2066) So that gives us a metric on G mod gamma.

[0:34:35](https://youtu.be/rChKr66KG-c?t=2075) So now we have also a left action of G on G. So if g is in G, the left action of g is also a map from G to itself. Left action g of h is g inverse on the left h. So the word left in left action is because the thing that comes from g is over here. The inverse is not super important in our discussion today, but it causes this to behave correctly when you compose elements in the group.

[0:35:12](https://youtu.be/rChKr66KG-c?t=2112) And so it's traditional to do this.

[0:35:23](https://youtu.be/rChKr66KG-c?t=2123) Lg is not an isometry of our group with our right invariant metric. And that's very important. It's very important to our whole discussion to understand, geometrically, how this operation affects this space.

[0:35:54](https://youtu.be/rChKr66KG-c?t=2154) So let's see what happens.

[0:36:01](https://youtu.be/rChKr66KG-c?t=2161) So it's easiest to see what happens around the identity. And you can do it starting at other points, too. So here's the identity. And here's some g. And I have a map, the left action, by g inverse that takes the identity over here. And now, this action, it's not clear from what we've said whether it preserves the metric. In fact, it distorts the metric. But how can we figure out what this mapping is doing to the metric?

[0:36:36](https://youtu.be/rChKr66KG-c?t=2196) Well, our metric was defined to be right invariant. So there's a map going the other direction, the right action by g inverse. That takes g back to the identity. And this one we do understand what it does to the metric. It's an isometry. So if I want to understand what this map did to the metric, it's a reasonable idea to look at this, and then this. That part is an isometry. So all of the distortion of the metric came from this part.

[0:37:08](https://youtu.be/rChKr66KG-c?t=2228) And this composition is going to be a little easier to understand. So what happens if I do this, if I take a left action by g inverse and a right action by g inverse and I apply it to h? These guys commute with each other. We'll see when we write it out. So I don't have to be super careful what order to write them in. Yeah.

[0:37:39](https://youtu.be/rChKr66KG-c?t=2259) AUDIENCE: Is one of these \[INAUDIBLE]?

[0:37:43](https://youtu.be/rChKr66KG-c?t=2263) LAWRENCE GUTH: What's that?

[0:37:44](https://youtu.be/rChKr66KG-c?t=2264) AUDIENCE: Oh, never mind.

[0:37:48](https://youtu.be/rChKr66KG-c?t=2268) LAWRENCE GUTH: Yeah. I guess I should really write it in the other order. Although, it doesn't matter.

[0:37:56](https://youtu.be/rChKr66KG-c?t=2276) So I want you to imagine I started with e, or maybe somebody else near e, I applied left g inverse, and then I applied right g inverse. And we're going to see what happens. So just from the formulas, this is right action of g inverse of gh. And that's g h inverse. So actually, it's something natural and important. It's conjugating. So I'll call this C sub g of h, conjugating g by h.

[0:38:32](https://youtu.be/rChKr66KG-c?t=2312) So conjugating by h then induces a map from the tangent space of e to itself. So conjugating by G is a map from the group to itself. And it sends the identity to the identity. And therefore, it sends the tangent space to the tangent space. And we can see what-- we can write this down. And we can see what it does. And if this thing is distorting the geometry, it means that this thing was distorting the geometry in exactly the same way.

[0:39:11](https://youtu.be/rChKr66KG-c?t=2351) So we could say more than this, but the singular values of Lg inverse are equal to the singular values of Cg.

[0:39:32](https://youtu.be/rChKr66KG-c?t=2372) So what they are depends on g. So I'm going to pick a nice g. And we're actually going to compute them to see how the geometry is being distorted.

[0:40:10](https://youtu.be/rChKr66KG-c?t=2410) So the interesting g's are going to be from those one-parameter subgroups up there. And the one that looks the nicest and where this calculation is the most important is for that diagonal matrix ar. So we're going to compute what that is. So I'm going to look at Car. That goes from the tangent space to itself. So the tangent space is the set of matrices a, b, c, d with a plus b equals 0.

[0:40:44](https://youtu.be/rChKr66KG-c?t=2444) And let's call this matrix M. And so Car of M is ar M ar inverse. That's e to the r, e to the minus r a, b, c, d, e to the minus r, e to the r. So this is a sub r. That's just a definition. This is a sub r inverse, so I switch the minus signs I did this very carefully.

[0:41:19](https://youtu.be/rChKr66KG-c?t=2479) It's not super instructive to watch me do it very carefully. So I'm just going to write down the answer. It's a, d, e to the 2r b, e to the minus 2r c. So this conjugation, what it does is it doesn't change the diagonal. It magnifies this component. And d magnifies that component. That happens to be something we can describe really nicely in the orthonormal basis that I picked.

[0:41:55](https://youtu.be/rChKr66KG-c?t=2515) So Car of u, so it is e to the 2r u. Car of u tilde is e to the minus 2r u tilde, and Car of d is d. This conjugation behaves really nicely in this basis, but you can see that it does not preserve the metric. It stretches that one. It compresses that one. And it preserves that one.

[0:42:29](https://youtu.be/rChKr66KG-c?t=2549) So the conclusion from this is that L ar inverse of u, the length of that is e to the 2r L ar inverse of u tilde has length e to the minus 2r L ar inverse. d has length of 1. And it also preserved angles.

[0:43:00](https://youtu.be/rChKr66KG-c?t=2580) So we just stretched things, but we didn't tilt anything. So L ar inverse of u of u tilde and of d are orthogonal.

[0:43:14](https://youtu.be/rChKr66KG-c?t=2594) So this describes how the left action by this matrix is distorting the geometry of the space.

[0:43:24](https://youtu.be/rChKr66KG-c?t=2604) And we also can make some kind of picture.

[0:43:29](https://youtu.be/rChKr66KG-c?t=2609) So suppose this is a, quote, unquote, "cube" in G, where this is the u direction. This is the u tilde direction. And the direction into the board is the d direction. So what will happen to it when we do this? The u direction gets really long. The u tilde direction gets really short. The other direction is preserved. So it looks kind of like that.

[0:44:04](https://youtu.be/rChKr66KG-c?t=2644) One thing that you can notice from the picture and from the calculation is that while L ar inverse distorts lengths, distorts the metric, it nevertheless preserves the volume.

[0:44:36](https://youtu.be/rChKr66KG-c?t=2676) So that's how the left action distorts the geometry.

[0:44:53](https://youtu.be/rChKr66KG-c?t=2693) And we also have a left action of G on G mod gamma.

[0:45:06](https://youtu.be/rChKr66KG-c?t=2706) And so how does that work? Well, Lg of h gamma is just g inverse h gamma. Because the gamma is on the right, and this guy is on the left, they sort of don't interfere with each other. And so this is an actual group action. So Lg maps G mod gamma to itself. And it is a bijection.

[0:45:38](https://youtu.be/rChKr66KG-c?t=2738) It distorts the metric.

[0:45:43](https://youtu.be/rChKr66KG-c?t=2743) But it preserves the volume.

[0:45:50](https://youtu.be/rChKr66KG-c?t=2750) And maybe I should say, we just computed this carefully for a particular element g, which is these diagonal elements. You could take any g. It doesn't have to be diagonal. It will distort the metric in some different way. Every g is different, but they all preserve the volume. We didn't really prove that, because I just did one example, but it's true. Yeah.

[0:46:12](https://youtu.be/rChKr66KG-c?t=2772) AUDIENCE: Are you able to show that by showing that the second u of t is a one-parameter subgroup? It doesn't change the volume. And do those two one-parameter subgroups generate a G of gamma?

[0:46:25](https://youtu.be/rChKr66KG-c?t=2785) LAWRENCE GUTH: Yeah. So the question was, if we wanted to check that it preserved-- that every g preserves the volume, it would be enough to check a few g's that generate it. And yeah, so if we did the computation with the unipotent 1, that would basically finish the job. So it's not that difficult. We also could just describe the g as a, b, c, d, and then the computation is a little messy, but we could just check once and for all.

[0:46:55](https://youtu.be/rChKr66KG-c?t=2815) There's probably a better way of doing it, but that's not my part of math.

[0:47:09](https://youtu.be/rChKr66KG-c?t=2829) So now we can say--

[0:47:17](https://youtu.be/rChKr66KG-c?t=2837) so now, an important and interesting part of this story is what is the difference between the unitary ones and the diagonal ones. And there are many different answers to this. We could come back to it multiple times, but I'm going to mention one way of looking at it, which will start to get us to the projection theory.

[0:47:48](https://youtu.be/rChKr66KG-c?t=2868) So there's a special feature of unitary matrices that we saw also maybe by coincidence, but it also came up when we were talking about SL2 of Fp, which what we saw there basically is that something cool happens when you conjugate a unitary matrix by a diagonal matrix. And that is the special feature of unitary matrices. And it really helps to understand them.

[0:48:32](https://youtu.be/rChKr66KG-c?t=2912) So what is special about unitary matrices?

[0:48:46](https://youtu.be/rChKr66KG-c?t=2926) So there's a computation that if I take a unitary matrix, and I conjugate it by a diagonal matrix. So I have e to r, e to the minus r.

[0:49:01](https://youtu.be/rChKr66KG-c?t=2941) AUDIENCE: Should it be unipotent?

[0:49:04](https://youtu.be/rChKr66KG-c?t=2944) LAWRENCE GUTH: Unipotent, yeah. Thanks Yes, thank you, unipotent.

[0:49:15](https://youtu.be/rChKr66KG-c?t=2955) So if you multiply this out, you get this. So if you're interested in a unipotent matrix ut, where this t is really large, you can get at it by starting with u1, and conjugating it by one of these diagonal matrices.

[0:49:36](https://youtu.be/rChKr66KG-c?t=2976) So let's say u0t, this is a set of unitary matrices. This is ut. t goes from 0 to t. And so to understand an orbit, our goal is to understand u0 t of x, where t is large.

[0:50:03](https://youtu.be/rChKr66KG-c?t=3003) And because of this slick little formula, we can write u0t x. So let's say we can write it as ar u01 a minus R of x, where e to the 2r is T.

[0:50:34](https://youtu.be/rChKr66KG-c?t=3034) Why is this helpful? So this here is just some point in G mod gamma. It's not the point we started with. It's some other point, but it's just a point. This here is a short unitary orbit.

[0:50:55](https://youtu.be/rChKr66KG-c?t=3055) A short unitary orbit isn't so intimidating to draw. And then all the action is here. And the question is, what does this map do to a short unitary orbit?

[0:51:11](https://youtu.be/rChKr66KG-c?t=3071) So if we make a picture, here's our G mod gamma. Here is our new starting point, a minus R x. Here is our orbit. It's not very long. It doesn't wrap around yet, so it's not that intimidating. That's this guy, short unitary orbit. And now we're going to take this map of G mod gamma to itself. And we're going to apply it. We want to see what happens to this blue thing.

[0:51:45](https://youtu.be/rChKr66KG-c?t=3105) Again, unipotent, fix. Short, unipotent. Thanks.

[0:51:55](https://youtu.be/rChKr66KG-c?t=3115) And we can also make a remark. aR is a subgroup. So aR is like a1 to the R over a little r to the big R over little r. Big R, little r is a natural number. So we don't have to do this all at once. We could take a less dramatic map from G mod gamma to itself, and we repeat that over and over again. We want to see what happens to this thing.

[0:52:26](https://youtu.be/rChKr66KG-c?t=3146) So we are led to try-- so we are led to a goal, which is to try to understand and visualize the action of a little r on G mod gamma. And if we understand that really well, we should be able to understand this orbit.

[0:53:35](https://youtu.be/rChKr66KG-c?t=3215) All right, so how might we try to get a handle on this? Let's look at a fundamental domain. So F is a fundamental domain of G mod gamma. F is a subset of G. And I'll draw it as a cube. It's not literally a cube. G mod gamma is not compact. So F cannot be compact. So it should have a little tail that's going off to infinity. But for what I'm about to say, that's not super important.

[0:54:05](https://youtu.be/rChKr66KG-c?t=3245) Actually, by the way, the whole story also makes sense for gammas that are cocompact. And then this picture would be better. So now, we'd like to understand what a does. So the first step is I'm going to apply aR. And we just thought about what aR does, geometrically, how it distorts the geometry. We were looking at what it does to the metric, but it's very similar to what it does to a unit cube. So this thing is going to be--

[0:54:44](https://youtu.be/rChKr66KG-c?t=3284) so this is aR of F, which is still sitting in the group G. It got longer in one way, shorter in another way. The real thing would have more dimensions than this, but just to give the idea. Now, the next thing that happens is we have to apply the quotient map. So pi is the map from G to G mod gamma.

[0:55:10](https://youtu.be/rChKr66KG-c?t=3310) And when we do that, I'll visualize G mod gamma, again, as this fundamental domain. And what happens is that this thing is going to wrap around the fundamental domain. So it will look something like this. There, it'll go in there.

[0:55:30](https://youtu.be/rChKr66KG-c?t=3330) It's very difficult to draw this well. This map from the fundamental domain to itself is going to be a bijection. So somehow this is going to wrap around. It's going to cover every point exactly once. It's like some sort of a beautiful jigsaw puzzle. I don't know how to draw that. So I'll just do-- I'll just have this come out here, and I'll do a dot, dot, dot. And then I don't know. If we were visualizing, if we had something of interest, eventually, it would be some unipotent orbits, but maybe, initially, we're interested in what happens to that.

[0:56:05](https://youtu.be/rChKr66KG-c?t=3365) So in the first step, I did this. And then in the next step, we see this and this. Somewhere over here, we see this.

[0:56:25](https://youtu.be/rChKr66KG-c?t=3385) Cool. So yeah.

[0:56:29](https://youtu.be/rChKr66KG-c?t=3389) AUDIENCE: Why does the math have to be a bijection? Why can't the projection map not be injective?

[0:56:36](https://youtu.be/rChKr66KG-c?t=3396) LAWRENCE GUTH: Yeah. So the question is, why is this map a bijection? So we were saying-- so this-- I guess it's actually there.

[0:56:53](https://youtu.be/rChKr66KG-c?t=3413) So we said earlier that the left action of G, G mod gamma goes to itself is a bijection. And if we believe that, then if we start with the fundamental domain, we have one representative of each point in G mod gamma. And then we end up back in G mod gamma. We should cover everything exactly once if this is a bijection. Now, why was this a bijection? So what is this map? It's the map h gamma goes to g inverse h gamma.

[0:57:27](https://youtu.be/rChKr66KG-c?t=3447) So why is this a bijection?

[0:57:40](https://youtu.be/rChKr66KG-c?t=3460) So it's injective. That means that if h1 g inverse h1 gamma is equal to g inverse h2 gamma, then h1 gamma equals h2 gamma. That's what it means for it to be injective. So is this clear? Yeah, you multiply on both sides by g. Surjective, that means that for every h2 in G, there exists h1 in G, so that g inverse h1 gamma is h2 gamma.

[0:58:39](https://youtu.be/rChKr66KG-c?t=3519) So yeah, we set these equal, and solve for g.

[0:58:43](https://youtu.be/rChKr66KG-c?t=3523) AUDIENCE: I guess that causes the invertibility of the \[INAUDIBLE].

[0:58:46](https://youtu.be/rChKr66KG-c?t=3526) LAWRENCE GUTH: Yeah. Yeah.

[0:58:57](https://youtu.be/rChKr66KG-c?t=3537) I have to say, I went through this, too. When I drew these pictures, I asked myself, is this actually going to fold up and cover everything exactly once? And then I was like, is it really bijective? Yeah, it should happen.

[0:59:19](https://youtu.be/rChKr66KG-c?t=3559) I guess you can make easier fun examples that are more linear. So you just have a matrix acting on R2 mod Z2, or something like that. And then the fundamental domain is a square. And you can draw what happens to it. And you can look up-- so this was inspired by that example. They called that the cat map, a particular map. And instead of a smiley face, they draw a cat. They have better artistic skills than me. You can google that, and see some nice pictures like this. OK, cool.

[0:59:49](https://youtu.be/rChKr66KG-c?t=3589) So here, this thing, Lg Ll with this diagonal action, you've broken it into two steps. And out of these two steps, I find this one quite confusing. It's difficult to write this down sufficiently accurately, and then to say what happened here. And did this come back over here, over there when it wrapped? Where did it wrap? All of that seems very intimidating to me, so very difficult to write down in a clean way. But this first step is not so bad.

[1:00:24](https://youtu.be/rChKr66KG-c?t=3624) And so we're going to try to understand the first step very well and use that to see how much we can figure out about this map Lg acting on our homogeneous space.

[1:00:39](https://youtu.be/rChKr66KG-c?t=3639) Now, the object that we want to follow is not a smiley face, but a unipotent orbit.

[1:00:52](https://youtu.be/rChKr66KG-c?t=3652) Let's see here.

[1:01:02](https://youtu.be/rChKr66KG-c?t=3662) I guess we can erase all of it.

[1:01:25](https://youtu.be/rChKr66KG-c?t=3685) So I erased something that is useful for us. So recall, we worked out that the left map by ar inverse, we found its singular values and singular vectors. So in the u direction, the length of this is e to the 2r. And in the u tilde direction, the length is e to the minus 2r. And in the diagonal direction, the length is 1.

[1:02:00](https://youtu.be/rChKr66KG-c?t=3720) So this is if we're thinking of L ar inverse from TeG to TarG.

[1:02:10](https://youtu.be/rChKr66KG-c?t=3730) And this little u, this is also the tangent direction of our group u.

[1:02:23](https://youtu.be/rChKr66KG-c?t=3743) So that group is going to get stretched, which is the same thing we saw in the computation over there.

[1:02:37](https://youtu.be/rChKr66KG-c?t=3757) Step one, so if we start with a single unipotent orbit, it starts at some point like this. So that's u 0, 1 of some point x. What happens when we apply Lar?

[1:03:00](https://youtu.be/rChKr66KG-c?t=3780) Well, this thing is going to get tall and skinny.

[1:03:09](https://youtu.be/rChKr66KG-c?t=3789) And the direction that got stretched is the direction of this orbit. So now we're going to have a long orbit. And the height of this thing is e to the 2r. Then we're getting a quotient. So when we quotient, I don't know exactly where it's going to go.

[1:03:31](https://youtu.be/rChKr66KG-c?t=3811) But it will go along-- it'll go out the top and come in the bottom somewhere. And then we'll have a bunch of these. So we'll have e to the 2r u orbits of length 1.

[1:03:51](https://youtu.be/rChKr66KG-c?t=3831) And I have no idea where they are. So we haven't made any progress yet on our main question, which is to see if these are distributing kind of evenly. But now we can start again. So let me move this over here.

[1:04:07](https://youtu.be/rChKr66KG-c?t=3847) So now we have a bunch of unipotent orbits. And we're going to stretch them out.

[1:04:18](https://youtu.be/rChKr66KG-c?t=3858) Now, what happens when we stretch them out this time?

[1:04:30](https://youtu.be/rChKr66KG-c?t=3870) So each one of them got longer. It's going to go from the bottom to the top. But also, there's also some compression, which is kind of in this direction. And that could potentially cause some orbits to get closer together, or not. And this is the part that we have to understand really carefully.

[1:04:57](https://youtu.be/rChKr66KG-c?t=3897) AUDIENCE: So this box is some small box \[INAUDIBLE]?

[1:05:02](https://youtu.be/rChKr66KG-c?t=3902) LAWRENCE GUTH: Yeah. So this--

[1:05:04](https://youtu.be/rChKr66KG-c?t=3904) AUDIENCE: \[INAUDIBLE]

[1:05:07](https://youtu.be/rChKr66KG-c?t=3907) LAWRENCE GUTH: Yeah. That's right. That's right. So this box here is a fundamental domain. So it's scale 1. And sure, it could be around the identity. The thing I most want to convey in this class is a feeling for how this map is like twisting and distorting these curves. And so I'm going to try to make a really big picture of it there. I also brought with me some props.

[1:05:58](https://youtu.be/rChKr66KG-c?t=3958) So I'm going to visualize our fundamental domain or maybe a piece of it as a cylinder. And I'm going to draw some unipotent orbits in different colors. And they all go from the bottom to the top.

[1:06:21](https://youtu.be/rChKr66KG-c?t=3981) Maybe I'll draw three of them.

[1:06:31](https://youtu.be/rChKr66KG-c?t=3991) Now, what happens to this picture? So first of all, it's going to stretch that way. That part's not that hard to imagine. It would go off the board. So I'm not going to emphasize that. Then each one of these slices, so we have some slices that say this is t equals 0 up to t equals 1-- other t's in the middle. Each one of these slices, one direction is going to be squished, and the other direction is going to stay the same. So let's say that at t equals 1 cross-section of this picture, I see a triangle that looks like this.

[1:07:11](https://youtu.be/rChKr66KG-c?t=4031) And there's going to be some direction that gets compressed. When I apply Lar inverse, that direction gets compressed. The output is going to look like so. And that has been actually worked out so that this blue and yellow get kind of squished together. And orange is still there. And at t equals 0, if I look at the bottom disk, I have the same picture.

[1:07:47](https://youtu.be/rChKr66KG-c?t=4067) And there's some direction that's going to be compressed. And it's not the same direction. I do somewhat arbitrarily that one.

[1:08:05](https://youtu.be/rChKr66KG-c?t=4085) If you compress that way, that will look sort of like so. And the blue point will be over here. And the yellow and orange point will be somewhat close together.

[1:08:19](https://youtu.be/rChKr66KG-c?t=4099) And as you look at all the intermediate heights, you'll have a similar picture. But this compression direction is going to be changing continuously as you go up.

[1:08:46](https://youtu.be/rChKr66KG-c?t=4126) So I'm not sure if this will be helpful, but I brought several props to try to illustrate this important process. So if you imagine that you have a bunch of unipotent orbits in your homogeneous space like so, and we apply a, first of all, they will stretch this way. And second of all, there will be some compression. So maybe at the top, they might get smooshed together this way. And at the bottom, they might get smooshed together a different way, sort of like this.

[1:09:22](https://youtu.be/rChKr66KG-c?t=4162) Or if this piece of Play-Doh is this cylinder here, what's going to happen to it when we apply the map? Well, it's going to get taller. And it's going to get squished. And maybe the top will get squished that way. But the bottom will get squished that way. And in the middle, it'll get squished some direction in the middle. And that will-- if I had done it carefully.

[1:09:52](https://youtu.be/rChKr66KG-c?t=4192) I practiced it at home. It produces something that's sort of like a helix.

[1:10:12](https://youtu.be/rChKr66KG-c?t=4212) So let me say briefly why this is relevant.

[1:10:20](https://youtu.be/rChKr66KG-c?t=4220) And then if we have time, if we feel like it, we could try to do some computations with matrices to check.

[1:10:27](https://youtu.be/rChKr66KG-c?t=4227) So why is this relevant? Well, if you had a situation like this, then--

[1:10:38](https://youtu.be/rChKr66KG-c?t=4238) yeah, OK, let's go back up here. So this guy is really tall. We're going to cut it into a sequence of pieces that each have height 1. And each one of these pieces is going to fold back in.

[1:10:59](https://youtu.be/rChKr66KG-c?t=4259) So over here, there is an unused color. There's a purple piece over here.

[1:11:11](https://youtu.be/rChKr66KG-c?t=4271) That purple piece is going to fold back in over here.

[1:11:21](https://youtu.be/rChKr66KG-c?t=4281) Can you see that? And then there are a bunch of other pieces of different colors. Those other pieces will somehow get slotted in in an order that I don't understand at all, but they'll all go somewhere.

[1:11:33](https://youtu.be/rChKr66KG-c?t=4293) Now, inside of this purple piece, so over here, we have a yellow orbit, a blue orbit, and an orange orbit. Let's just focus on those three. They are all going to pass through here. And we could have the blue orbit and the orange orbit and the yellow orbit all kind of spread out from each other. Or it could be that they are smooshed together close to each other. Anyway, whatever happens here, this is just slotted in over here.

[1:12:06](https://youtu.be/rChKr66KG-c?t=4326) So that will tell us how they're spaced over here.

[1:12:15](https://youtu.be/rChKr66KG-c?t=4335) When we have something like this, that will produce two unit length unipotent orbits over here in this purple thing that are really close to each other. And that phenomenon is the enemy of the unipotent orbit spreading out. Unipotent orbit is definitely going to be really long, but it may not spread out because it may have many segments that are really close to each other.

[1:12:43](https://youtu.be/rChKr66KG-c?t=4363) Let me scribe. So to prove that u0t of x spreads out, we want to prove that coincidences of this type are rare.

[1:13:11](https://youtu.be/rChKr66KG-c?t=4391) And why are they rare? Well, they're rare because this compression angle is rotating smoothly as we vary t. And so even though there may be one projection that smooshes a lot of things together, we have learned in this course that if you look at the projection in all different directions, most of them do not smoosh things together very much. And that forces this unipotent orbit to spread out.

[1:14:02](https://youtu.be/rChKr66KG-c?t=4442) AUDIENCE: I'm sorry. I'm a little confused. Why doesn't the direction get compressed and changed? From that picture, after we applied LaR inverse, it's always long. \[INAUDIBLE] another direction.

[1:14:28](https://youtu.be/rChKr66KG-c?t=4468) LAWRENCE GUTH: Yeah, OK, all right. So the question is, why does this happen? And in fact, there was a point that the picture that I drew up there sort of makes it look like it doesn't happen. Correct?

[1:14:48](https://youtu.be/rChKr66KG-c?t=4488) So in my picture, this is not exactly accurate, but I drew it as though this map was just a linear map that takes a cube to a rectangular solid. And I drew it as though-- well, I tried to show that this-- so these unipotent orbits are more or less vertical. So here's what would work. So if the unipotent orbits were literally vertical and if this map was literally a linear map, then this would not happen. The compression direction would be the same at every height.

[1:15:21](https://youtu.be/rChKr66KG-c?t=4521) So I think you have to pick--

[1:15:29](https://youtu.be/rChKr66KG-c?t=4529) this Lie group is like it's a submanifold of R4. But if you want to represent it in coordinates in three dimensions, you have to make a choice. There's not a completely canonical choice. So we could make the choice that the unipotent orbits were literally vertical. And if we did that, this picture would not be right anymore. If we did that, then this would have a bit of a twist. This would be more-- this would be like a helix thing.

[1:16:05](https://youtu.be/rChKr66KG-c?t=4565) Or another way of looking at it is I think we could choose coordinates in such a way that this map was pretty close to a linear map, but then these unipotent orbits would not be vertical. And instead, they would kind of spiral a little bit as you went up. And so the angle between two unipotent guys would be changing.

[1:16:33](https://youtu.be/rChKr66KG-c?t=4593) Now, at some point, why do you believe that? So at some point, we should do a computation.

[1:16:45](https://youtu.be/rChKr66KG-c?t=4605) I don't think I'll do it in the last 2 minutes, but I'll just say what you should do.

[1:16:52](https://youtu.be/rChKr66KG-c?t=4612) So we figured out which direction is compressing. We could have written it down a little bit more clearly. But if you trace through our thoughts today, at any point in the homogeneous space, if you ask which is the tangent direction that's going to get compressed when you apply Lar, we have a formula for it. And then we can also say if you have two nearby unipotent orbits, and down here at the bottom there's a little vector between them, as you go up the orbit, what happens to that vector?

[1:17:30](https://youtu.be/rChKr66KG-c?t=4650) There's a formula for that, too. If you compare these two formulas, you will see that they don't match. So if this direction initially was the compression direction, compression direction, if you follow it up, you don't get the compression direction up here. And you can compute it all with matrices.

[1:17:54](https://youtu.be/rChKr66KG-c?t=4674) Yeah.

[1:17:55](https://youtu.be/rChKr66KG-c?t=4675) AUDIENCE: Is this just kind of saying, if there's a curvature tensor that's non-zero from the, I guess, one of the two one-parameter groups?

[1:18:06](https://youtu.be/rChKr66KG-c?t=4686) LAWRENCE GUTH: Yeah. The question is, is this saying that there's some kind of curvature tensor that doesn't vanish? Maybe. It's not the Riemann curvature tensor, but it might be something in that spirit. Yeah.

[1:18:21](https://youtu.be/rChKr66KG-c?t=4701) AUDIENCE: The theorem before is really two cases. One, this periodic number, the picture kind of makes it seem like it wasn't structured. It's not periodic. Is there something I'm missing?

[1:18:35](https://youtu.be/rChKr66KG-c?t=4715) LAWRENCE GUTH: Yeah, great question. The question was, this appears to be a proof sketch that every orbit of the unipotent group is evenly distributed. And we saw at the beginning that that is not true. So what did what did we do wrong?

[1:19:02](https://youtu.be/rChKr66KG-c?t=4742) All right, so this orbit, we can write this way. Actually, so let me say the shortest answer to your question. The key thing is that the cusp of the homogeneous space is really important here. And we were neglecting the cusp. So the cusp produces a piece of the fundamental domain that's really skinny. And it doesn't look like the rest of it. And when something's in the cusp, it doesn't apply. And so, actually, Hedlund proved another theorem, which is that if you take a cocompact subgroup gamma, then in G mod gamma, every unipotent orbit is dense.

[1:19:42](https://youtu.be/rChKr66KG-c?t=4782) And this is a sketch of the proof of that theorem. But now what goes wrong is that this a to the minus Rx could be very far in the cusp, and then our whole analysis is wrong. So what we've roughly shown is that either our orbit is very well distributed, or a minus Rx is deep in the cusp. And if you unwind what that means about x, it means that it's very close to being a periodic orbit.

[1:20:23](https://youtu.be/rChKr66KG-c?t=4823) Cool. So we'll talk a little more about homogeneous dynamics on Thursday. I'm not sure exactly what we'll say. Maybe we'll fill in some things.
