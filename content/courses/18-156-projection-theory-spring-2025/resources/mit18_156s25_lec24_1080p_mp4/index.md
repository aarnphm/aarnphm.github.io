---
title: 'lecture 24: sharp projection theorems, part 3: combining different scales'
description: 'Ren and Wang proved the full Furstenberg conjecture by applying the AD regular case at many different scales.'
id: 'mit-18-156-spring-2025-resource-resources-mit18-156s25-lec24-1080p-mp4'
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
    n: 24,
    youtube: 'h0Ln82BR4R0',
    duration: 4638,
    cues: 1159,
    captions: 'static_resources/mit18_156s25_lec24_1080p_captions.vtt',
    captionsId: 'mit18_156s25_lec24_1080p_captions_vtt',
    transcript: 'static_resources/mit18_156s25_lec24_1080p_transcript.pdf',
    transcriptId: 'mit18_156s25_lec24_1080p_transcript_pdf',
  }
layout: 'resource'
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
---

[0:00:12](https://youtu.be/h0Ln82BR4R0?t=12) LAWRENCE GUTH: OK well, today is our last day together. I was remembering the first day at the beginning I said to everybody I'm excited about projection theory and these basic problems we just solved in the last couple years. And we're trying to describe the ideas in this. And the beginning of the class I say all these things I'm excited to do. And I wonder, am I going-- are we going to manage to do this?

[0:00:37](https://youtu.be/h0Ln82BR4R0?t=37) Yeah, anyway. Now it's at the end and we've been working steadily and done some of it.

[0:00:45](https://youtu.be/h0Ln82BR4R0?t=45) OK.

[0:00:49](https://youtu.be/h0Ln82BR4R0?t=49) So today we'll try to say the-- finish the story of proving the sharp projection theorems and to say the last couple of ideas that go into it.

[0:01:03](https://youtu.be/h0Ln82BR4R0?t=63) So the first idea goes back to the beginning of the class. It goes back to using the Fourier method and getting a little bit more out of it than we had realized at that time.

[0:01:19](https://youtu.be/h0Ln82BR4R0?t=79) And then after that-- so this thing, I'll do by itself. It's like remembering the Fourier method and seeing how much we can learn from it. Then I'll remember what was the whole big theorem that we wanted to prove, and what are all the things we knew about it. And the last idea in the class is combining information from many different scales, which allows us to combine all of the previous ideas into one big argument. Master argument.

[0:01:52](https://youtu.be/h0Ln82BR4R0?t=112) And this way of combining scales has only been-- I would say has only been around for five years or so, and is also something-- it's not that complicated. And I think also something that might be of value to take with you away from the class. OK.

[0:02:20](https://youtu.be/h0Ln82BR4R0?t=140) All right. So part 1 is the well-spaced case and the Fourier method.

[0:02:38](https://youtu.be/h0Ln82BR4R0?t=158) So all of these things that were-- these sharp projection theorems, there's some kind of geometric measure theory analog of the Szemerédi-Trotter theorem. So for reference, let's remember first the Szemerédi-Trotter theorem.

[0:02:55](https://youtu.be/h0Ln82BR4R0?t=175) And here's the version of it that will be relevant for us. So if E contained in R2 is a set of N points.

[0:03:07](https://youtu.be/h0Ln82BR4R0?t=187) And L sub r of E is the set of lines, little l. So that little l intersect E is at least r. Then the question is, how big could this be in terms of N and r? And the conclusion is that the size lr of E is bounded by n squared. Actually, let me do a big R to match some other notation.

[0:03:38](https://youtu.be/h0Ln82BR4R0?t=218) N squared over R cubed plus N over R. Is N squared over R cubed comes from the grid example. And this N over R comes from a very simple example, where you just draw a few lines and you put R points on each of them.

[0:04:00](https://youtu.be/h0Ln82BR4R0?t=240) OK. And we're trying to investigate, to what extent this continues to hold if we replace our pure points by some little balls and by our pure lines by some thin tubes? And here is one clean version of that, that if the points are quite spread out, they're what's called well-spaced, then this holds almost as real.

[0:04:30](https://youtu.be/h0Ln82BR4R0?t=270) So this is a theorem of Hong Wong, Noam Solomon, and me from around 2018 or something. So E contained in R2 is a set of N delta balls.

[0:04:47](https://youtu.be/h0Ln82BR4R0?t=287) E is contained in the unit ball. AND they're well spaced. They're as spread out as possible. So they obey the condition that if you take E and you intersect it with a ball of radius N to the minus 1/2, then that's at most around 1. So in other words, if I take a unit square and I subdivide it into squares of side length N to the minus 1/2-- that's been chosen so that there are N squares in this picture.

[0:05:22](https://youtu.be/h0Ln82BR4R0?t=322) And then I put 1 or roughly 1 point in each of these squares at one delta ball, then this is what? This is our what our set looks like. So this is a well-spaced set. Yeah.

[0:05:39](https://youtu.be/h0Ln82BR4R0?t=339) AUDIENCE: So if the size of E intersect B, is that supposed to be a delta ball covering number?

[0:05:44](https://youtu.be/h0Ln82BR4R0?t=344) LAWRENCE GUTH: Yeah. Let's make that. Yeah, that's right. Thank you. That's a delta ball covering number. OK. All right. And then there's one more hypothesis, which is OK, so then we could say what TR of E is the set of delta tubes T, so that T intersect E delta is at least R. And here, if you take such a delta tube and you wiggle it imperceptibly, you will have another such delta tube.

[0:06:19](https://youtu.be/h0Ln82BR4R0?t=379) And we want to count those as being the same. So I'll put here essentially distinct.

[0:06:28](https://youtu.be/h0Ln82BR4R0?t=388) OK. Now there's one other hypothesis, which is that R is not too small, which I'll explain in a minute. The hypothesis says R is at least a little bit bigger than delta times E delta. And then the conclusion is almost the same as before, that TR of E is less same same squared over R cubed. This two tildes here means that it's hiding something like a delta to the epsilon or something like that.

[0:07:04](https://youtu.be/h0Ln82BR4R0?t=424) But it looks almost the same as this. OK, let's compare these two theorems. One difference is that in the first theorem, we had an extra term here, which I have not written in the second theorem. And the reason is that this second term is relevant when R is bigger than the square root of N. So the second term, this dominates if and only if R is bigger than square root of N.

[0:07:38](https://youtu.be/h0Ln82BR4R0?t=458) But in the well space case, it's not possible for a tube to have more than square root of N points on it, so this case never arises. OK. So that's why we don't have that. The other difference is that we added this lower bound on R. So let me explain why we did that. So remark. An average for a random delta tube T, the expected size of T intersect E, that's covering, that would be delta times E delta.

[0:08:20](https://youtu.be/h0Ln82BR4R0?t=500) This is for each of these delta points, each of these little delta balls, there are this many of them, the probability that a random delta tube intersects that point is about delta.

[0:08:31](https://youtu.be/h0Ln82BR4R0?t=511) OK. SO if you just take any old tube, you'll have that many. You'll hit that many delta balls from E-- tubes that hit this many delta balls of E are not special. Any old tube will do. So if R takes that value, this upper bound isn't true at all. This could be way bigger than that because it could be basically all the tubes. But rather remarkably-- there's a rather remarkably sharp phase transition that you increase R just a little bit beyond this threshold, and then you get a bound that matches Szemerédi-Trotter.

[0:09:10](https://youtu.be/h0Ln82BR4R0?t=550) OK. So that is the statement of the theorem. And it says in a nutshell that Szemerédi-Trotter basically continues to hold if you take a well-spaced set, which is one of the cases we mentioned last time.

[0:09:27](https://youtu.be/h0Ln82BR4R0?t=567) Any questions or comments about the statement?

[0:09:37](https://youtu.be/h0Ln82BR4R0?t=577) So I'm going to do a proof sketch of this theorem.

[0:09:48](https://youtu.be/h0Ln82BR4R0?t=588) So let me first mention the tools that will go into it. There's the Fourier method. There's double counting. And then there's-- I guess I'll call this playing with different scales.

[0:10:08](https://youtu.be/h0Ln82BR4R0?t=608) The theme of this last class is playing with different scales, which is a simple sounding and old idea, which nevertheless, people recently have found a lot of new ways of doing it. OK.

[0:10:26](https://youtu.be/h0Ln82BR4R0?t=626) All right. So let me start with the Fourier method.

[0:10:32](https://youtu.be/h0Ln82BR4R0?t=632) Let me label this list of hypotheses. Hypotheses.

[0:10:43](https://youtu.be/h0Ln82BR4R0?t=643) And Fourier method. OK. So it's been a while. In lectures 3 and 4 of this class, we developed a Fourier analytic method to estimate stuff like this. And if we had kept having problem sets, and if I had been really clever, I would have had you remember it on the last problem set, getting ready for this lecture. Anyway. If you take the method exactly the way we did it back at the beginning of the class, say that if we have the hypotheses, then the size of TR of E will be bounded by delta--

[0:11:28](https://youtu.be/h0Ln82BR4R0?t=688) OK, I have to actually check.

[0:11:37](https://youtu.be/h0Ln82BR4R0?t=697) Delta inverse E delta R to the minus 2. I considered redoing this in the last class to remind us all, because I don't expect it really to have it on the tip of your brain. But I decided it was not that much fun for a thing to do in the last class, and also that I had to manage the time to explain the other stuff. So I'd like to take this as a black box, that it's something you can do with the tools that we've already have.

[0:12:10](https://youtu.be/h0Ln82BR4R0?t=730) OK.

[0:12:16](https://youtu.be/h0Ln82BR4R0?t=736) Now how does this compare to our goal? This E delta, by the way, is N. So this is delta inverse N R to the minus 2.

[0:12:30](https://youtu.be/h0Ln82BR4R0?t=750) Well, it depends on how delta and N and R are related to each other. But there's a particular case where it just gives exactly the right thing. In a particular case is that if R is the smallest that it's allowed to be-- so let's say it is delta to the minus epsilon delta E delta.

[0:13:04](https://youtu.be/h0Ln82BR4R0?t=784) So in this case, if I solve for the delta inverse, delta inverse would be around N over R. And then I plug in here. And I get the one that this is N over R, is N squared over R cubed. And this is Szemerédi-Trotter bound.

[0:13:34](https://youtu.be/h0Ln82BR4R0?t=814) So this Fourier method actually gives the right answer in an interesting case.

[0:13:42](https://youtu.be/h0Ln82BR4R0?t=822) But then if you make R bigger, then if you do this computation, this ends up being bigger. And this will be bigger than N squared over R cubed.

[0:14:01](https://youtu.be/h0Ln82BR4R0?t=841) OK. Now another interesting basic thing about this that is perhaps surprising is that as delta-- if delta gets bigger, the right-hand side gets smaller. So as we make our tubes thicker, we actually get a better bound.

[0:14:26](https://youtu.be/h0Ln82BR4R0?t=866) So if R is bigger-- this is the smallest R could be. So the other options that R is bigger than this. So delta to the minus epsilon delta E delta is less than R. And let's say that R is less than the square root of N. Because as we mentioned earlier, R can't be any bigger than the square root of N. OK. Then I'm going to set a scale rho so that rho times N is around R. Rho times N is around R. So how big is rho?

[0:15:15](https://youtu.be/h0Ln82BR4R0?t=915) Well, rho is going to be R over N. So it will be smaller than N to the minus 1/2. But it will be bigger than delta. Because delta N was smaller than R. And I'm going to increase delta to rho so that rho N will be around R. So rho will be around there. And then we're going to study E rho, which is the rho neighborhood of E. So let's make a picture.

[0:15:55](https://youtu.be/h0Ln82BR4R0?t=955) Here is the picture of our well-spaced set E. And then we're going to have a scale rho, which is smaller than this scale N to the minus 1/2 but bigger than delta. And we'll look at that guy, E rho.

[0:16:17](https://youtu.be/h0Ln82BR4R0?t=977) And originally, we're trying to find--

[0:16:23](https://youtu.be/h0Ln82BR4R0?t=983) our original problem is to understand delta tubes that hit a lot of balls. So this might be one.

[0:16:36](https://youtu.be/h0Ln82BR4R0?t=996) This is T delta tube and TR of E delta. E delta is just E, but I put the delta to remind us. But now that we have these thicker guys, we could study--

[0:16:53](https://youtu.be/h0Ln82BR4R0?t=1013) I could study a rho tube that contains a lot of rho balls. So this might be T rho. That might be a tube that intersects E rho a lot. Well, maybe for the same R, but also maybe for different R's, we might be interested.

[0:17:16](https://youtu.be/h0Ln82BR4R0?t=1036) So now let's make a definition.

[0:17:22](https://youtu.be/h0Ln82BR4R0?t=1042) So TR of-- TR tilde of E rho, this is the set of rho tubes T rho. So that if I take T rho intersect it with E rho and I take its rho covering number, that's at least R tilde.

[0:17:47](https://youtu.be/h0Ln82BR4R0?t=1067) And now the same argument as before gives us a bound for this thing. So if we have our hypotheses and if R tilde--

[0:18:01](https://youtu.be/h0Ln82BR4R0?t=1081) let's see.

[0:18:05](https://youtu.be/h0Ln82BR4R0?t=1085) Well, let me not keep track of it. If we have our hypotheses and R tilde is big enough, then, yeah, I'll just-- yeah, here we can put R tilde as at least as big as R, then TR tilde of E rho will be bounded by--

[0:18:32](https://youtu.be/h0Ln82BR4R0?t=1112) Well, we'll have rho inverse times E delta times R to the minus 2-- times R tilde to the minus 2.

[0:18:48](https://youtu.be/h0Ln82BR4R0?t=1128) And this rho inverse is N over R. So that's N squared over R times R tilde squared. Just by thickening the tubes, we get these bounds. And let me mention an important special case. So e.g, TR of E rho is bounded by N squared over R cubed.

[0:19:34](https://youtu.be/h0Ln82BR4R0?t=1174) All right.

[0:19:37](https://youtu.be/h0Ln82BR4R0?t=1177) So as it's-- it's a surprising thing that as we increase rho, we get a better and better upper bound for this.

[0:19:53](https://youtu.be/h0Ln82BR4R0?t=1193) Now we can't increase it. There's a limit to how much we can increase it. Because we do need--

[0:20:01](https://youtu.be/h0Ln82BR4R0?t=1201) we do need this condition.

[0:20:06](https://youtu.be/h0Ln82BR4R0?t=1206) So this gives some upper bound for rho. But taking rho as close as you can to that upper bound is a good idea. And when we make rho that big, then we get this.

[0:20:25](https://youtu.be/h0Ln82BR4R0?t=1225) OK. Now have we proven the theorem?

[0:20:31](https://youtu.be/h0Ln82BR4R0?t=1231) So what I got-- yeah.

[0:20:33](https://youtu.be/h0Ln82BR4R0?t=1233) AUDIENCE: Yeah. Just a question about the bounds there. You apply the same Fourier method?

[0:20:38](https://youtu.be/h0Ln82BR4R0?t=1238) LAWRENCE GUTH: Yes.

[0:20:40](https://youtu.be/h0Ln82BR4R0?t=1240) AUDIENCE: But will the E delta state not become a rho?

[0:20:47](https://youtu.be/h0Ln82BR4R0?t=1247) LAWRENCE GUTH: Right. So it would be E rho. But you can maybe see from the picture. Where did we introduce E rho? So E rho is equal to E delta, is equal to N. Now that's a special-- that's where we're using that our set is well-spaced right. So that's a great thing to-- So the so the question was, should this should be a rho instead of a delta?

[0:21:18](https://youtu.be/h0Ln82BR4R0?t=1278) And how does that affect things? And the answer is yeah, great point. This should be a rho instead of a delta. But for our set, it doesn't change anything.

[0:21:30](https://youtu.be/h0Ln82BR4R0?t=1290) So if you look at our set, our set has only one delta ball in each N to the minus 1/2 ball. So when you thicken them a little bit, the number of balls you see doesn't change. It's still N. So that is where we are using this hypothesis.

[0:21:56](https://youtu.be/h0Ln82BR4R0?t=1316) Yeah, great point.

[0:22:00](https://youtu.be/h0Ln82BR4R0?t=1320) OK. So when I first saw this, I thought that we were done. But it's not quite true. So our conclusion in the theorem is that we're going to try to estimate how many of these guys there are, thin tubes, delta tubes that have R balls in them. And we've counted the number of thick tubes that have R balls in them. Now, every thin tube with R balls, if you thicken it, you get a thick tube with at least R balls.

[0:22:35](https://youtu.be/h0Ln82BR4R0?t=1355) But inside of a thick tube with R balls, there may be many thin tubes with R balls, and we've only counted them once. So we do-- we still have an issue. That's not super difficult, but it's important to recognize it.

[0:22:56](https://youtu.be/h0Ln82BR4R0?t=1376) So issue 1, rho tube may contain many tubes of TR of E delta. So what would that scenario look like? Well, we have our rho tube here. And inside we would have a bunch of delta balls.

[0:23:30](https://youtu.be/h0Ln82BR4R0?t=1410) So at least R of them, but maybe more than R of them. So let's say that T rho intersected with E, so delta, that would be the same as T rho intersected with E rho. Let's say that it's R tilde. Which is at R. Might be R but it might also be bigger. And then inside of here we may have many delta tubes that each contain R. OK.

[0:24:22](https://youtu.be/h0Ln82BR4R0?t=1462) So now we have some cousin of our original problem. We have R tilde delta balls. And we want to count how many R-rich tubes there are. We don't know as much about how they are spaced, but also we don't need to have such an incredible bound. So by induction, we can assume-- we can reduce to the case that each T and TR of E delta has many delta balls at both ends.

[0:25:17](https://youtu.be/h0Ln82BR4R0?t=1517) So what I mean by that is here's a tube. So the tube, it has a bunch of delta balls in it. So it means that there are a bunch at one end and a bunch at the other end. Maybe also a bunch in the middle. Maybe it's better to draw what it's not.

[0:25:46](https://youtu.be/h0Ln82BR4R0?t=1546) What it's not is like they're just all on the left.

[0:25:53](https://youtu.be/h0Ln82BR4R0?t=1553) OK. So in a nutshell, how would we do this? If this was the situation that all of the delta balls were actually in a much shorter tube, well, then in my original picture, I could cut my set E into separate pieces and just look at R-rich tubes on each of those pieces. And that could do some induction. OK. So then we do a double counting argument.

[0:26:28](https://youtu.be/h0Ln82BR4R0?t=1588) I'm going to do--

[0:26:32](https://youtu.be/h0Ln82BR4R0?t=1592) so we're going to count just pairs of delta balls. And so I'll have T of E delta intersect tubes in T rho. So the number of yellow tubes in this picture. Times R squared. So I take a tube and I pick two points in the tube kind of near opposite ends.

[0:27:03](https://youtu.be/h0Ln82BR4R0?t=1623) And for any two points, there is only one tube that joins them, so this should be bounded by R tilde squared. Double counting.

[0:27:20](https://youtu.be/h0Ln82BR4R0?t=1640) So we're counting-- we're counting pairs T, X1, X2, where T is in this set and X1 and X2 are in T and they're near opposite ends.

[0:27:44](https://youtu.be/h0Ln82BR4R0?t=1664) So how many are there? Well, for each tube, there are R squared choices of X1 and X2. On the other hand, there are only R tilde squared choices for X1 and X2. And for each X1 and X2, there's only one choice of T. So this is bounded by this.

[0:28:07](https://youtu.be/h0Ln82BR4R0?t=1687) OK. So double counting and the Fourier method are the two foundational approaches that we did at the beginning of the class. This one was short enough that I thought I could recall it for us.

[0:28:23](https://youtu.be/h0Ln82BR4R0?t=1703) OK.

[0:28:30](https://youtu.be/h0Ln82BR4R0?t=1710) All right. So now TR of E delta is bounded by maybe sum on R tilde bigger than R dyadic. We're going to look at TR tilde of E rho. But then we'll multiply by R tilde over R squared.

[0:29:01](https://youtu.be/h0Ln82BR4R0?t=1741) All right. Why? So every R-rich delta tube, if you thicken it up to be a rho tube, it will be an R tilde-rich rho tube. And R tilde will be at least R. So they're all in here somewhere. And in each one of these guys, there's at most this many R-rich delta tubes. This is bounded by this. And then if we plug in our bounds, everything will work. So this is bounded by N squared over R times R tilde squared.

[0:29:40](https://youtu.be/h0Ln82BR4R0?t=1780) And then we have R tilde squared over R squared. Just cancels. And we get N squared over R cubed. And that's the end of the proof sketch.

[0:30:05](https://youtu.be/h0Ln82BR4R0?t=1805) Yeah.

[0:30:05](https://youtu.be/h0Ln82BR4R0?t=1805) AUDIENCE: What is the sum?

[0:30:09](https://youtu.be/h0Ln82BR4R0?t=1809) LAWRENCE GUTH: The sum has logarithmically many terms, and so it gets absorbed. Maybe I should write that. That's right. That's gets absorbed in the last step.

[0:30:33](https://youtu.be/h0Ln82BR4R0?t=1833) Yeah.

[0:30:33](https://youtu.be/h0Ln82BR4R0?t=1833) AUDIENCE: So the only part where we use that is well space, it's like the delta rho \[INAUDIBLE] is the only place we're using.

[0:30:49](https://youtu.be/h0Ln82BR4R0?t=1849) LAWRENCE GUTH: Yeah, so the question is, where did we use its well spaced? And I think it's instructive to trace where we used all the hypotheses. So let's label this WS. Well, let's label it well spaced.

[0:31:08](https://youtu.be/h0Ln82BR4R0?t=1868) Which we can abbreviate WS. So where did we use that as well space?

[0:31:15](https://youtu.be/h0Ln82BR4R0?t=1875) We used it here. And that's the only place.

[0:31:21](https://youtu.be/h0Ln82BR4R0?t=1881) So this guy is N. That's the only place that we used. OK. We had another hypothesis that R is bigger than average.

[0:31:45](https://youtu.be/h0Ln82BR4R0?t=1905) R bigger than average. I have, unfortunately, hidden a little bit where we used that hypothesis. But it was in this Fourier estimate. So here we use R big and perhaps a little bit-- we might use a little bit of well, space too. Although we don't need anything as strong as what's up there. Where did R big come in? Well, we had all these tubes and we took some of the characteristic functions of the tubes, smoothed out a little bit and we did Fourier analysis with them.

[0:32:19](https://youtu.be/h0Ln82BR4R0?t=1939) And there was a high frequency part and a low frequency part, and some middle frequencies. This is the contribution of the high frequency part. The high frequency part dominates, immediately you get this. The very low frequencies corresponds-- with the low frequency dominated, it would mean that R was equal to this. And there are some in-between frequencies, so you have to compute a little bit to see that they're controlled by those two cases.

[0:32:50](https://youtu.be/h0Ln82BR4R0?t=1970) Good. Yeah.

[0:32:53](https://youtu.be/h0Ln82BR4R0?t=1973) AUDIENCE: Perhaps because the Fourier method doesn't really need well-spacing condition, but would it be possible to use well-spacing condition to put a cap on the high frequency domain? Because it seems like if the frequency is corresponding to the delta balls, it should be low if the delta balls are very small space.

[0:33:18](https://youtu.be/h0Ln82BR4R0?t=1998) LAWRENCE GUTH: So the question is about the relationship between the spacing of E and the frequencies in the Fourier transform of E. Is that the question? Yeah.

[0:33:33](https://youtu.be/h0Ln82BR4R0?t=2013) So if you just took one delta ball here and you smoothed out its characteristic function, it would have a Fourier transform that was supported on a delta inverse bar. Now we have a bunch of them. And so each of them has a Fourier transform that's supported on a delta inverse ball. And now we're going to add them all up. Now when we add them all up, what happens is complicated and it depends on exactly where we put them all. But absent any detailed information about exactly where we put them all, you would typically expect that they would have maybe square root cancellation and that the sum would still be pretty evenly spread over a delta inverse point.

[0:34:16](https://youtu.be/h0Ln82BR4R0?t=2056) Yeah. So being well spaced, we would see quite a lot of high frequency. But I don't know that we could recognize whether we were well spaced. I guess if you had more points closer together, you would see more low frequency stuff.

[0:34:36](https://youtu.be/h0Ln82BR4R0?t=2076) Good. Other questions or comments?

[0:34:45](https://youtu.be/h0Ln82BR4R0?t=2085) OK. I had, myself, a couple comments. So one thing was, it seemed kind of magical to me that we ended up with the right answer. And looking back, in all the proofs of anything related to Szemerédi-Trotter, it seems rather magical to me that we get the right answer. So the Szemerédi-Trotter, or this theorem, there's a construction. You take a grid. You take some lines at rational slopes. You compute a little bit and you see what happens.

[0:35:16](https://youtu.be/h0Ln82BR4R0?t=2116) And you get this N squared over R cubed. Then independently from that, you look at some cell partitions or some this or some that, or some Fourier analysis and with completely unrelated argument, you get something. And if you're lucky, it's N squared over R cubed. And we declare victory. In all the cases, I don't really understand why we are getting the right answer. And it's perhaps worth mentioning that Szemerédi-Trotter has many cousin problems, which are almost as simple to state, but no one knows the right answer.

[0:35:53](https://youtu.be/h0Ln82BR4R0?t=2153) So for instance, if you replace straight lines with circles or degree 2 curves or unit circles, or make up your favorite class of curves, you get natural questions that people have studied hard and almost every one of them we cannot prove matching examples and upper bounds. So it is plausible that there is no deep reason why this bound matches the example, but that we were just lucky.

[0:36:26](https://youtu.be/h0Ln82BR4R0?t=2186) Yeah.

[0:36:26](https://youtu.be/h0Ln82BR4R0?t=2186) AUDIENCE: Most of other variations of the problem, is it harder to show a similar bound or to find examples analogous to the kind of different problem?

[0:36:37](https://youtu.be/h0Ln82BR4R0?t=2197) LAWRENCE GUTH: So the question is, for all these other variations, is it harder to prove upper bounds or harder to find good examples? In all these problems, all of the known examples look a lot like the grid example. They involve grids of points, and you try to intersect it with a lot of circles or a lot of parabolas or what have you. And there's some upper bound. And the proofs of the upper bounds are not so different from the proofs that we have shown. But the numbers that come up, they just don't match.

[0:37:09](https://youtu.be/h0Ln82BR4R0?t=2229) And popular opinion probably is on the side that the upper bound should be improved. But we are talking about open problems. So nobody knows.

[0:37:23](https://youtu.be/h0Ln82BR4R0?t=2243) OK. So that's one comment. Another comment is that in the context of the Furstenberg problem, there is a key difficulty that the Furstenberg problem was false over the complex numbers. The Furstenberg conjecture. But it's true over the real numbers. So always on the lookout for what steps distinguish the real numbers from the complex numbers. This argument does not distinguish them. This statement can be-- you can replace all the R's with C's and adjust a few things and it's still true. So this part of the proof is not helping with that issue. Yeah.

[0:37:53](https://youtu.be/h0Ln82BR4R0?t=2273) Cool. OK.

[0:38:00](https://youtu.be/h0Ln82BR4R0?t=2280) So that was part 1 of the class, the well-spaced case using the Fourier method. Part 2 combining all the stuff we've done. Let's start by remembering what is the big theorem that we're discussing the proof of. And then what are all the things that we've done. In the last part we'll talk about how to put the pieces together. OK.

[0:38:48](https://youtu.be/h0Ln82BR4R0?t=2328) All right. So here is the big theorem that we would like to prove.

[0:38:56](https://youtu.be/h0Ln82BR4R0?t=2336) So if E is a delta T set in the unit ball in R2.

[0:39:06](https://youtu.be/h0Ln82BR4R0?t=2346) And let's say that the size of E is delta to the minus T. And then for every point in E, TX is a delta S set of tubes through the point x. And there's no loss of generality in assuming that this is delta E to the minus S. Put them all together to make a set of tubes T. OK.

[0:39:40](https://youtu.be/h0Ln82BR4R0?t=2380) And then thinking about how these tubes intersect E. So let's say R is the typical value of E intersect T delta covering number typical over T and T. Could mean average if you like. But after a little bit of pigeonholing, we can assume that they're all about the same as each other. OK.

[0:40:10](https://youtu.be/h0Ln82BR4R0?t=2410) So then the goal is to say that R is not too big. These tubes can't overlap the points too much. R is bounded by the maximum of three things corresponding to three scenarios.

[0:40:33](https://youtu.be/h0Ln82BR4R0?t=2433) OK. So A is the stars example.

[0:40:40](https://youtu.be/h0Ln82BR4R0?t=2440) These tubes have nothing to do with each other. So the number of dots on each tube is just 1. That's scenario A. B is the really interesting one, it's the integer grid. And so this-- we've written things a little differently but this corresponds to the numerology in Szemerédi-Trotter. And C is the regime where random tubes-- No. If this set E is really large, random tube will hit a lot of points of E. And this set T certainly could be made out of random tubes.

[0:41:14](https://youtu.be/h0Ln82BR4R0?t=2474) OK. So that is the theorem we're trying to prove.

[0:41:20](https://youtu.be/h0Ln82BR4R0?t=2480) And we have developed many techniques and tools to work on this theorem. And we have proven many cases of it. And let's document that.

[0:41:53](https://youtu.be/h0Ln82BR4R0?t=2513) OK. So cases we have discussed the proof.

[0:42:08](https://youtu.be/h0Ln82BR4R0?t=2528) OK. So there are these A, B, and C scenarios. And so we should sort this theorem according to who's bigger, A or B or C. So A dominates, that's if and only if, S is greater than or equal to T. Let me write it actually as T is less-- S is greater than or equal to T. And in this case, it's true by a double counting argument.

[0:42:42](https://youtu.be/h0Ln82BR4R0?t=2562) And C dominates if and only if S plus T is greater than or equal to 2. And that's also true by the Fourier method.

[0:43:01](https://youtu.be/h0Ln82BR4R0?t=2581) So B dominates otherwise, and it's actually important to write down this otherwise. So S is between-- T is between S and 2 minus S. That's what's left. And this one we don't know yet.

[0:43:21](https://youtu.be/h0Ln82BR4R0?t=2601) But we do know a couple of special cases of this hard case. But we do know if E is AD regular.

[0:43:37](https://youtu.be/h0Ln82BR4R0?t=2617) So that was my open-ended Shmerkin. We talked about it last time. And we do know so it is true E is regular and it is true if E is well spaced.

[0:43:55](https://youtu.be/h0Ln82BR4R0?t=2635) We just talked about. And that's striking because AD regular and well spaced are the two extremes in the way that E could be spaced. And the other stuff is in the middle here.

[0:44:14](https://youtu.be/h0Ln82BR4R0?t=2654) So actually, people in the field have run into this problem a number of times.

[0:44:21](https://youtu.be/h0Ln82BR4R0?t=2661) OK. So there are several cousin problems of this Furstenberg set problem. There's the Kakeya problem, which is the most famous. There's also the Falconer distance set problem. And in both of those problems, there was a period where people knew how to do these two extreme cases. And we thought-- we hoped maybe we were close and we tried to do the stuff in the middle, and it was hard and messy, and we couldn't do it. So an open-ended Shmerkin wrote their paper. They said we solved the AD regular case, and we know that previously some people solved the well-spaced case, and what's left is to do the stuff in the middle.

[0:44:54](https://youtu.be/h0Ln82BR4R0?t=2694) But we're not sure whether that might be very hard. OK. Cool.

[0:45:10](https://youtu.be/h0Ln82BR4R0?t=2710) And so the last idea in the proof is an idea from Reyn and Wang how to break up the general case into pieces that can be handled by these methods. The first part of what they did is they souped up this well-spaced case a little bit. So they proved that it's true if E is something called semi well-spaced that I will define for you. Definition soon.

[0:45:41](https://youtu.be/h0Ln82BR4R0?t=2741) OK.

[0:45:45](https://youtu.be/h0Ln82BR4R0?t=2745) It's a mild generalization of this. And the proof is fairly-- a little more complicated but it's similar ideas to what I showed you. It doesn't distinguish R from C. It's the same tools that I showed you. OK. And then the really important thing is how to put all these things together. So we're going to put all these things together by using a multi-scale argument, which is similar to the submultiplicative Lemma.

[0:46:17](https://youtu.be/h0Ln82BR4R0?t=2777) That was one of the key things in the AD regular case. So let me remind you of that and draw the important picture again.

[0:46:54](https://youtu.be/h0Ln82BR4R0?t=2814) All right. So multi-scale argument.

[0:47:10](https://youtu.be/h0Ln82BR4R0?t=2830) OK. So we're going to have delta-- we're going to have some scale which is in between delta and rho. And we're going to visualize E at this scale but also at that scale. So now we have a general E. But it's uniform.

[0:47:28](https://youtu.be/h0Ln82BR4R0?t=2848) And so I think of it as having some rho balls. So these circles are E rho. Inside of each rho ball, we have some delta balls.

[0:47:44](https://youtu.be/h0Ln82BR4R0?t=2864) So those are the delta balls in E delta. OK.

[0:47:51](https://youtu.be/h0Ln82BR4R0?t=2871) And we have some fat tubes.

[0:47:59](https://youtu.be/h0Ln82BR4R0?t=2879) Like that. So those are the tubes that scale rho. Inside of the fat tubes we have some thin tubes.

[0:48:10](https://youtu.be/h0Ln82BR4R0?t=2890) I should draw it a little differently.

[0:48:27](https://youtu.be/h0Ln82BR4R0?t=2907) So those are the delta tubes.

[0:48:33](https://youtu.be/h0Ln82BR4R0?t=2913) Now what do we notice? We notice that the R that's associated to E delta and T delta can be bounded by a coarse scale. So this R is how many delta balls are there in an orange tube like this? So first, we can ask, how many rho balls does the orange tube have? So that would be R of E rho T rho.

[0:49:05](https://youtu.be/h0Ln82BR4R0?t=2945) And then we look inside each of those rho balls at this short delta tube. And we ask, how many delta balls does it hit? And let's call one of these balls B, the typical rho ball. And that will be R of EB TB.

[0:49:25](https://youtu.be/h0Ln82BR4R0?t=2965) What is EB TB? So EB is just E intersect B. And then let's rescale it so that B becomes a unit ball. And TB is you take T-- well, you take T intersect B, where T is in T. You take all these little pieces of tubes and then again you rescale so that B becomes the unit ball.

[0:49:56](https://youtu.be/h0Ln82BR4R0?t=2996) And so EB is a set of delta over rho balls. And TB is a set of delta over rho tubes. We do this rescaling so we can relate this. Think of this as a case of our original problem. We can use induction. OK. But this inequality is just how many delta balls are there in the orange tube? Well, how many rho balls does it hit and how many delta balls are in each rho ball?

[0:50:30](https://youtu.be/h0Ln82BR4R0?t=3030) OK, cool. So by this multi-scale argument, we can take our original problem and break it up into pieces. We get to choose rho, so we have some flexibility there. Then we can repeat that. We can break these up into pieces and so on. And we would be very happy if all of these pieces are in the list of things that we understand. And then we could multiply together all those bounds and get a bound. And in some cases, we'll get the sharp bound. And then that's how they prove the theorem.

[0:51:05](https://youtu.be/h0Ln82BR4R0?t=3065) OK. Now to keep track of what happens, everything is described in terms of the branching function. And for people's intuition, it has been very helpful to draw pictures of the graph of the branching function.

[0:51:38](https://youtu.be/h0Ln82BR4R0?t=3098) OK. Do graph of the branching function.

[0:51:46](https://youtu.be/h0Ln82BR4R0?t=3106) So we're going to suppose that E is uniform. And then I'm going to draw a graph that encodes the information of how many rho balls there are to cover E at every scale. So on this axis, I want to put rho. And what I'm going to put is log delta of rho. So it's going to be a number between 0 and 1. And now on this axis, I want to understand the covering number E rho. And what I'm going to put here is the logarithm in the base 1 over delta of this covering number.

[0:52:24](https://youtu.be/h0Ln82BR4R0?t=3144) OK. So if E delta or delta to the minus T, then when you plug in 1 here, that corresponds to the scale delta. So this is delta E to the minus T. So this logarithm would be T of this. So I'm going to call this variable T-- I'm going to call this variable X. And this is F of X. F is the branching function.

[0:52:55](https://youtu.be/h0Ln82BR4R0?t=3175) So if E of-- E delta is delta E to the minus T, it means F of 1 equals T. If E is a delta T set, then it means that F of X is bigger than T times X. So here's the straight line with slope T. And I'll put this in blue.

[0:53:26](https://youtu.be/h0Ln82BR4R0?t=3206) F of X is bigger than T times X. There are many things that we could draw that do that, maybe like that.

[0:53:43](https://youtu.be/h0Ln82BR4R0?t=3223) So these pictures always start at 1. Because when X is 1, it means we're talking about the scale rho equals 1. And this is a subset of the unit ball. So that's the covering number at scale 1 is 1. So it always starts there.

[0:54:04](https://youtu.be/h0Ln82BR4R0?t=3244) Any questions or comments about these, setting this up as a graph?

[0:54:11](https://youtu.be/h0Ln82BR4R0?t=3251) OK. So now what do we know about this function F? So F is monotonically increasing.

[0:54:24](https://youtu.be/h0Ln82BR4R0?t=3264) Why? As we go this way, rho is getting smaller and the rho covering number is only going to get bigger. But it doesn't increase too quickly. So F of X plus delta X is less than or equal to F of X plus 2 delta X. So F is 2 Lipschitz.

[0:54:48](https://youtu.be/h0Ln82BR4R0?t=3288) Why? Well, if we have a rho ball and then we decrease rho, we can always cover that with a certain number of rho tilde balls. So B rho is covered by rho tilde rho over-- so if rho tilde is less than rho, then B rho is covered by rho over rho tilde squared rho tilde balls. And we may or may not need all of them, but the most we could need is all of them.

[0:55:22](https://youtu.be/h0Ln82BR4R0?t=3322) And that would correspond to equality here. So F is 2 Lipschitz.

[0:55:30](https://youtu.be/h0Ln82BR4R0?t=3330) Yeah.

[0:55:31](https://youtu.be/h0Ln82BR4R0?t=3331) AUDIENCE: So the 2 comes from the dimension?

[0:55:33](https://youtu.be/h0Ln82BR4R0?t=3333) LAWRENCE GUTH: Two comes from the dimension. This 2 is this 2, which comes from the dimension. Yeah, OK. That's right. So let me say E is uniform, E is contained in the unit ball, and E is in two dimensions. And if you change this 2 to a D dimension, the only thing in this discussion that would change is that these 2's would become D's. Yeah, thank you.

[0:55:57](https://youtu.be/h0Ln82BR4R0?t=3357) Yeah, good. OK. So to get a feel for this, I will draw a few pictures of the branching functions for cases that we have considered. OK. So for the AD regular set, before I draw it picture in your mind's eye, what does the branching function look like?

[0:56:31](https://youtu.be/h0Ln82BR4R0?t=3391) It is a straight line.

[0:56:40](https://youtu.be/h0Ln82BR4R0?t=3400) The well-spaced example.

[0:56:51](https://youtu.be/h0Ln82BR4R0?t=3411) So well-spaced example looks like this. It's a straight line of slope 2 up to the height T, and then it's a horizontal line. It's a flat line after that.

[0:57:14](https://youtu.be/h0Ln82BR4R0?t=3434) Cool. OK. And now I can tell you what is the semi well-spaced set. So here's what we do.

[0:57:26](https://youtu.be/h0Ln82BR4R0?t=3446) So semi well-space definition is now. So T is going to be in between minus S and S. That's good. That's the region we don't already understand. So E delta is going to be delta to the minus T. So that point there is going to be in our branching function. And then what I will do is I draw here a line with slope S, and here a line with slope 2 minus S. By the way, S is between 0 and 1.

[0:58:00](https://youtu.be/h0Ln82BR4R0?t=3480) So 2 minus S is the bigger one. And I extend these lines until they meet. They'll meet somewhere.

[0:58:09](https://youtu.be/h0Ln82BR4R0?t=3489) And then the branching function F of X has to lie above this stuff, and it has to go there. That's the definition of semi well spaced.

[0:58:27](https://youtu.be/h0Ln82BR4R0?t=3507) OK. So Reyn and Wang, they proved that in the semi well-spaced case, this theorem is true. And we won't do the whole proof. But the proof is a variant on what we saw here, where you combine in a clever way the Fourier method and the double counting method. And at a very high level, this setup is-- this slope 2 minus S is the slope where the Fourier method works. And this slope S is the slope where the double counting method works.

[0:59:01](https://youtu.be/h0Ln82BR4R0?t=3541) And so this picture turns out to be the situation where you can combine those two methods well. OK.

[0:59:11](https://youtu.be/h0Ln82BR4R0?t=3551) All right. So now let's think about what happens when we do this multi-scale argument. What happens to the branching functions? In other words, I start with this setup, I want to understand this guy. And it depends on the branching function of E delta. And I get these two new smaller problems. What is the relationship between this branching function and these branching functions? The relationship is really simple.

[0:59:54](https://youtu.be/h0Ln82BR4R0?t=3594) So F of X itself, this is the branching function of E delta. And it looks like something. OK. It could be anything. And then we have this scale rho. So over here is log delta of rho. And if you think about this set E rho, well, it's just the branching function of E rho is the same as the branching function of E delta. Except it just stops at rho.

[1:00:25](https://youtu.be/h0Ln82BR4R0?t=3625) And we don't look at any smaller scales. So this here in orange is basically the branching function of E rho.

[1:00:45](https://youtu.be/h0Ln82BR4R0?t=3645) Then we look inside of this ball and we look at all of the-- start at the scale rho and then we look at all of the smaller scales of E. So this thing here is-- let me say this describes F EB. This is basically the branching function of EB.

[1:01:16](https://youtu.be/h0Ln82BR4R0?t=3676) OK, great.

[1:01:19](https://youtu.be/h0Ln82BR4R0?t=3679) So what's the situation? We begin with an arbitrary branching function that just obeys this condition. Yeah.

[1:01:30](https://youtu.be/h0Ln82BR4R0?t=3690) AUDIENCE: Going back to the multi-scale or the submultiplicative Lemma, why is it-- like the second one, the R of EB TB when it's like the number of delta balls is the same in the rho ball. They have the same.

[1:01:48](https://youtu.be/h0Ln82BR4R0?t=3708) LAWRENCE GUTH: Right. Right. So this R of EB TB is supposed to be counting the number of delta balls in a segment like that. Now, why is it counting that? Who is EB? EB are the delta balls in the rho ball in B. And TB are-- I take the tubes and I just take the part of the tube that is in the-- that's in the ball.

[1:02:25](https://youtu.be/h0Ln82BR4R0?t=3745) AUDIENCE: OK. So then R E rho, that's the number of rho balls in the rho 2 based on the product is just the number that amounts to 2.

[1:02:41](https://youtu.be/h0Ln82BR4R0?t=3761) LAWRENCE GUTH: Right. Right. Right. So at T rho is this blue tube and R of E rho T rho is the number of rho balls in the blue tube. And then if we take an orange tube, it will go through this many of the rho balls. And we want to know how many delta balls are there in each of those bits. So now we have some tubes at this tubes inside of B and points inside of B. And we blow that up.

[1:03:12](https://youtu.be/h0Ln82BR4R0?t=3792) And that's over here.

[1:03:15](https://youtu.be/h0Ln82BR4R0?t=3795) OK, good OK. So here's something we might try. We start with E delta T delta. We don't know how to handle it. We look at the branching function. I can cut the branching function anywhere I like and produce two branching functions. And I can repeat that. And then I can take the different branching functions I get. And hopefully, I can put them into the list that I know how to handle. And then I'll multiply together and I'll get an answer.

[1:03:47](https://youtu.be/h0Ln82BR4R0?t=3827) So let me do a couple examples of this concatenation method and we'll see how it works.

[1:04:12](https://youtu.be/h0Ln82BR4R0?t=3852) Concatenation method.

[1:04:18](https://youtu.be/h0Ln82BR4R0?t=3858) All right. So let me do the example where our graph consists of two lines. So it has a line of slope T1 up to somewhere. So here's log delta of rho. And then it has a line at a different slope, which let's say is smaller. So here's slope T2 to the end. It's 1. And let's say that this final height is T. So how are they related?

[1:04:49](https://youtu.be/h0Ln82BR4R0?t=3889) So delta to the minus T is E delta, which I can read off from my final branching function. That's the size of E rho times the size of EB. Number of rho balls times number of delta balls per rho ball. So that is rho to the minus T1. So this is slope T1. Times delta over rho to the minus T2.

[1:05:19](https://youtu.be/h0Ln82BR4R0?t=3919) So now let's see what happens. So R of ET is bounded by R of E rho T rho R of EB TB. Scenario 1.

[1:05:40](https://youtu.be/h0Ln82BR4R0?t=3940) In this scenario 1, T1 and T2 are both in this middle range where B dominates.

[1:05:51](https://youtu.be/h0Ln82BR4R0?t=3951) So then this guy will give me rho to the S over 2, rho to the minus T1 over 2. Because this is option B. B dominates. And this guy here will give me delta over rho to the S over 2, delta over Rho to the minus T2 over 2. Because that's option B. This is not B as in ball, this is B as in A, B, and C, the three options.

[1:06:24](https://youtu.be/h0Ln82BR4R0?t=3984) All right. When you multiply it out, something nice happens. What happens when we multiply these together? We get delta to the S over 2. And what happens when we multiply those together? At first it looks like an unholy mess, but it's the same as this.

[1:06:48](https://youtu.be/h0Ln82BR4R0?t=4008) And so it's delta to the minus T over 2, which is B, which is what we were hoping for.

[1:06:59](https://youtu.be/h0Ln82BR4R0?t=4019) This is a smiley face. But let me mention another scenario. Scenario 2. Maybe T1 is bigger than 2 minus S, and maybe T2 is smaller than S. So maybe this first one is really quite steep. And then the second one is quite shallow. OK. Well, now this first one, when I evaluate this, I'm in the C domain.

[1:07:35](https://youtu.be/h0Ln82BR4R0?t=4055) I get rho to the 1 minus T1. And when I evaluate this, because this T2 is less than S, I'm in the A domain, which is 1.

[1:07:49](https://youtu.be/h0Ln82BR4R0?t=4069) And that's not what I'm hoping for. That's actually much bigger than my goal.

[1:08:02](https://youtu.be/h0Ln82BR4R0?t=4082) So actually, let me add to scenario 2. But T is in the middle. T is a average of T1 and T2. T1 is bigger-- is quite big, T2 is quite small. But this average is going to be in the middle. Since T is in the middle, this is the goal.

[1:08:20](https://youtu.be/h0Ln82BR4R0?t=4100) How does this compare to the goal? Not so obvious, but it's actually much bigger than the goal. What went wrong? Remember that at each step when I bounded R, my theorem bounds R by a maximum of three things. Maximum of A, B, and C. So this C is way bigger than B in this regime. And this A is way bigger than B in this regime. And B times B would be my goal.

[1:08:52](https://youtu.be/h0Ln82BR4R0?t=4132) So this is way too big. So it doesn't work. OK. So lesson that we learned here is that we want to break up F into pieces so that-- OK, I'll say it in words. So first of all, we'd like to be able to analyze each piece. So we want each piece to be on the list of things we understand. So if you don't have a sharp bound for the piece, there's no way you're going to have a sharp bound for the whole thing.

[1:09:25](https://youtu.be/h0Ln82BR4R0?t=4165) But also, we're not happy with just any of these. We really want them all to be in case B. Right.

[1:09:43](https://youtu.be/h0Ln82BR4R0?t=4183) So we really want each piece to be either AD regular with T of that piece in our middle range, in the B range, or semi well spaced.

[1:10:19](https://youtu.be/h0Ln82BR4R0?t=4219) OK.

[1:10:24](https://youtu.be/h0Ln82BR4R0?t=4224) Cool.

[1:10:32](https://youtu.be/h0Ln82BR4R0?t=4232) So the very last idea is that indeed, you can do this. You can start with an arbitrary branching function, and then you can break it up into pieces where each piece is like this. I'll write it down in a second. Draw an example.

[1:10:53](https://youtu.be/h0Ln82BR4R0?t=4253) Probably the most important idea behind this is the Rademacher theorem. So the Rademacher theorem says that a differentiable function-- a Lipschitz function is differentiable almost everywhere. And so this function F, we don't know a lot about it, but we did see that its Lipschitz function. And so it's differentiable almost everywhere. So that means that we can cut up our interval into little intervals where maybe you have to throw out a tiny bit. But on almost every little interval, it's approximately a constant slope example.

[1:11:28](https://youtu.be/h0Ln82BR4R0?t=4288) So you can reduce everything-- just by Rademacher theorem, you can reduce everything to the AD regular irregular case. And that's an indication that AD regular case is really important. It's not just a special case. However, if you do that, we would not only have slopes in here, we would also have slopes that are too small or too large. So we want everybody to be in case B. And that's why we need some extra stuff like the semi well-spaced. OK. So I will write down this last thing.

[1:12:10](https://youtu.be/h0Ln82BR4R0?t=4330) So final Lemma.

[1:12:14](https://youtu.be/h0Ln82BR4R0?t=4334) If F is a function from 0, 1 to R, which is 2 Lipschitz and increasing, then we can break up 0, 1 as a disjoint union of some intervals, maybe plus tiny leftover.

[1:12:37](https://youtu.be/h0Ln82BR4R0?t=4357) And then on each I, either F restricted to I-- either F is almost linear with slope TI, and TI is in our range.

[1:13:00](https://youtu.be/h0Ln82BR4R0?t=4380) I should say one other thing. Let's say F of 1 is T and T is in our range. Oh, yeah. And we should-- sorry. We should have F of X is bigger than T of X. So this is our setup. We have a delta T set. We're looking at its branching function. So either its almost linear and its slope is in the desired range or on I, F is semi well spaced.

[1:13:29](https://youtu.be/h0Ln82BR4R0?t=4409) OK. It's not so difficult to prove once you think to try to prove it, but I'm just going to illustrate it with an example.

[1:13:44](https://youtu.be/h0Ln82BR4R0?t=4424) So suppose F looks like this. And so the slope is increasing for a while. I guess it needs to start bigger than T. So a bit tricky to draw this picture here. So it needs to stay above this dotted line, right? So it's going to start tangent to the line, which is OK. The slope increases for a while. Has a maximum slope, and the slope decreases. And it goes like that. And as this slope increases, there is a key moment where the slope gets too big.

[1:14:20](https://youtu.be/h0Ln82BR4R0?t=4460) Over here, slope is 2 minus S. And maybe at the end the slope is too small. The slope is less than S. OK, so what should we do? Well, this whole thing here, this whole thing is semi well-spaced.

[1:14:47](https://youtu.be/h0Ln82BR4R0?t=4487) And at the beginning we can take very short intervals. So we'll have this guy and this guy, and this guy. And these blue guys are almost-- they're short, so it's almost linear.

[1:15:04](https://youtu.be/h0Ln82BR4R0?t=4504) And the slope is not yet bigger than 2 minus S. So the slope TI is less than 2 minus S. Bigger than S. OK. This is not a proof. But there are not that many ways this function could look. And if you draw five of them, you can convince yourself. Then if you want, you can prove.

[1:15:30](https://youtu.be/h0Ln82BR4R0?t=4530) OK. So to summarize, This picture, this fact is very powerful because it lets us break up our original problem into smaller problems. And the smaller problems in terms of the branching functions because the smaller problems are just little pieces of the branching function of the bigger problem. And when you do that, there are--

[1:15:57](https://youtu.be/h0Ln82BR4R0?t=4557) what originally looked like we just had a few special cases, now we can take any combination of those few special cases. And that turns out to be a lot. And even to be everything.

[1:16:12](https://youtu.be/h0Ln82BR4R0?t=4572) So that's the last idea. The proof. The proof how to put the pieces together. And so these pieces were not just special cases. They are really the fundamental building blocks of the whole story. And it also is striking to me that this multi-scale argument was really crucial in both the AD regular case and the not the AD regular case. In the AD regular case, these problems were all the same. And then seeing that sameness at all these different situations, it gave us some extra rigidity and structure.

[1:16:48](https://youtu.be/h0Ln82BR4R0?t=4608) In the not AD regular case, by definition, these problems are different. And that means that gives us a whole menu of problems to choose from. And then we can look through this menu and try to find the things that we understand.

[1:17:02](https://youtu.be/h0Ln82BR4R0?t=4622) OK, cool. OK. So that was an outline of projection theory. Thank you for coming with me on this outline. It's been nice talking with you all. And yeah, let's stop the class there.
