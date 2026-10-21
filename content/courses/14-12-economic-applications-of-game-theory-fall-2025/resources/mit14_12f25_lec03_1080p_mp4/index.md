---
title: 'lecture 3: dominance'
description: "This lecture covers the Prisoner's Dilemma, the concept of dominant strategies, best responses, and how to identify which strategies are worth considering in competitive situations."
id: 'mit-14-12-fall-2025-resource-resources-mit14-12f25-lec03-1080p-mp4'
tags:
  - 'course'
  - 'mit'
  - 'ocw'
  - 'social science'
  - 'economics'
  - 'game theory'
  - 'lecture videos'
semantic: false
resource:
  {
    kind: 'video',
    n: 3,
    youtube: 'b7BAHSV1EBo',
    duration: 4583,
    cues: 1557,
    captions: 'static_resources/1kTWizDnEqFpQ6k4Kcw5Ji0T6HyKr3zCk_transcript.webvtt',
    transcript: 'static_resources/1kTWizDnEqFpQ6k4Kcw5Ji0T6HyKr3zCk_transcript.pdf',
  }
layout: 'resource'
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
---

[0:00:09](https://youtu.be/b7BAHSV1EBo?t=9) IAN BALL: So today, we're going to start analyzing games.

[0:00:19](https://youtu.be/b7BAHSV1EBo?t=19) So in my view, this is where things start getting a lot more interesting. So far, we've just said how to set up a game-- or first, how to set up an individual decision problem, and then how to set up or represent a game. And now, we're actually going to start thinking about analyzing the game and trying to make predictions about how people will play in the game. So I want to first go over the approach that we're usually going to take in this course. We'll usually start with some informal description of a strategic setting. That may be something that you get in a problem set or from thinking about the world.

[0:00:54](https://youtu.be/b7BAHSV1EBo?t=54) Then, based on the informal description, we'll usually write down the extensive form representation.

[0:01:06](https://youtu.be/b7BAHSV1EBo?t=66) Remember, this is the very detailed representation that we talked about a lot on Tuesday. And then we'll reduce this extensive form representation to the strategic form representation.

[0:01:26](https://youtu.be/b7BAHSV1EBo?t=86) And later on in the course, we will discuss some issues that can only be perceived in the extensive form representation. But for the beginning of the course-- and in particular, in today-- everything we're going to say is going to be about a strategic form representation. So today, our starting point is going to be 3. We're going to imagine that we're given a strategic form game. But you can imagine, in the background, that we may have derived arrived this given strategic form game from an extensive form that we previously wrote down.

[0:02:02](https://youtu.be/b7BAHSV1EBo?t=122) So let's just recall the notation for a strategic form game. We have the players, i equals 1 through n And then for each player i, we have a strategy set associated with that player i. So we have big S1 up to big Sn, where S1 is the set of strategies that are available to player 1. Remember, the set can be quite complicated, if we've derived this set by evaluating all complete contingent plans in the extensive form representation.

[0:02:36](https://youtu.be/b7BAHSV1EBo?t=156) And then we have our payoffs, ui from S to R, where, remember, S is the set of strategy profiles. So S, mathematically, is a product set of S1 up to Sn. But the key thing is that what lives in S are profiles files specifying a strategy for each of the n players. And we have a function like this, again for each of the n players, that specifies each player i's utility as a function of the profile of strategies that the players play.

[0:03:12](https://youtu.be/b7BAHSV1EBo?t=192) Now, what is our goal when we analyze a game? I think we're asking the following fundamental question-- given a game, we want to ask, how do we think players will play or should play the game?

[0:03:46](https://youtu.be/b7BAHSV1EBo?t=226) Now, notice, I think the distinction between will play and should play gets into two different interpretations of the analysis that we do. So the answers that we get and the formal analysis we do won't really depend on which interpretation you want to keep in mind. But depending on the application, one may be more or less compelling. So will play-- if we want to think about will play, this is what sometimes called positive analysis or descriptive analysis. And the idea of this is that, if we observe some game, we want to be able to make a prediction about how rational players will choose to play this game.

[0:04:24](https://youtu.be/b7BAHSV1EBo?t=264) So the goal here is really a prediction, which will, of course, only be valid if we think that the utility functions that we write down and the rationality assumptions that we make are compelling. Another interpretation of game theory is to put yourself in the role of an advisor. Someone comes to you and says, how should I play in this game? What is your recommendation for how I should play? And companies do sometimes hire game theorists to consult and make recommendations for maybe how they should bid in an auction. That would be one example.

[0:04:55](https://youtu.be/b7BAHSV1EBo?t=295) So here, this is sometimes called normative or prescriptive analysis. And the goal of our analysis is to provide what I might say is a recommendation for how players should play if they want to achieve their objectives and they recognize that other players are rational as well and maybe also are getting advice from game theorists or from other informed parties about how they should play in the game.

[0:05:30](https://youtu.be/b7BAHSV1EBo?t=330) But I just want you to keep this in mind, in the background, throughout the course. But when you're formally asked to solve for an equilibrium or to analyze a game, your answer won't really depend on your particular interpretation in that context. So I want to start with an example of a game, an example of a strategic form game. And this is the most famous game in game theory, probably. You may have heard of it. It's called the prisoner's dilemma.

[0:06:00](https://youtu.be/b7BAHSV1EBo?t=360) Make sure to \[INAUDIBLE].

[0:06:06](https://youtu.be/b7BAHSV1EBo?t=366) Now, the first thing I want to write down is think about how we can represent this game. It's kind of ugly to think about all these functions and all these sets. So normally, we represent strategic form games using what's called a by matrix, just a box with some numbers in it. So in this game, each player is going to have two strategies, which we'll call M and F. So let me tell the story here. So we have two prisoners.

[0:06:36](https://youtu.be/b7BAHSV1EBo?t=396) And we assume that they're placed in separate cells. Again, this is going to be a colorful story to help motivate this particular game. But the lessons from this game, I think a lot of game theorists and economists believe, are much broader than this kind of silly story that we're going to tell here. So we have two prisoners. And they've been accused of a crime. And each prisoner has two options. We'll call them M and F. So M is to stay mum-- that is, don't admit to anything. And F is sometimes the word is used fink or tattle.

[0:07:09](https://youtu.be/b7BAHSV1EBo?t=429) It's to tell on the other person-- say, yeah, the other guy did it. So the prosecutor comes by and says, do you want to accuse your co-conspirator of committing the crime? And the prosecutor says, look, we'll give you a good deal. So if you're willing to testify against the other prisoner, we'll give you a sentence reduction. But now, we're going to be able to catch the other guy. And the other guy is going to get a bigger sentence. So now, what we want to do is fill in the payoffs in this matrix. And what we imagine is, if both players stay silent, then both players get a payoff of 2.

[0:07:47](https://youtu.be/b7BAHSV1EBo?t=467) They both get convicted of, say, a minor crime, because they didn't have testimony to convict either of them. Now, let's say I'm player 1. So the way I'm writing this here is player 1 is the column-- sorry, the row player. Player 1 is choosing the row. So they're choosing between M and F. Player 2 is the column player. They're choosing between the column m and F. So the strategy set-- maybe this is helpful here-- for each player S1 equals S2 equals M comma F.

[0:08:22](https://youtu.be/b7BAHSV1EBo?t=502) So if we want to formally connect it to our strategic form representation over here, each player has two strategies. They each have the same two strategies, simply staying silent, M, or testifying against the other player, finking, F. And then when we write 2, 2 in this box, the first number represents player 1's utility from this strategy profile corresponding to this box. So this first box corresponds to the strategy profile where player one chooses M and player 2 chooses M.

[0:08:55](https://youtu.be/b7BAHSV1EBo?t=535) And then in the box, we're writing player 1'2 payoff from that consequence or that strategy profile, and then player 2's payoff from that strategy profile. Now, let's go down here. We want to say what happens if player 1 instead testifies or finks against the other prisoner. What this should do is it should increase the utility of the prisoner who finks, because that prisoner gets a sentence reduction. But it decreases the utility of the other prisoner, because the other prisoner is now going to be convicted of a more serious crime.

[0:09:26](https://youtu.be/b7BAHSV1EBo?t=566) So the exact numbers we'll use are 3 and negative 1.

[0:09:34](https://youtu.be/b7BAHSV1EBo?t=574) So relative to the case where they both stay silent, if I fink on the other prisoner, my utility goes up by 1 util. But my opponent's utility goes down by 3 utils. And now, we're going to put the symmetric payoffs over here. But remember, over here, it's prisoner 2 who's testifying. And therefore, prisoner 2 gets the higher payoff. And prisoner 1 gets the lower payoff. So here, we're going to put negative 1, 3 in this box. And then finally, we have to decide what happens if they both testify.

[0:10:08](https://youtu.be/b7BAHSV1EBo?t=608) And here, the payoff is 0, 0. So the exact numbers aren't so important. But the order is pretty important. So why is it 0 here? Well, the other person is testifying against me. So I get convicted of a pretty serious crime. But it's not quite as bad as the negative 1 over here, because I still get a little bit of a sentence reduction for testifying against the other guy. So you can see, there's an ordering between the four outcomes. And this is really what's important.

[0:10:41](https://youtu.be/b7BAHSV1EBo?t=641) So the highest payoff I get is if I testify and the other guy doesn't. The worst payoff is if the other prisoner testifies and I don't, because then I get convicted and I don't get any sentence reduction. And then in between, we have the payoff where we both stay silent, because we only get charged with a pretty small crime. And then the 0 here is where we both testify, we both get charged with a serious crime, but we get a sentence reduction because we each cooperated with the authorities.

[0:11:17](https://youtu.be/b7BAHSV1EBo?t=677) Now, I guess the question is, how would you play in this game? If you were here, you were one of the prisoners, you're alone in your cell, how would you think about this? And how would you expect the other person to play?

[0:11:34](https://youtu.be/b7BAHSV1EBo?t=694) STUDENT: \[INAUDIBLE] go F.

[0:11:35](https://youtu.be/b7BAHSV1EBo?t=695) IAN BALL: Say again?

[0:11:36](https://youtu.be/b7BAHSV1EBo?t=696) STUDENT: \[INAUDIBLE] go F.

[0:11:36](https://youtu.be/b7BAHSV1EBo?t=696) IAN BALL: You would go F? OK, tell me why.

[0:11:39](https://youtu.be/b7BAHSV1EBo?t=699) STUDENT: Because the other one doesn't, then you're just \[INAUDIBLE].

[0:11:45](https://youtu.be/b7BAHSV1EBo?t=705) IAN BALL: So you're saying you want to go F because you're really afraid of-- so let's say you're player 1. So you're going to say, if I go F, you're saying, well, I can either get 3 or 0. But I'm at least not going to get negative 1. So that's good. So certainly, the worst possible outcome is better if I go F. I think F is a good strategy. Any other thoughts about why I might want to go F? Yeah?

[0:12:09](https://youtu.be/b7BAHSV1EBo?t=729) STUDENT: Regardless of what the other player chooses, you're better off.

[0:12:11](https://youtu.be/b7BAHSV1EBo?t=731) IAN BALL: Right. So this is a really critical observation to make in this game. So I'm player 1. I don't know whether the other prisoner is going to mum or fink. I don't know if they're going to testify against me. But let me reason separately by cases, OK? One possibility is the other prisoner is going to stay silent. If the other prisoner stays silent, well, what happens? Either I get a payoff of 2, if I stay silent, or I get a payoff of 3, if I testify. So I'm better off testifying. Now, let's look at the other case.

[0:12:42](https://youtu.be/b7BAHSV1EBo?t=762) What if the other prisoner is going to testify against me? That's certainly worse for me. I'm certainly worse off. But if the other prisoner testifies, well, if I stay silent, I get negative 1. And if I fink, I get 0, which is higher? So in both cases, I do strictly better if I fink. So often, we think that a prediction of this game might be we might be pretty confident that, for each player--

[0:13:12](https://youtu.be/b7BAHSV1EBo?t=792) what are we saying? Well, F is better than M no matter what the other player does.

[0:13:34](https://youtu.be/b7BAHSV1EBo?t=814) And in a sense, games like this are a bit easier to analyze. Because, remember, we said what makes game theory hard is what we called strategic interdependence, where the best thing for me to do depends on what the other player does. In this case, it is an interactive problem. But the best thing for me to do is independent of what my opponent does. So there actually is not this element of strategic interdependence. F is better than M if my opponent plays M. And F is still better than M if my opponent plays F. So I don't really have to think too carefully about exactly how my opponent is going to play because, however they play, I know I'm better off if I play F.

[0:14:14](https://youtu.be/b7BAHSV1EBo?t=854) So why is this called a dilemma? Or sometimes, people think of it as a paradox. Well, if both players play F, then-- if I have chalk, but I'll just highlight this-- we both get a payoff of 0. But I think it's reasonable for someone to come along and say, wait a second, these prisoners must be doing something wrong. They're both getting a payoff of 0. But if they both did M, they'd both be strictly better off. So are they being irrational? Are they doing something wrong?

[0:14:44](https://youtu.be/b7BAHSV1EBo?t=884) In this game, what do you think? I mean, why is this a good prediction of the game? We said we're studying game theory. We said people are very rational. They act towards their objectives. And if they both do this, they're both strictly better off. So why don't they do that? Is something wrong with game theory?

[0:15:00](https://youtu.be/b7BAHSV1EBo?t=900) Yeah?

[0:15:01](https://youtu.be/b7BAHSV1EBo?t=901) STUDENT: If one player doesn't know what the other player is going to do, they're just operating in like just maximize my utility independent of what the other person chooses.

[0:15:15](https://youtu.be/b7BAHSV1EBo?t=915) IAN BALL: Right. And another way of saying it is I can't influence what the other person chooses. So when I framed it, I said, oh, we're down here. Wouldn't we like to be up here? But how do we get up here? Well, both players would have to behave differently. And I can't control what the other person does. I can only control what I do. And I think this observation was surprising to a lot of people early on. I mean, mathematically, it's not very deep. But it definitely surprised people. And I think the key interpretation of the prisoner's dilemma, or what we learned from it, is that, in games, rational behavior can lead to inefficiency.

[0:15:56](https://youtu.be/b7BAHSV1EBo?t=956) Or more precisely, rational behavior can make everyone worse off.

[0:16:14](https://youtu.be/b7BAHSV1EBo?t=974) Worse off than what, you might say? Well, what I mean is than some other behavior.

[0:16:26](https://youtu.be/b7BAHSV1EBo?t=986) And this is a novel implication or a novel feature of interactive decision making. In a single decision maker problem, this never happens. In a single agent problem, I just do what's best for me. And that gives me the best possible outcome. But with an interactive problem, even if we're both behaving, quote unquote, "rationally," the outcome may be very bad for both of us and may be worse than a different outcome that would be feasible for both of us.

[0:16:56](https://youtu.be/b7BAHSV1EBo?t=1016) So I think today we want to continue what we started here. And we argued that this game was relatively easy to analyze because of this property, that the best thing for me is independent of what the other player does. And I think that's a guide for the way we're going to approach analysis in this class. So moving forward, we're going to start with minimal assumptions, minimal assumptions about behavior.

[0:17:31](https://youtu.be/b7BAHSV1EBo?t=1051) And because these are minimal or weak assumptions, we're going to be quite confident in the implications of these assumptions. So this might make us confident about predictions.

[0:17:48](https://youtu.be/b7BAHSV1EBo?t=1068) But what we'll see is that a lot of games are not like the prisoner's dilemma. And we're not able to make very sharp predictions only based on minimal assumptions. So we might be confident about our predictions. But these predictions are generally going to be very weak predictions. Or they could be very weak predictions.

[0:18:06](https://youtu.be/b7BAHSV1EBo?t=1086) And here, notice, I'm using the interpretation of prediction rather than recommendation for now. In fact, sometimes, if we're only willing to make minimal assumptions about behavior, our only prediction might be anything could happen. That's not a very useful prediction. We might not be able to narrow things down very much. So what we want to do, over the course of the class, is gradually strengthen the assumptions that we make about behavior. We might gradually lose some confidence in the accuracy of these predictions. But at the same time, we'll be able to sharpen and refine the predictions that we make.

[0:18:39](https://youtu.be/b7BAHSV1EBo?t=1119) And that's the direction we're going to go. So today, we're going to start with what I think of as the most basic or most minimal assumption about behavior. And later on, we'll move to more complicated assumptions.

[0:18:54](https://youtu.be/b7BAHSV1EBo?t=1134) So the first thing I need to do is just introduce some notation for strategic form games. And then I can formally define and capture what we described in this game. So strategic form games often have many players. But we're often interested in what happens when a single player modifies their behavior, modifies their strategy. So we'd like to have some notation to capture that. And what we're going to write is a strategy profile will often decompose it in this way.

[0:19:28](https://youtu.be/b7BAHSV1EBo?t=1168) So we'll say we can think of the profile as player i's strategy.

[0:19:35](https://youtu.be/b7BAHSV1EBo?t=1175) And S negative i's specifies the strategies of everyone else.

[0:19:44](https://youtu.be/b7BAHSV1EBo?t=1184) And this notation is important. So let's just go through an example with three players. If S equals S1, S2, S3, this strategy profile specifies a strategy for each of the three players. But there's three different ways we can express this depending on which player we want to focus on. So we could write this as S1, S negative 1, meaning we're going to put player 1's strategy by itself. And then we're going to group the other two players' strategy under this notation S negative 1.

[0:20:15](https://youtu.be/b7BAHSV1EBo?t=1215) Here, negative 1 is not really the number negative 1. What we mean is not player 1. So think of the minus here as not rather than minus. But then there's another way we could write this. We could write it as S2 S negative 2 or as S3 S negative 3. And you might object that, well, S3 is supposed to come last. We're not really going to be consistent about the order. As long as we understand what this means, this is a strategy profile where player three chooses S3 and players 1 and 2 collectively choose S negative 3.

[0:20:49](https://youtu.be/b7BAHSV1EBo?t=1249) It's just notation. But it's going to help us write things down more cleanly. Yeah?

[0:20:53](https://youtu.be/b7BAHSV1EBo?t=1253) STUDENT: So I need to add more of the \[INAUDIBLE] first, \[INAUDIBLE].

[0:20:57](https://youtu.be/b7BAHSV1EBo?t=1257) IAN BALL: Sorry, I don't mean that. I'm not making any-- it's just notation, no behavioral assumptions. So literally, when I write S negative 3, what I mean is S1, S2. It's just notation. So I'm not claiming that they're going to decide together. Or I'm not doing anything like that. It's just to save space on the board. Instead of writing S1, S2, I'm going to write S negative 3. But they're still choosing independently of each other. That's a good clarification. Thank you. And then we need some notation for where these things live.

[0:21:29](https://youtu.be/b7BAHSV1EBo?t=1289) So we're going to use notation. We've already introduced Si. But now, we're going to use S negative i to be the set of-- you might call them partial strategy profiles. These are the set of profiles specifying a strategy for every player except player i. So S little i lives in SI. And S negative i, or not i, lives in S0 So that notation will be useful below.

[0:21:58](https://youtu.be/b7BAHSV1EBo?t=1318) So now, we'd like to formalize what happened in the prisoner's dilemma. And the formal term for this is dominance, a natural term because F is better than M no matter what the other player does. We would say that strategy F dominates strategy M. But we have to be a little more careful because there's strict and weak dominance. So now, I'm going to define these two terms. So for player i-- so let's consider player i.

[0:22:33](https://youtu.be/b7BAHSV1EBo?t=1353) And let's consider two strategies--

[0:22:39](https://youtu.be/b7BAHSV1EBo?t=1359) Si and Si prime.

[0:22:44](https://youtu.be/b7BAHSV1EBo?t=1364) And I want to say, so I'm focusing on player i. I'm focusing on two strategies-- they could be the same, but two strategies of player i. This is a prime here. And I want to formally define what it means for strategy Si to strictly dominate strategy S negative i. So I'm going to say Si strictly dominates--

[0:23:10](https://youtu.be/b7BAHSV1EBo?t=1390) sorry, I may have said S negative i. I mean Si prime.

[0:23:15](https://youtu.be/b7BAHSV1EBo?t=1395) Can anyone guess in words, strict dominates, what that might mean, just if we think through from the example, what it might mean for player i if strategy Si strictly dominates Si prime? Maybe at the back? Yeah?

[0:23:28](https://youtu.be/b7BAHSV1EBo?t=1408) STUDENT: They have a better payoff regardless of S negative i.

[0:23:32](https://youtu.be/b7BAHSV1EBo?t=1412) IAN BALL: Exactly right. And by better, strictly better. So Si dominates Si prime means-- sometimes I'll use this arrow just to mean a definition. So I'm saying this is exactly what this phrase means. It means that-- well, let me first just partially write it out.

[0:23:57](https://youtu.be/b7BAHSV1EBo?t=1437) So I'm going to fill it in in a second. But the first thing is I'm looking-- let's just focus on what's here already. I'm focusing on player i. And player i is making a comparison between things that are in player i's control. Player i is comparing what happens if she plays Si to what happens if she plays Si prime. But what we want to argue is that this inequality holds whatever the opponents play. So we're going to fill in S negative i here and S negative i here. And we want this to hold for all S negative i in S negative I.

[0:24:33](https://youtu.be/b7BAHSV1EBo?t=1473) So notice a key point here. S negative i can take any possible value. But in the inequality, the same S negative i appears on both sides. So fixing the way that my opponent is playing, I'm strictly preferred to play Si to playing Si prime. Why do we write it this way? Because S negative i's outside my control. I can't control what my opponents are doing. But they're doing something. So however they're behaving, this inequality should hold. Now, we want to give a slightly weaker version of this. This is very demanding.

[0:25:04](https://youtu.be/b7BAHSV1EBo?t=1504) And it may not hold. So let's now say weakly dominates. I won't fill out everything else, but just copy these words. And now, we're going to weaken this to weak inequality.

[0:25:30](https://youtu.be/b7BAHSV1EBo?t=1530) But this is a little too weak because this allows these two strategies to give me the same payoff whatever my opponents do. And we want there to be some reason that Si is better than Si prime. So we demand this weak inequality for every S negative i, but strict inequality for some S negative i.

[0:25:57](https://youtu.be/b7BAHSV1EBo?t=1557) So however my opponents play, Si does weakly better than Si prime. But there's some, at least one particular way my opponents could play such that Si is just strictly better than Si prime. And in particular, the strict part of it means that no strategy can weakly dominate itself, because if we put the same strategy here for Si and Si prime, then we would never get a strict inequality.

[0:26:27](https://youtu.be/b7BAHSV1EBo?t=1587) So far, these definitions have been pairwise comparisons. We've taken one strategy and we've compared it to another strategy. But in reality, the player just needs to choose one strategy. So what we'd like to know is when is one strategy better than everything else or dominates everything else. So we'll say a strategy Si is weakly dominant--

[0:26:52](https://youtu.be/b7BAHSV1EBo?t=1612) we could do weakly or strictly.

[0:27:00](https://youtu.be/b7BAHSV1EBo?t=1620) Si is weakly dominant if Si weakly or strictly dominates every other strategy that player i has. So maybe I'll just do weakly to avoid confusion. We say Si is weakly dominant if Si weakly dominates every strategy Si prime. Now, I need to be a little careful. There's something wrong here. I can't say every strategy, because Si can't dominate itself.

[0:27:33](https://youtu.be/b7BAHSV1EBo?t=1653) So every strategy Si prime that's not equal to C. So we think that, in this case, if player i has a weakly dominant strategy, that's a pretty safe prediction about how that player will play in the game, about which strategy that player will use.

[0:27:57](https://youtu.be/b7BAHSV1EBo?t=1677) And we have one final definition.

[0:28:02](https://youtu.be/b7BAHSV1EBo?t=1682) And that is, well, let's think about the simplest possible case. That's where every player has some dominant strategy, has some weakly dominant strategy. If that's the case, then we can be pretty confident our prediction in that game is that each player is going to play their corresponding weakly dominant strategy. So we can give a definition here, our first notion of an equilibrium. We say a profile S1 star up to Sn star is a dominant strategy equilibrium.

[0:28:45](https://youtu.be/b7BAHSV1EBo?t=1725) And here, I'm just going to write dominant strategy equilibrium. But more precisely, we mean weakly dominant strategy equilibrium. But we'll just say dominant. And sometimes, we'll call this DSE--

[0:29:00](https://youtu.be/b7BAHSV1EBo?t=1740) if what? Well, this is a profile. So we basically made a prediction about how each player in the game is going to play. We've made a prediction about each player's strategy. And we say that this is a dominant strategy equilibrium if each player is playing a strategy that's a weakly dominant strategy for that player. So if, for each player i, strategy-- well, what is the strategy used by player i?

[0:29:32](https://youtu.be/b7BAHSV1EBo?t=1772) It's Si star. So strategy Si star is weakly dominant.

[0:29:45](https://youtu.be/b7BAHSV1EBo?t=1785) But notice an important thing here that, when we make a prediction about a game, it's not enough just to predict one player's strategy. We have to make a prediction about how every player is going to play. So our prediction here is a profile of strategies specifying a strategy for each of the n players. And it's a dominant strategy equilibrium if, for each player, their strategy is weakly dominant for them, meaning that strategy, Si star-- if we go back to our definition, Si star weakly dominates every other strategy by player i.

[0:30:18](https://youtu.be/b7BAHSV1EBo?t=1818) Any question on this? Yes?

[0:30:20](https://youtu.be/b7BAHSV1EBo?t=1820) STUDENT: Is it possible to have multiple weakly dominant strategies, like one better than a different one?

[0:30:28](https://youtu.be/b7BAHSV1EBo?t=1828) IAN BALL: That's a great question. So let's put this here. Let's write this on the board because I was going to get there. That's a great question. Is it possible for there to be multiple weakly dominant strategies? So maybe I'll say WD strategies.

[0:30:43](https://youtu.be/b7BAHSV1EBo?t=1843) It's a great question. Any ideas? I'll put it to the class. Yeah?

[0:30:51](https://youtu.be/b7BAHSV1EBo?t=1851) STUDENT: \[INAUDIBLE]

[0:30:52](https://youtu.be/b7BAHSV1EBo?t=1852) IAN BALL: Say it again?

[0:30:54](https://youtu.be/b7BAHSV1EBo?t=1854) STUDENT: \[INAUDIBLE]

[0:30:54](https://youtu.be/b7BAHSV1EBo?t=1854) IAN BALL: OK. So what you have in mind is, what if there's two strategies that give me exactly the same payoff however my opponent plays? Now, we have to be really careful. The key thing, though, is this part of the definition. So if we didn't have this part of the definition, we could have two weak-- we could have two strategies that were both weakly dominant because they give the same payoff however the opponents play. But two strategies that give the same payoff are not going to satisfy strict inequality for any way my opponents are playing.

[0:31:24](https://youtu.be/b7BAHSV1EBo?t=1884) So it's not possible for two strategies-- for one strategy to weakly dominate another and for that strategy to weakly dominate the first. And for that reason, it's impossible for there to be multiple weakly dominant strategies. So the answer is no. Let's go through the argument again. Suppose I had two strategies that were both weakly dominant. Let's call them SA and SB. Well, that means SA must weakly dominate SB. And SB must weakly dominate SA. But that means SB must do weakly weekly better than SA no matter how my opponents do, but sometimes strictly better.

[0:32:00](https://youtu.be/b7BAHSV1EBo?t=1920) But if SB sometimes does strictly better, then it can't be the case that SA weakly dominates SB. Maybe a bit fast, it might be a good little exercise to write down. But that's a great question. And it's good to keep in mind. And that's one of our motivations for demanding strict inequality for some S negative i, to rule out this case of multiple weakly dominant strategies. Great. Any other questions? Yeah?

[0:32:23](https://youtu.be/b7BAHSV1EBo?t=1943) STUDENT: Just to clarify, for the \[INAUDIBLE] for strictly dominates, is it just that the \[INAUDIBLE]?

[0:32:35](https://youtu.be/b7BAHSV1EBo?t=1955) IAN BALL: Yeah. So what I mean is let's list-- for each S negative i, we have an inequality that looks like this. So let's say we have 10 of them. And I'm just going to go through each of them and say, does equality hold in every single one of those inequalities? And to get weak dominance, equality cannot hold in every single one. So another way of saying it is there must be one inequality where it's strict. But it could be equality in all but one. That's right. Yeah, so when I say strict, I'm referring to that being strict.

[0:33:06](https://youtu.be/b7BAHSV1EBo?t=1986) Great. Any other questions?

[0:33:13](https://youtu.be/b7BAHSV1EBo?t=1993) So now, let's look at a slightly more complicated game than the prisoner's dilemma and see how we would approach this.

[0:33:27](https://youtu.be/b7BAHSV1EBo?t=2007) Let me make sure I get \[INAUDIBLE] two on top. So we have a two player game.

[0:33:41](https://youtu.be/b7BAHSV1EBo?t=2021) So remember, we, always put player 1 here. Player 1 is always choosing rows. Player 2 is always choosing columns. And when we fill in the payoffs, we always put player 1's payoffs first. So here, player 1 has three strategies. We call them top, middle, and bottom, just to make it easy to remember. And player 2 also has two strategies, which we call left and right. So for what I'm going to do here, I really only care about player 1's payoffs. So to avoid just distractions. I'm only going to fill in player 1's payoffs.

[0:34:16](https://youtu.be/b7BAHSV1EBo?t=2056) So there should be something here. Maybe I'll put this to indicate, if I wanted to finish filling in the game, I'd have to put in all those numbers. But for what I'm doing now, I only care about these first numbers.

[0:34:30](https://youtu.be/b7BAHSV1EBo?t=2070) So let's ask, in this game, does any strategy dominate any other? Does it strictly dominate, weakly dominate any other for player 1?

[0:34:43](https://youtu.be/b7BAHSV1EBo?t=2083) What do you think?

[0:34:50](https://youtu.be/b7BAHSV1EBo?t=2090) How do you feel about this game? Would you be happy to play any of these strategies? Or do you think some of them seem worse than others?

[0:35:03](https://youtu.be/b7BAHSV1EBo?t=2103) OK, maybe we'll go on. But the answer is nothing dominates anything else. Let's go through it. Does T dominate M? Well, you might think so because 2 is greater than 0. But over here, we have negative 1 and 0. So M does better than T if my opponent plays R. So it can't be the case that T dominates M. And similarly, does M dominate B? Well, even though M does better than B when my opponent plays L, M does worse than B when my opponent plays R.

[0:35:34](https://youtu.be/b7BAHSV1EBo?t=2134) And you can go through each of these comparisons and see that no strategy weakly dominates any other strategy. So now, we have a bit of a conundrum. How should you play in this game? It's not so clear. Before, it was pretty easy. In the prisoner's dilemma, we said, if you're rational, play F. You don't have to worry about your opponent. However they play, you're strictly better off playing F. But here, it's not so clear. So how would you approach this problem, if you were player 1 here? What would be relevant to you? What would you need to think about?

[0:36:05](https://youtu.be/b7BAHSV1EBo?t=2165) Yeah?

[0:36:05](https://youtu.be/b7BAHSV1EBo?t=2165) STUDENT: You want to know what the chances are the other player plays \[INAUDIBLE].

[0:36:10](https://youtu.be/b7BAHSV1EBo?t=2170) IAN BALL: Exactly. So you're going to have to form beliefs about how the other player plays, just like on the first class when we said whether you walk home or take the T depends on, after checking your phone, what you think the probability is that it's going to rain. In the same way, we're going to form beliefs or assign probabilities to the strategies that my opponent is playing. So in this context, maybe we'll assign a probability of p to L and 1 minus p to R. So now, let's compute my payoffs I'm going to write u of L, p.

[0:36:44](https://youtu.be/b7BAHSV1EBo?t=2204) So a few things here-- because my opponent only has two strategies, if I want to specify my belief about their strategy, it's enough to specify a single number p. And I want to be clear here. This is a number p. It's not a vector. It's just a number. Once I know how much probability I assigned to L, then the amount of probability I assign to R is always just 1 minus that probability. So this is enough information. And then here, remember, technically-- I'll put 1 here-- we only define the utility u1 of a strategy profile.

[0:37:15](https://youtu.be/b7BAHSV1EBo?t=2235) So technically, we haven't defined what this means yet. But we're going to extend our notation to define what my expected payoff is if I play L and I believe that my opponent is going to play L with probability p and R with probability 1 minus p. Yes?

[0:37:30](https://youtu.be/b7BAHSV1EBo?t=2250) STUDENT: You're playing T or M or R \[? instead. ?]

[0:37:33](https://youtu.be/b7BAHSV1EBo?t=2253) IAN BALL: Ah, yes, T, sorry. Thank you. T, yes. Great. So here we are. So what is it/ Well, with probability p, my opponent plays L. And therefore, my payoff is u1 of T, L. But with probability 1 minus p, my opponent plays R. I'm still playing T. Thanks for the clarification. So we have T, R.

[0:38:04](https://youtu.be/b7BAHSV1EBo?t=2284) So notice, we're treating my beliefs about my opponent's strategy just like we treated lotteries before. When I have beliefs about something, I think about my choices as corresponding to lotteries over outcomes. And then I evaluate expected utility over those outcomes. So here, we can just plug in the numbers. We're going to get this is going to be a 2. And this is going to be a negative 1. So what do we get? We get 2p minus 1 plus p, if we do the math right, which equals 3p minus 1.

[0:38:43](https://youtu.be/b7BAHSV1EBo?t=2323) And it's often good to check if this makes sense. The higher is p-- yes?

[0:38:49](https://youtu.be/b7BAHSV1EBo?t=2329) STUDENT: So I just \[INAUDIBLE] question-- the utility, in this case, is specifically only for the lottery, which is represented by the value p. But we also have to include strategy, right?

[0:38:58](https://youtu.be/b7BAHSV1EBo?t=2338) IAN BALL: Right. So if we want to be a little more formal, what is this/ This is the lottery. Well, this is a two player game. So this is a lottery over the strategies of my opponents. So this lives in delta of, in this case, S2, but, more generally, S negative 1, because what I need to do is I need to say, more generally, of all the possible profiles of strategies that my opponents could play, I need to form a belief that specifies the probability on each of them.

[0:39:29](https://youtu.be/b7BAHSV1EBo?t=2369) In this case, I can represent that belief by a single number. But more generally, it's going to be a more complicated object.

[0:39:34](https://youtu.be/b7BAHSV1EBo?t=2374) STUDENT: So that means utilities usually calculate the cross-product of a strategy \[INAUDIBLE].

[0:39:40](https://youtu.be/b7BAHSV1EBo?t=2380) IAN BALL: Yes, yes. And I'll be more formal in a second, yeah. Other question, yeah?

[0:39:44](https://youtu.be/b7BAHSV1EBo?t=2384) STUDENT: So is the utility function on the left side \[INAUDIBLE]?

[0:39:51](https://youtu.be/b7BAHSV1EBo?t=2391) IAN BALL: Good point, good point. People are being very, very precise here. You might say it's an abuse of notation. Or you might say we're extending the notation on the right-hand side. So originally, we defined utilities only for strategy profiles. And now, we're going to define-- formally, what we're doing is we're going to extend this function to a larger domain. So we're going to imagine that it has the same meaning on the original domain. But now, we're going to define it more broadly. So maybe if I was being very pedantic, I would use different notation here. But we're going to be a little less pedantic.

[0:40:22](https://youtu.be/b7BAHSV1EBo?t=2422) And we're going to think of these utilities as being defined on a bit more general space. But you're right that, conceptually, there is a distinction here. Though, one way to think about it is, how do I interpret u1 of T, L? One interpretation is this is what happens when I play T and I hold the belief that my opponent is certain to play L. So you can think of this as a special case where p equals 1. That would be one way you can see how these functions fit together. Yeah. Any other questions? Yes?

[0:40:46](https://youtu.be/b7BAHSV1EBo?t=2446) STUDENT: Can you just use big U on the left-hand side and then little u on the right-hand side?

[0:40:50](https://youtu.be/b7BAHSV1EBo?t=2450) IAN BALL: I could do that. I'm not doing that here. I could do it. But eventually, we just find it easier to always do this. So from now on, we're going to use little u. And we can put in a belief or a strategy over here. And if we put in a strategy, the interpretation is that's a belief that puts all the probability on that particular strategy. Yeah. Any other questions here? Good. OK. And we could similarly compute-- I won't do it here. But we could compute u1 of M, p and u1 of B, p.

[0:41:24](https://youtu.be/b7BAHSV1EBo?t=2484) Then I guess the question suggested we need to be a little more rigorous about what these beliefs are and what we mean. So let me bring this down and erase.

[0:41:43](https://youtu.be/b7BAHSV1EBo?t=2503) So let's be a little more precise about what a belief is.

[0:41:48](https://youtu.be/b7BAHSV1EBo?t=2508) Well, a belief for player i is just a probability distribution over the strategies that the other players are playing. So I'll say, for player i, it's a function. Maybe we'll say beta negative i. We're going to use the negative i because even though it's the belief that's held by player i, it's a belief about everyone else's strategies. And that's why we use the negative i here. And what is it? Well, it's a function from S negative i to 0, 1, if you want to be really formal, where it says, for each profile of strategies, or partial profile of strategies, that my opponents play, what is the probability that I assign to that particular profile of strategies?

[0:42:33](https://youtu.be/b7BAHSV1EBo?t=2553) How likely is it that they played that particular profile? Now, what we need-- we have to be careful. For this to make sense, when we sum up all these probabilities, we have to get 1. So it's a function like this that's satisfying what? Well, the total probability is 1. So let me take a summation. What am I summing over? I'm summing over all the partial strategy profiles of my opponent. So I'm going to sum over s negative i in big S negative i.

[0:43:04](https://youtu.be/b7BAHSV1EBo?t=2584) If you're not used to seeing summations that aren't just numbers 1 to n, think of this as just I'm writing out addition with one term for each strategy profile S negative i. And we have beta negative i of S negative i equals 1.

[0:43:26](https://youtu.be/b7BAHSV1EBo?t=2606) So this is formally what a belief is. And now, let me formally define the function that people rightly challenged me on a little bit. So if I want to formally extend this function, we're going to define ui of Si beta negative i. So remember, so far, we only defined player i's utility from a strategy profile. Now, we're going to think about what is player i's expected utility if player i plays strategy Si and player i holds this belief about everyone else's strategies.

[0:44:00](https://youtu.be/b7BAHSV1EBo?t=2640) We're going to take expectations. So we're going to need to do a summation. We always sum over things we don't know. What we don't know is how my opponents play. So this is going to be a sum over s negative i in S negative i. And we're taking an expectation. So for each way that my opponents could play, I want to consider the probability that that happens. And then I want to multiply that by the utility I get if that does happen. So what's the probability that S negative i happens?

[0:44:31](https://youtu.be/b7BAHSV1EBo?t=2671) It's beta negative i of S negative i. And I want to multiply this by the utility I get if that happens. Well, what's my utility? Well, I'm still playing Si. If it turns out my opponents play S negative i, then my utility is ui of Si, S negative i. Let me maybe draw a line here to make it clear this is a separate line. And this is the formal definition-- or the formal extension of this function to strategies and beliefs.

[0:45:04](https://youtu.be/b7BAHSV1EBo?t=2704) Now, we can also see here what we mentioned earlier. What if beta negative i is the special belief that puts probability 1 on a particular strategy of my opponents? If that happens, then all of these terms become 0 except one of them. And I simply get the same utility function that I started with, where S negative i is exactly the strategy that my belief puts probability 1 on. Any questions about this? Yeah?

[0:45:33](https://youtu.be/b7BAHSV1EBo?t=2733) STUDENT: So in this case, just for the left side, we have the cross-product of \[INAUDIBLE] strategy \[INAUDIBLE] while the ui on the right side is just purely \[INAUDIBLE].

[0:45:42](https://youtu.be/b7BAHSV1EBo?t=2742) IAN BALL: Exactly right. So this lives in S. And this lives in Si cross delta of S negative i. But I don't want people to be scared by the math. But yeah, that's where these things live. Great. OK. So now we've talked about beliefs, now we can talk about a best response to a belief. And this is, I think, one of the most central ideas in game theory.

[0:46:16](https://youtu.be/b7BAHSV1EBo?t=2776) Mathematically, it's not very deep. But conceptually, it's really, I think, a step forward. In interactive decision problems, the idea is that there's two steps. First, I form a belief about how my opponents are going to play. And then I choose a strategy that's best given the belief that I form about everyone else. So we want to formally define that. So we say strategy Si is a best response to belief beta negative.

[0:46:54](https://youtu.be/b7BAHSV1EBo?t=2814) And remember, for any of this to make sense, we have some game in the background. We have some game in the background. We have some player. And if I wanted to be really precise, I would say, for player i, strategy Si is a best response to belief beta negative i, because we're evaluating things here from the perspective of player i. Player i chooses a strategy for herself and forms beliefs about everyone else's strategy. Well, what do we need? Well, we want Si to do better than any other strategy given the belief.

[0:47:28](https://youtu.be/b7BAHSV1EBo?t=2848) So we need ui of Si.

[0:47:42](https://youtu.be/b7BAHSV1EBo?t=2862) So what am I going to-- I'll fill it in a second. I'm player i. I'm comparing the utility that I get from Si to the utility I get from Si prime. And I want Si to be better than Si prime for any Si prime that I contemplate. But what goes in here? Well, what goes in here is the fixed belief that I have. So this is beta negative i. And this is beta negative i.

[0:48:10](https://youtu.be/b7BAHSV1EBo?t=2890) So given the belief beta negative i, I can compute my expected utility from any strategy. And we say that strategy Si is a best response to my belief if the expected utility I get from Si is weakly larger than the expected utility I get from any other strategy Si prime. Now, let's go back to the question that was asked earlier. Can I have multiple best responses to a belief? Is that possible? Can two strategies both be best responses to the same belief?

[0:48:47](https://youtu.be/b7BAHSV1EBo?t=2927) Yes. So this goes back to the suggestion before. If I had two strategies, Si and Si prime, that gave me exactly the same utility no matter what my opponents did, then it's possible for them both to be best responses. In fact, they could both be best responses even if they didn't give me exactly the same utility, but the expected utility was the same. So let's note here-- there can be multiple best responses to the same belief.

[0:49:28](https://youtu.be/b7BAHSV1EBo?t=2968) If we wanted there to only be one, we sometimes might use the phrase strict best response to say this is a best response and nothing else is a best response. But we'll generally just use the weak sense. OK. Let's try to understand this graphically. I think there's space. Maybe let's do this over here. So let me just erase this.

[0:49:55](https://youtu.be/b7BAHSV1EBo?t=2995) STUDENT: \[INAUDIBLE]

[0:50:00](https://youtu.be/b7BAHSV1EBo?t=3000) IAN BALL: Yeah. Yes, so delta is the space of lotteries over the set, yeah.

[0:50:09](https://youtu.be/b7BAHSV1EBo?t=3009) And let me actually just erase this to give myself a bit more space.

[0:50:16](https://youtu.be/b7BAHSV1EBo?t=3016) So let's try to understand best responses in this game. Best responses to what? Well, best responses for player 1 to player 2's-- to player 1's belief about player 2's strategy. So what we want to do is we're going to draw some payoffs here.

[0:50:36](https://youtu.be/b7BAHSV1EBo?t=3036) I'm going to draw a graph here where, on the horizontal axis, I'm--

[0:50:44](https://youtu.be/b7BAHSV1EBo?t=3044) maybe it's better if I put p on R. So let me just switch this so that we're geometrically better.

[0:51:00](https://youtu.be/b7BAHSV1EBo?t=3060) So I can describe my belief by my probability on L or my probability on R. It's more convenient to put the probability on R because then the graph is going to move left to right. And that's going to be a bit better. So p down here is the probability on R. And I'm going to draw this graph from 0 to 1. So if p equals 1, then I'm certain that my opponent is playing R. If p equals 0, then I'm certain that my opponent is playing L. And say if p is 1/2, then I think it's equally likely that my opponent plays L and that my opponent plays R.

[0:51:34](https://youtu.be/b7BAHSV1EBo?t=3094) You can see why I switch things so that we move from left to right. I think it's a bit more intuitive. And now, to think about best responses, I basically want to plot each of these functions. I want to say, if I play T, what is my utility from T as a function of the belief that I have about my opponents? So let's start. Let's put negative 1, 0, 1, 2. If I play T, well, if my opponent plays L, I get a payoff of 2.

[0:52:09](https://youtu.be/b7BAHSV1EBo?t=3129) So that means, if I'm certain that my opponent is going to play L, if that's my belief, then my payoff better be 2. So we're going to get a point up here. And if I'm certain that my opponent is playing R, then I get a payoff of negative 1. So it better be that we get a point like this. And it turns out-- we could calculate it. But it turns out expected utilities are always linear in my beliefs. So once we know these two points, we can just fill it in. It's going to be a line like this.

[0:52:40](https://youtu.be/b7BAHSV1EBo?t=3160) So this is u1 of T, p. And remember, if you did the formula earlier in your notes, we've switched p to 1 minus p. So the formula in your notes was for the old convention. So that no longer applies. What about B? Well, B is symmetric. If I play B, then if I know my opponent is playing left, I get negative 1. And if I know my opponent is playing right, I get 2.

[0:53:12](https://youtu.be/b7BAHSV1EBo?t=3192) Let me test my drawing skills here. And it's going to be linear. So we're going to get something like this. What about M? What is the graph of M going to look like? It's going to look a bit different, right? What will that line look like? Yeah?

[0:53:29](https://youtu.be/b7BAHSV1EBo?t=3209) STUDENT: \[INAUDIBLE]

[0:53:30](https://youtu.be/b7BAHSV1EBo?t=3210) IAN BALL: It's just a horizontal line at 0 because, if I play M, whatever my opponent does, I get a payoff of 0. So whatever I believe my opponent's doing, I'm going to get a payoff of 0. So now, we have a horizontal line here at 0. And if we're careful, this intersection should be at 1/2. You can check that. So let me finish writing these out.

[0:54:03](https://youtu.be/b7BAHSV1EBo?t=3243) So to be clear, each of these is a different function of p. We have three different functions of p, where p is a number between 0 and 1. And I've graphed all three of the functions here. So from the graph, we should be able to say a lot about best responses. Can anyone read off from the graph what my best response is?

[0:54:25](https://youtu.be/b7BAHSV1EBo?t=3265) Let's go through it, right? Let's fix a belief. A belief is some point on the horizontal axis. Now, once I fix that belief, if I want to understand my payoff from my different strategies, I'm going to draw a vertical line upwards. And as I pass up through that vertical line, each curve I intersect is telling me my expected payoff from that particular strategy given this belief. So at this point, what it tells me is my lowest payoff is from B. M gives me my middle payoff.

[0:54:56](https://youtu.be/b7BAHSV1EBo?t=3296) And then T gives me the highest payoff. So at any beliefs where the highest curve is T, T is my best response. At any beliefs where B is the highest curve, B is my best response. So let's look at this. I should have gotten out the colored chalk.

[0:55:19](https://youtu.be/b7BAHSV1EBo?t=3319) That'd be nice if I have it.

[0:55:23](https://youtu.be/b7BAHSV1EBo?t=3323) Maybe no colored chalk today. Ah, yes.

[0:55:30](https://youtu.be/b7BAHSV1EBo?t=3330) So what I'm actually going to do-- the easiest thing to do is I'm going to shade in what's sometimes called the upper envelope. I'm going to go through all these curves. And I'm going to look at what's highest. So if I'm here, this is the highest curve. It comes down. Here, it changes. Now, this becomes the highest curve. And I get something like this. So now, you can see down here-- can anyone now read off from the graph what my best responses are as a function of my beliefs?

[0:56:01](https://youtu.be/b7BAHSV1EBo?t=3361) So when is T a best response? For which beliefs is T a best response? Yeah?

[0:56:06](https://youtu.be/b7BAHSV1EBo?t=3366) STUDENT: From 0 to 1/2 probability of choosing R. And then from 1/2 to 1, it's better to use B instead.

[0:56:14](https://youtu.be/b7BAHSV1EBo?t=3374) IAN BALL: Great, so exactly right. And a key point to make at the end, these are inclusive. So at 1/2, I actually have two best responses. T and B are both best responses at 1/2, as you pointed out. And if my belief is below 1/2, then T is the unique best response, or the strict best response. And if p is above 1/2, then B is the unique best response or the strict best response. Great. Does this make sense/ Well, yes, because T does really well if my opponent plays left. So if my probability on right is small enough and I think it's likely enough that my opponent is playing left, then T is the best option.

[0:56:51](https://youtu.be/b7BAHSV1EBo?t=3411) If it's likely enough that my opponent is playing R, then B is the best option. And we see that B is the best response-- is a best response. But now we have a predicament, which is, what about M?

[0:57:09](https://youtu.be/b7BAHSV1EBo?t=3429) M is not a best response to any belief. Does this mean we shouldn't play M? What do you think about M? Would you ever play it?

[0:57:20](https://youtu.be/b7BAHSV1EBo?t=3440) Yeah?

[0:57:21](https://youtu.be/b7BAHSV1EBo?t=3441) STUDENT: If you're super uncertain about your beliefs \[INAUDIBLE], you might consider M.

[0:57:27](https://youtu.be/b7BAHSV1EBo?t=3447) IAN BALL: We have to be really careful with those words. What you're describing is something important that we're not going to cover in this class, which is called ambiguity aversion. So I don't want to go too far down this path. But we're going to stick with expected utility. And risk aversion would not actually cause you to choose M. Something called ambiguity aversion would. But we're not going to talk about that. So we're imagining that we have our beliefs well-defined about how the other person is doing. So I think it's actually fair to say I don't really want to play M. There's no belief against which M is a best response.

[0:58:00](https://youtu.be/b7BAHSV1EBo?t=3480) But I think it's natural to think back to dominance. What we said before is you might want to throw out a strategy if it's dominated by another strategy. But now, we're in this weird middle ground, where strategy M is not a best response to any belief, but it's also not dominated by any other strategy. And that is a weird conundrum. But we're going to show that, in fact, M is dominated. We just have to be a little more careful. Can anyone see a sense in which M is dominated? Is there anything I could do that would always do better than M?

[0:58:32](https://youtu.be/b7BAHSV1EBo?t=3512) Yeah?

[0:58:32](https://youtu.be/b7BAHSV1EBo?t=3512) STUDENT: \[INAUDIBLE] strategy that basically varies on whatever \[INAUDIBLE].

[0:58:41](https://youtu.be/b7BAHSV1EBo?t=3521) otherwise, that strategy \[INAUDIBLE].

[0:58:43](https://youtu.be/b7BAHSV1EBo?t=3523) IAN BALL: OK, so I like the first part of what you said about a mixed strategy. But crucially, with a mixed strategy, we can't vary our mixing probability based on our belief. But the mixed idea is crucial. So what if I said, instead of choosing strategy M, I'm going to flip a coin. Do you want to add something?

[0:58:59](https://youtu.be/b7BAHSV1EBo?t=3539) STUDENT: \[INAUDIBLE]

[0:58:59](https://youtu.be/b7BAHSV1EBo?t=3539) IAN BALL: OK. Imagine I flipped a coin. And if the coin is heads, I'm going to play T. So consider the strategy which we'll call 1/2 T plus 1/2 B. What does this mean? I'm going to flip a coin. If it's heads, I play-- well, if it's heads I play T. That's a bit weird. If it's tails I play T. If it's heads, I play B. If I were to do that-- let's add this strategy down here. So this is 1/2 T plus 1/2 B. If my opponent plays L, what is my payoff from this mixed strategy where I flip a coin?

[0:59:41](https://youtu.be/b7BAHSV1EBo?t=3581) What's my expected payoff? Well, with probability 1/2 I play T and I get 2. With probability 1/2 I play B and I get negative 1. So my expected payoff is 1/2 of 2, which is 1, plus 1/2 negative 1, which is minus 1/2. And if I add it up, I get 1/2. And now, let's go to this side. If I do my same coin flip, with probability 1/2, I'm going to get negative 1. With probability 1/2, I'm going to get 2.

[1:00:14](https://youtu.be/b7BAHSV1EBo?t=3614) So maybe I'll do the addition here. It's 1/2 times 2 plus 1/2 times negative 1. It's easier to distribute and say, well, 2 minus 1 is 1. So it's 1/2 times 1, which is 1/2.

[1:00:31](https://youtu.be/b7BAHSV1EBo?t=3631) Well, now, is M dominated? So in this case, we see that M is not dominated by any pure strategy. But it is dominated by what's called a mixed strategy. There's a way I can flip a coin so that, if I flip a coin in this way, well, if my opponent plays L, I get 1/2 from this mixed strategy and I only get 0 from m. And if my opponent plays R, I get 1/2 from this mixed strategy and I only get 0 from M. And in fact, whatever belief I have about however my opponents play, I do strictly better under this mixture.

[1:01:05](https://youtu.be/b7BAHSV1EBo?t=3665) We could even write down the payoff of this mixture. It's actually going to look exactly like this. If we wanted to say, what is my expected payoff from this mixed strategy as a function of my belief? So what I have here-- let me put it a bit separate-- is u1 of 1/2 T plus 1/2 B comma p. So this says, if I believe my opponent is playing right with probability p and I flip a coin and play T with probability 1/2 and B with probability 1/2, what is my payoff going to be?

[1:01:36](https://youtu.be/b7BAHSV1EBo?t=3696) And you can see the dominance relation because this is always higher than this. So do we think people really play mixed strategies in practice? We can debate about it. But I want to point out that this resolves this seeming paradox at first where, on the one hand, M was never a best response to any belief. So it seemed like a really bad strategy. But we couldn't directly show what would always be better than it. By using mixed strategies, we can. We see that, in fact, M is dominated. It's, in fact, strictly dominated, just not by a pure strategy.

[1:02:06](https://youtu.be/b7BAHSV1EBo?t=3726) It's strictly dominated by a mixed strategy. So let's formally define a mixed strategy. It's what you would think, but be a little careful.

[1:02:26](https://youtu.be/b7BAHSV1EBo?t=3746) So what is a mixed strategy? It says, well, just like a belief, it's a probability distribution, or a lottery. But it's not a lottery about other people's strategies. It's a lottery about my own strategy. So a mixed strategy for player i is a function-- or is a probability distribution.

[1:02:50](https://youtu.be/b7BAHSV1EBo?t=3770) We often use the notation sigma i. So we use Si for a pure strategy. And sigma i, it's like the Greek version of S. We use that for a mixed strategy. Well, now, it's going to be a function from Si to 0, 1 that says, for each possible strategy, what is the probability that I play that strategy? But once again, we need another condition for this to make sense. The probabilities have to sum up to 1. So it's a function satisfying--

[1:03:21](https://youtu.be/b7BAHSV1EBo?t=3801) well, if I sum over all my strategies, the probability I put on that strategy, that better sum to 1.

[1:03:35](https://youtu.be/b7BAHSV1EBo?t=3815) And I asked before, do people ever play mixed strategies? In a lot of economic contexts, they don't. But there's actually one domain where mixed strategies are crucial. Does anyone know what's a domain where people are very careful to play mixed strategies?

[1:03:51](https://youtu.be/b7BAHSV1EBo?t=3831) Or are there times where you've played a mixed strategy?

[1:03:57](https://youtu.be/b7BAHSV1EBo?t=3837) Yeah?

[1:03:57](https://youtu.be/b7BAHSV1EBo?t=3837) STUDENT: Rock, paper, scissors.

[1:03:59](https://youtu.be/b7BAHSV1EBo?t=3839) IAN BALL: Rock, paper, scissors is a good one. And if you played in rock, paper, scissors in competitions, yeah, that would probably be really important. There's another domain where people make a lot of money where randomization is also important. Yeah?

[1:04:09](https://youtu.be/b7BAHSV1EBo?t=3849) STUDENT: Poker.

[1:04:09](https://youtu.be/b7BAHSV1EBo?t=3849) IAN BALL: Professional poker. So professional poker players are extremely good at randomization. And they train to randomize well. Humans are very bad at randomizing naturally. And there's all these techniques professional poker players make. 15 years ago, you could play poker and win money without being good at randomizing. Today, you would just get destroyed because the best players are extremely good at randomizing. And they study game theory. And they randomize extremely carefully. And basically, the people who are better at randomizing with exactly the right probabilities win a lot of money. So poker is a great example of mixed strategies that really happen.

[1:04:39](https://youtu.be/b7BAHSV1EBo?t=3879) If I taught this course 20 years ago, it would be hard to come up with an example of people playing mixed strategies. And now, the reality has caught up to the game theory. And now, there's a lot of people who make a lot of money by mixing. OK. So now, let's move on a little bit and try to understand how general this phenomenon is. So I want to make one observation here. And then we want to try to see how general that is. So if you're approaching a game like this and you're player 1, there's two ways you could go about it. If I were to ask you what is a reasonable strategy to play, you could say, well-- so let me say, let's look at player 1.

[1:05:18](https://youtu.be/b7BAHSV1EBo?t=3918) And I'll just say, which strategies?

[1:05:23](https://youtu.be/b7BAHSV1EBo?t=3923) Well, one thing you could do is you could start with the good strategies. And maybe I'll call them the reasonable strategies.

[1:05:33](https://youtu.be/b7BAHSV1EBo?t=3933) This is not really a formal term. But I'll just say reasonable strategies. And these would be the strategies that are best responses to some belief.

[1:05:52](https://youtu.be/b7BAHSV1EBo?t=3952) So I don't want you to think too much, for the moment, about how you form your beliefs. But let's just say, is there some belief I could have that would make this a best response? So in this game, we saw in the graph which strategies in this game are best responses to some belief. So which strategies are reasonable in this sense?

[1:06:11](https://youtu.be/b7BAHSV1EBo?t=3971) Yeah?

[1:06:12](https://youtu.be/b7BAHSV1EBo?t=3972) STUDENT: T and B.

[1:06:13](https://youtu.be/b7BAHSV1EBo?t=3973) IAN BALL: T and B, right? So I might say, look, I have three strategies. But the reasonable ones are T and B. Another way you could approach it, though, is instead of looking for what you should do, you could try to look for what you don't want to do and avoid that. So I could say, what are the unreasonable strategies?

[1:06:35](https://youtu.be/b7BAHSV1EBo?t=3995) And by an unreasonable strategy, I mean a strategy that's strictly dominated by some other strategy, possibly mixed.

[1:07:00](https://youtu.be/b7BAHSV1EBo?t=4020) Now, technically, I only defined dominance for pure strategies. But everything makes sense if I just replace the Si's with the sigma i's and I interpret these utilities as expected utilities. So in this game, what are the unreasonable strategies? What are the strategies I could throw out by saying they're dominated by something else?

[1:07:21](https://youtu.be/b7BAHSV1EBo?t=4041) Yeah?

[1:07:22](https://youtu.be/b7BAHSV1EBo?t=4042) STUDENT: M.

[1:07:22](https://youtu.be/b7BAHSV1EBo?t=4042) IAN BALL: Just M, right? So notice here, there's two different ways to get to the same answer. I could say, let me only consider strategies that are best responses to some beliefs. I would go through. And I would just get T and B. Or I could do the inverse. I could say, let me throw out all the strategies that are dominated, that are strictly dominated by something. I would throw out M. And I'd be left with just T and B. And the question is, is this a coincidence? Is this a special property of this game? Or is this more generally true, that we can always split our strategies into two groups, those that are best responses to some belief and those that are strictly dominated by some mixed strategy?

[1:08:03](https://youtu.be/b7BAHSV1EBo?t=4083) And it turns out this is a very general property. And that'll be the first theorem that we state in this course. So let me state the theorem.

[1:08:30](https://youtu.be/b7BAHSV1EBo?t=4110) So here's our theorem. We need some assumptions. So let's say we're in a finite strategic form game.

[1:08:40](https://youtu.be/b7BAHSV1EBo?t=4120) Finite just means the set of strategies for every player is finite. And there's finitely many players.

[1:08:58](https://youtu.be/b7BAHSV1EBo?t=4138) So for each player i and each strategy Si--

[1:09:11](https://youtu.be/b7BAHSV1EBo?t=4151) so before I write it, let's just understand. We fixed our finite strategic form game. We're considering a particular player i and a particular strategy Si. And what we want to argue, what we want to say, is that this strategy falls in exactly one of these two categories. It can't be in both. And it can't be in Neither. So what we say is exactly one of the following holds.

[1:09:42](https://youtu.be/b7BAHSV1EBo?t=4182) Exactly one means not both and not neither.

[1:09:57](https://youtu.be/b7BAHSV1EBo?t=4197) And one is that this strategy Si is reasonable in this sense. Meaning what/ Si is a best response to some belief.

[1:10:17](https://youtu.be/b7BAHSV1EBo?t=4217) And when I say belief, what is that a belief about? It's a belief-- maybe I'll say over S negative i. Let's remind ourselves, when player i forms beliefs, they form beliefs over the strategy profiles of the other players. And then the other case is Si is strictly dominated.

[1:10:43](https://youtu.be/b7BAHSV1EBo?t=4243) By what? Well, by some mixed strategy. And we have to allow mixed. If we only allowed pure strategies, we already saw up here that the theorem wouldn't be true because there are strategies that are never best responses, but are not dominated by any pure strategy. So we have to allow for mixed strategies. Si is strictly dominated by some mixed strategy.

[1:11:09](https://youtu.be/b7BAHSV1EBo?t=4269) And of course, when I say strategy, that has to be a strategy of player i because player i compares two different strategies when they look at strict dominance. But wait a second, I said Si is strictly dominated by some mixed strategy. What about pure strategies? Am I including those?

[1:11:27](https://youtu.be/b7BAHSV1EBo?t=4287) STUDENT: \[INAUDIBLE]

[1:11:28](https://youtu.be/b7BAHSV1EBo?t=4288) IAN BALL: Why?

[1:11:29](https://youtu.be/b7BAHSV1EBo?t=4289) STUDENT: \[INAUDIBLE]

[1:11:35](https://youtu.be/b7BAHSV1EBo?t=4295) IAN BALL: Exactly right. So every pure strategy is a mixed strategy. But not every mixed strategy is a pure strategy. So as long as I cover mixed strategies-- in particular, I'm including the mixtures that put all the probability on a single strategy, which are simply pure strategies. So if we want to say that I'm not doing that, sometimes we say properly mixed strategy or nonpure mixed strategy. But here, we're allowing all kinds of mixed strategies, including the ones that are just pure strategies in disguise.

[1:12:09](https://youtu.be/b7BAHSV1EBo?t=4329) And again, I think the interpretation of this is that there are two very natural ways of thinking about a game and which strategies someone might play. I think it would be very natural to say, it's reasonable to play anything that's a best response to some belief. And it would also be natural to say, here's what you should do. Just don't play anything that's strictly dominated by something else. And if those two approaches gave us different answers, we'd have to really think hard about which one was more reasonable. But it turns out these are literally going to give us the same answer.

[1:12:42](https://youtu.be/b7BAHSV1EBo?t=4362) So you can think positively about what's the best response to something or you can think negatively about what's strictly dominated by something. And either way, you're going to get the same two groups of strategies-- the good ones and the bad ones.

[1:12:59](https://youtu.be/b7BAHSV1EBo?t=4379) Let me say a little bit about how we prove this. The proof won't be covered on exams. But I want to give you some idea.

[1:13:13](https://youtu.be/b7BAHSV1EBo?t=4393) So what's the argument? Proof? Well, how do you show that exactly one thing happens? You have to show that both can't happen and neither can't happen. So another way of saying it is at most one--

[1:13:29](https://youtu.be/b7BAHSV1EBo?t=4409) so at most one of these two things can be true and at least one.

[1:13:38](https://youtu.be/b7BAHSV1EBo?t=4418) So to say at most one of the statements is true is another way of saying both statements can't be true-- at most one, not two. At least one is a way of saying it can't be the case that neither is true, at least one, either one or two, but not zero. And by one or two, I mean the number-- maybe I'll call these A and B to avoid confusion.

[1:14:03](https://youtu.be/b7BAHSV1EBo?t=4443) I'm talking about at most one, meaning the number of statements that are true, not the labels of the statements.

[1:14:11](https://youtu.be/b7BAHSV1EBo?t=4451) So one of these directions is a bit easier. One of these is quite tricky. Does anyone see one that they think is kind of intuitive and they could maybe-- I won't ask you for the proof, but could reason through on their own?

[1:14:26](https://youtu.be/b7BAHSV1EBo?t=4466) Yeah?

[1:14:27](https://youtu.be/b7BAHSV1EBo?t=4467) STUDENT: At most one.

[1:14:27](https://youtu.be/b7BAHSV1EBo?t=4467) IAN BALL: At most one. So this is the easier direction-- not easy maybe, but easier. And what's your intuitive argument? We won't give a formal one.

[1:14:36](https://youtu.be/b7BAHSV1EBo?t=4476) STUDENT: \[INAUDIBLE]

[1:14:39](https://youtu.be/b7BAHSV1EBo?t=4479) IAN BALL: We're going to get a contradiction. So if both were true, you're saying this is the best we can do against some belief. But there's something else that does strictly better whatever my opponent does. Well, if it does strictly better, whatever my opponent does, it should also do strictly better given my belief. But that means the other thing couldn't be a best response. So I think it's pretty intuitive. I'd encourage you to work out the algebra on your own. I think this is a good exercise to do at home. The at least one is a lot harder. And that I wouldn't expect you to be able to figure out at home. There is a proof given in the appendix of the lecture notes.

[1:15:11](https://youtu.be/b7BAHSV1EBo?t=4511) And actually, let me clarify. I said I won't test you on the proof. But I could test you on this part of the proof. I think the top is fair game. The bottom is not. Let me be clear about that. So the bottom, for the harder part, see appendix. And it uses a mathematical result called either the separating hyperplane theorem, which you may have heard of, or something called Farkas' lemma is also my preferred way of proving it.

[1:15:43](https://youtu.be/b7BAHSV1EBo?t=4543) And just to see why this is harder, well, if Si-- we want to show at least one is true. So we have to say if Si is not a best response to some belief, then it's strictly dominated by something. But that means the proof requires you to construct something. And that's what's hard. All we know is it's not a best response to any belief. But how do we know what dominates it? Well, how do we come up with that. That's pretty hard. So this theorem gives you a way of constructing either the belief you need or the dominating strategy that you need. And that's much more involved. So that won't be covered in the course.

[1:16:13](https://youtu.be/b7BAHSV1EBo?t=4573) But I encourage you to read the appendix if you want to see that. And let me stop there. And I'll see everyone next week. Good luck on the quiz tomorrow.
