---
title: 'lecture 11: one shot deviation principle and bargaining'
description: 'In this lecture, Ian Ball discusses multi-stage games and introduces the tool called the one-shot deviation principle.'
id: 'mit-14-12-fall-2025-resource-resources-mit14-12f25-lec11-1080p-mp4'
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
    n: 11,
    youtube: 'iL3XRRI5rQs',
    duration: 4705,
    cues: 1593,
    captions: 'static_resources/1a0r8j9_sZF8SA59joegTqYjrkEdDd6VQ_transcript.webvtt',
    transcript: 'static_resources/1a0r8j9_sZF8SA59joegTqYjrkEdDd6VQ_transcript.pdf',
  }
layout: 'resource'
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
---

[0:00:10](https://youtu.be/iL3XRRI5rQs?t=10) IAN BALL: OK, so today we're going to continue our study of subgame-perfect Nash equilibria. And specifically, we're going to look at multistage, something called multistage games. And the key technical tool that we're going to introduce today is something called the one-shot deviation principle.

[0:00:39](https://youtu.be/iL3XRRI5rQs?t=39) So I'll go into detail about what this is later in the class. But it's going to allow us to compute or to check that a strategy profile is, in fact, a subgame-perfect Nash equilibria, even in games that have an infinite horizon. And using this tool, we're then going to revisit one of the applications we looked at before, this price haggling game, where you have a buyer and a seller haggling over what the price is going to be that they exchange the good. And if you recall, when we first modeled that game, we had this artificial time at which the world ended because we didn't have the tools to analyze an infinite horizon game.

[0:01:10](https://youtu.be/iL3XRRI5rQs?t=70) But now we're going to apply these techniques to be able to fully analyze the infinite horizon version of bargaining. So I want to start by saying that what I did last class was true but was maybe a little more subtle. And I want to point out that there's kind of-- I want to give a cautionary note about some of the reasoning we did last class. So we looked at games maybe like this. Let's do this. Enter and exit game where we had first player 1 chose whether to exit or enter. If they exited, let's say, each player got 1.5.

[0:01:42](https://youtu.be/iL3XRRI5rQs?t=102) And then if they entered-- let's say they played the Boston game, so player 2 moved and chose Celtics or Red Sox. And then player 1 moved and chose Celtics or Red Sox. And then the payoffs were, let's see, player 1, let's see, 2, 1, 1, 2, 0, 0, and 0, 0. And we argued that one equilibrium of this game is the equilibrium where we have XR, R.

[0:02:22](https://youtu.be/iL3XRRI5rQs?t=142) And we argue that this is a subgame-perfect Nash equilibrium. And let's see how we reasoned about that. Well, we said there's two subgames. There's the entire game, and there's the subgame that starts here. So the first step is always to identify our subgames. And then we said, let's find a Nash equilibrium within this subgame, within this smaller subgame. Remember, we always start at the end. We start with the smaller subgames that come at the end, and we work backwards. And we know that this subgame is the standard BOS game that we studied before.

[0:02:55](https://youtu.be/iL3XRRI5rQs?t=175) And we know that one equilibrium of this game is that both players go to the Red Sox game. So we filled in that equilibrium.

[0:03:06](https://youtu.be/iL3XRRI5rQs?t=186) And then we took a step back. We said, well, if this is the equilibrium that's being played in this subgame, then the payoff in this subgame is going to be 1, 2. We said if this subgame is reached, that's what the payoffs are going to be. And then we said if I'm player 1, well, I'm effectively choosing to exit and get a payoff of 1.5 or to enter and get the payoff in the Nash equilibrium of this subgame and get 1. So I'm better off exiting, and we said great. This is an SPNE. So that's true. Everything is correct.

[0:03:36](https://youtu.be/iL3XRRI5rQs?t=216) But I want to argue that there's a little subtlety here. And the subtlety is that we of skipped a step. And what we didn't think about is-- and I'll say, what about complicated deviations by player 1?

[0:04:04](https://youtu.be/iL3XRRI5rQs?t=244) What do I mean by this? Well, we check that this strategy constitutes a Nash equilibrium in the subgame. That's pretty clear. But we also have to check that this constitutes an equilibrium in the full game because part of the definition of subgame-perfect Nash equilibrium is that the strategy profile must constitute a Nash equilibrium within every subgame, and the full game is one particular example of a subgame. So what we have to show is that player 1 does not have a profitable deviation in the full game. Now, what did we actually show?

[0:04:35](https://youtu.be/iL3XRRI5rQs?t=275) What we showed is, well, if I'm player 1, and I deviate to enter and then continue the strategy down here, that won't be profitable. But when we filled in this subgame and treated the payoff as 1, 2, we basically fixed the way that player 1 plays in this subgame. And in principle, player 1 could consider a more complicated deviation. What they could do is they could say, well, what if I enter the game and then deviate as well in this subgame and play differently in this subgame? How do we know that that's not going to be profitable?

[0:05:06](https://youtu.be/iL3XRRI5rQs?t=306) All we checked is that if they enter and then play as they're supposed to in the subgame, that's not profitable. But we haven't checked what happens if player 1 enters and then modifies their behavior in the subgame. So we didn't explicitly check that. I argue that that's OK. But why? Why do we not need to worry about these more complicated deviations? Does anyone see why? I think this is a subtle point, so if people are confused, feel free to ask a question.

[0:05:35](https://youtu.be/iL3XRRI5rQs?t=335) Yeah.

[0:05:36](https://youtu.be/iL3XRRI5rQs?t=336) AUDIENCE: Is it because we already performed backward induction on the subgame?

[0:05:40](https://youtu.be/iL3XRRI5rQs?t=340) IAN BALL: Exactly right. So what happens in one of these complicated deviations? We choose N here, but then instead of following the play that we're supposed to use in this subgame, we then deviate again in this subgame. But we already know that, in this subgame, our play constituted a Nash equilibrium. So this further deviation can't be any better because it involves suboptimal play within this subgame given the way the other player is playing. So it turns out what about these more complicated deviations?

[0:06:11](https://youtu.be/iL3XRRI5rQs?t=371) The answer is they can't be profitable.

[0:06:18](https://youtu.be/iL3XRRI5rQs?t=378) And why? Because we know that our strategy profile already constitutes a Nash equilibrium within this subgame. And therefore, given the way the other player is playing, we're already doing optimally in this subgame. So if I were to enter and then play suboptimally in the subgame, that can't make me better off because of backward induction, I'll say.

[0:06:45](https://youtu.be/iL3XRRI5rQs?t=405) And it's, quote, unquote, "backward induction" because we can't technically apply backward induction to this game, but it's the reasoning of working backwards. So maybe this is just so intuitive that everyone already intuited this. I think it is kind of intuitive, but I just want to highlight that there is kind of a subtlety going on. And in class today, because we're going to deal with games that have an infinite horizon, we have to be even more careful about the subtlety. Yes.

[0:07:07](https://youtu.be/iL3XRRI5rQs?t=427) AUDIENCE: So why isn't CC the Nash equilibrium?

[0:07:10](https://youtu.be/iL3XRRI5rQs?t=430) IAN BALL: I didn't say this is the only subgame-perfect Nash equilibrium. This is one subgame-perfect Nash equilibrium. There's also a subgame-perfect Nash equilibrium where the players play CC and the first player enters. That's a different subgame-perfect Nash. But here, I'm just questioning have we fully verified that this is a subgame-perfect Nash equilibrium? And I'm just arguing that it is. But the verification is a little more subtle than maybe you might think at first. Yeah. Any other questions on this?

[0:07:45](https://youtu.be/iL3XRRI5rQs?t=465) So it turns out that dealing with all these complicated deviations in really big games can be tricky. And we're now going to study a class of games where the analysis becomes pretty simple. And this is a class of games that covers a lot of applications of interest. And these are going to be called multistage games.

[0:08:06](https://youtu.be/iL3XRRI5rQs?t=486) And I want to emphasize that these possibly have an infinite horizon.

[0:08:21](https://youtu.be/iL3XRRI5rQs?t=501) So what is the definition? What is a multistage game? Well, it's a game that has multiple stages, so let's define what that means. So it's a game where we have stages, maybe 0, 1, 2, dot dot dot, and it could either be finite. It could either go to T, or it could go on forever. So I'm allowing both of those possibilities. So it could be finite or infinite. Both are allowed.

[0:08:52](https://youtu.be/iL3XRRI5rQs?t=532) But what happens in these stages? Well, there's two key properties of these stages. So in each stage, some subset of players are going to move simultaneously.

[0:09:06](https://youtu.be/iL3XRRI5rQs?t=546) So in each stage, some subset of players move simultaneously.

[0:09:28](https://youtu.be/iL3XRRI5rQs?t=568) And then the other key assumption is that at the beginning of each stage, everyone observes all the moves that were made prior to that stage.

[0:09:40](https://youtu.be/iL3XRRI5rQs?t=580) So at the beginning of each stage all prior moves are observed.

[0:09:59](https://youtu.be/iL3XRRI5rQs?t=599) So sometimes these are called multistage games with observed actions to emphasize the second point. But I don't want to use that terminology because if we look within a stage, we have multiple players moving simultaneously. So at the stage where the player's moving simultaneously, they're not observing the other player's actions. So it's not actually a game of perfect information because we have these simultaneous moves within the game. Does anyone remember any examples of multistage games or any games that we've studied so far constituted multistage games? Yes.

[0:10:29](https://youtu.be/iL3XRRI5rQs?t=629) AUDIENCE: Could this be the one where firms have to pick entry and then they pick quantities after they observe?

[0:10:34](https://youtu.be/iL3XRRI5rQs?t=634) IAN BALL: Exactly right. So in that Cournot game with entry cost, that was exactly a multistage game that had two stages. Remember, in the first stage, the firms simultaneously chose whether to enter the market. That was stage-- I guess we can call it stage 0 if you want to start at 0. And then, in the second stage, a subset of players, namely the firms who had chosen to enter in the first stage, then simultaneously chose what quantities to produce. And we made the critical assumption that, at the beginning of that second stage, everyone observed the moves in the first stage, that is, all the firms observed, which subset of firms had entered in the first stage.

[0:11:11](https://youtu.be/iL3XRRI5rQs?t=671) And maybe I should be a little more precise when I say some subset of players move simultaneously. It could be that the subset that moves simultaneously depends on what happened in previous stages. So if you look at that Cournot entry game, if firms 1 and 2 entered in the first stage, then firms 1 and 2 play simultaneously in the second stage. If firms 3 and 4 entered in the first stage, then firms 3 and 4 made choices simultaneously. So some subset, which maybe I'll say, it could be history dependent, where history means what happened in the previous stages.

[0:11:52](https://youtu.be/iL3XRRI5rQs?t=712) Another example, actually, is this game here. So how can we think of this game as a multistage game?

[0:12:00](https://youtu.be/iL3XRRI5rQs?t=720) Walk me through why we can think-- I mean, I model it as an extensive-form game, but how can we think of it as a multistage game? Well, in the first stage, or-- it's always awkward to say with 0. In the zero stage, in the initial stage, player 1 makes a move. Player 1 is a subset of players. It's a subset of players with just 1 player. That's the only player who moves. So maybe let's pull this down and look at this. So if we want to represent this as a multi-stage game, this is stage 0.

[0:12:33](https://youtu.be/iL3XRRI5rQs?t=753) And in stage 0, the only player who moves is player 1. Then we can think of this here as stage 1. Why? Well, remember, technically, we said player 2 moves first and then player 1 moves after. But because player 1 doesn't observe the move that player 2 makes, it's as if players 1 and 2 are moving simultaneously. I could say player 1 moves and then player 2 moves not observing player 1's move. The order doesn't matter as long as the players don't observe what the other player's doing.

[0:13:05](https://youtu.be/iL3XRRI5rQs?t=785) So effectively what's happening in this game is the players are playing a simultaneous-move game within stage 1, and that simultaneous-move game is exactly the Boston game that we studied. Now we have to check one more thing. Is it true that, at the beginning of stage 1, the players know what happened in stage 0? Yes. If we look at this subgame, both players know that player 1 must've played enter because, when I'm player 1 here, maybe I don't know which node I'm at within this information set, but I certainly know that I'm on this branch of the tree, and therefore, I know what happened in stage 0.

[0:13:39](https://youtu.be/iL3XRRI5rQs?t=819) So this is a special case that we can think of as an example of a multistage game.

[0:13:47](https://youtu.be/iL3XRRI5rQs?t=827) So this is quite a flexible general class of games. And it turns out that it's very easy in this class of games to check whether a strategy profile is a subgame-perfect Nash equilibrium. And that's what we're going to go over next.

[0:14:13](https://youtu.be/iL3XRRI5rQs?t=853) So the next result is what's called the one-shot deviation principle.

[0:14:25](https://youtu.be/iL3XRRI5rQs?t=865) And maybe it's helpful to first, before I state the result, describe what I mean by a one-shot deviation. So what is a one-shot deviation? Well, let's suppose a player has some strategy s in the game. And I want to say, what does it mean for s prime-- maybe I'll call this i for s prime to be a one-shot deviation from s.

[0:15:01](https://youtu.be/iL3XRRI5rQs?t=901) Well, in principle, I could deviate from si in a very complicated way. There's many, many different histories in the game. The game's really complicated. I could make a lot of different deviations. A one-shot deviation means I only deviate once. Or more precisely, I only deviate at one history or one contingency. So in multistage games, we often refer to information sets as histories because, whenever a player is called upon to play in a multistage game, whatever stage it's in, what they know is exactly what happened in all the previous stages.

[0:15:40](https://youtu.be/iL3XRRI5rQs?t=940) So their information set is a history of play in all previous stages. So a one-shot deviation from si if-- what do we want to say? We want to say si prime of h is not equal to si of h at one history h.

[0:16:05](https://youtu.be/iL3XRRI5rQs?t=965) But si prime of h equals si of h prime prime at all other histories h prime. So remember, a strategy is a complete contingent plan. It says how player i is going to play at every history or every contingency at which they're asked to move. So when I compare two strategies, I can ask how different are these strategies? At how many histories do these strategies prescribe different behavior?

[0:16:38](https://youtu.be/iL3XRRI5rQs?t=998) And we say that si prime is a one-shot deviation from si if there's exactly one history at which they prescribe different behavior. And then, at all the other histories, they prescribe exactly the same behavior.

[0:16:55](https://youtu.be/iL3XRRI5rQs?t=1015) But we have to be really, really careful about what this means at one history. So even if I modify my strategy at a single history, it could mean that play changes at many histories. So let's go back to our example of the Cournot game to understand what this means.

[0:17:23](https://youtu.be/iL3XRRI5rQs?t=1043) Suppose we have a player who, in the first stage, was not supposed to enter the game, or they weren't entering the game. And now they contemplate a one-shot deviation. So let's say firm i deviates to enter at the initial history.

[0:17:48](https://youtu.be/iL3XRRI5rQs?t=1068) So previously, the firm was not entering at this initial history at the beginning of the game. Now they enter. This is a one-shot deviation. They're only changing their strategy at that initial history. But now because they've entered at that initial history, now the way everyone plays in the second period becomes quite different. So even though if the firm is deviating at a single history, it changes play at all subsequent histories.

[0:18:25](https://youtu.be/iL3XRRI5rQs?t=1105) Why? Well, now a different firm, a different number of firms have entered the game, so the quantities that every firm is going to produce in the second period has now changed because the quantities that they produce depend upon how many firms enter. So even if the deviation only changes behavior at one history, that then changes which future histories we reach and therefore can change a lot of things in the game.

[0:18:52](https://youtu.be/iL3XRRI5rQs?t=1132) And we'll see that a bit later. All right.

[0:19:00](https://youtu.be/iL3XRRI5rQs?t=1140) So now that we've defined what a one-shot deviation is, we can actually-- let me give a little more notation. We've said si prime is a one-shot deviation from si. We can be a little more precise. If si prime differs from si at precisely the history h, we can call this a one-shot deviation at h. It's a one-shot deviation where the strategy specifies different play precisely at h. So we'll maybe say a one-shot deviation from si, and maybe we'll add at h if we want to be more precise about which history player i is deviating at.

[0:19:36](https://youtu.be/iL3XRRI5rQs?t=1176) OK.

[0:19:46](https://youtu.be/iL3XRRI5rQs?t=1186) So now I think the final ingredient we need before we can state the one-shot deviation principle for multistage games is to keep track of, well, what are the subgames of a multistage game?

[0:20:09](https://youtu.be/iL3XRRI5rQs?t=1209) Remember, that's always the first step. When you want to try to identify or solve for a subgame-perfect Nash equilibrium, you have to say, what's the set of subgames? Any thoughts on this? What are the subgames going to be? Remember, to find subgames, you just have to find nodes in the game that can start a subgame. So what are the set of possible subgames in this multistage game? And it may be helpful to think we know we studied this Cournot entry game on Tuesday. That was a multistage game, and we identified the subgames of that game, so that might guide you about what the subgames are in this game.

[0:20:42](https://youtu.be/iL3XRRI5rQs?t=1242) Well, let's go through step by step. What about in stage 0? How many subgames are there that start in stage 0?

[0:20:53](https://youtu.be/iL3XRRI5rQs?t=1253) There's just one-- the entire game. We start in stage 0. Nothing has happened so far. So this is just the whole game.

[0:21:04](https://youtu.be/iL3XRRI5rQs?t=1264) What about in stage 1?

[0:21:10](https://youtu.be/iL3XRRI5rQs?t=1270) How many subgames start in stage 1?

[0:21:19](https://youtu.be/iL3XRRI5rQs?t=1279) Well, remember, in the Cournot game, we had a subgame associated with every subset of firms that could enter in the previous stage. And it's going to be the same idea here that, in stage 1, every history of play in stage 0 is going to initiate a new subgame starting in stage 1. So stage 1, we have a subgame for each stage 0 history.

[0:21:48](https://youtu.be/iL3XRRI5rQs?t=1308) That is, however people played in stage 0, we all know that's how people played because we observe it. And now we have a subgame starting at that history of play from here on from here onwards. What about in stage 2?

[0:22:07](https://youtu.be/iL3XRRI5rQs?t=1327) Maybe I'll call this h0, where h0 says how people played in stage 0. What about in stage 2? How many subgames do we have in stage 2? Now it's getting a little more complicated. Well, what do we know in stage 2? We know how people played in stage 0. And we know how people played in stage 1. So any history of play in stage 0 and in stage 1 is going to start a subgame in stage 2. So now we have this for each, maybe I'll say, history.

[0:22:39](https://youtu.be/iL3XRRI5rQs?t=1359) Maybe I'll call it h0, h1 that says, this is how we played in stage 0. This is how we played in stage 1, and now we're going to move on to stage 2. And in general, we're going to see that, in stage t, we're going to have a much more complicated history h0, h1, up to ht minus 1. So at stage t, we know how everyone played up to stage t minus 1. And since we know that, that's going to start a new subgame from stage t onward.

[0:23:14](https://youtu.be/iL3XRRI5rQs?t=1394) So we see there's going to be a lot of histories. And it's going to be helpful. We'll just use a generic term history to mean a history at any point. So it could be a history here that just says how we played in h0. It could be a history here that says how we played in 0 and 1. It could be a history here, how we played up in here. But we'll look at all those together, and we'll just look at arbitrary-- we'll call something a history if it's a history at any stage.

[0:23:39](https://youtu.be/iL3XRRI5rQs?t=1419) So what are the subgames of a multistage game? They are described by the histories of the game. So we have the--

[0:23:52](https://youtu.be/iL3XRRI5rQs?t=1432) maybe I'll call it the-- we have an h subgame for every history h.

[0:24:06](https://youtu.be/iL3XRRI5rQs?t=1446) So give me a history h. It could be a history like this. It could be history like this. It could be history like this. That's going to start a subgame from here on out. And conversely, any subgame is exactly going to have this form. So we have an association between the subgames of a multistage game and the histories of play in the game. Just so we can make this concrete, in this example, we had one subgame starting in stage 0, which was the subgame following the null history where nothing had happened.

[0:24:41](https://youtu.be/iL3XRRI5rQs?t=1481) And then we actually had one subgame starting in stage 1 following the single history n. And that was the only possible history at which we were called to play in stage 1. So that's how we map it into this example over here.

[0:24:58](https://youtu.be/iL3XRRI5rQs?t=1498) OK, so now that we've stated that we can now give our theorem on the one-shot deviation principle.

[0:25:08](https://youtu.be/iL3XRRI5rQs?t=1508) So let's consider a multistage game.

[0:25:19](https://youtu.be/iL3XRRI5rQs?t=1519) You say, for any strategy profile s star, the following are equivalent.

[0:25:41](https://youtu.be/iL3XRRI5rQs?t=1541) So we're going to list some statements about a strategy profile s star that are equivalent. So all the statements hold, or none of the statements hold. So the first statement is what we're interested in. It's that s star is a subgame-perfect Nash equilibrium. So this is what we're interested in. You give me a strategy s star, a strategy profile s star, we want to understand, is this strategy profile a subgame-perfect Nash equilibrium? Now, before we really do anything, let's just understand the definition of a subgame-perfect Nash equilibrium.

[0:26:13](https://youtu.be/iL3XRRI5rQs?t=1573) So the first step is going to be pretty easy. I just need to apply the definition of subgame-perfect Nash equilibrium to this specific multistage game. So we know that, what does it mean for s star to be a subgame-perfect Nash equilibrium? It means, in any subgame, the restriction of the strategy profile to that subgame constitutes a Nash equilibrium in that subgame. But now let's use our observation over here about the structure of the subgames. So instead of making a statement about every subgame, we can make a statement about every history because a subgame is basically equivalent to a history.

[0:26:49](https://youtu.be/iL3XRRI5rQs?t=1609) So we'll say if and only if, for every history h--

[0:26:58](https://youtu.be/iL3XRRI5rQs?t=1618) well, what we want to say is that the restriction of s star to the h subgame is a Nash equilibrium of the h subgame. So we'll say, for every history h, s star h is a Nash equilibrium of the h subgame, where what is the h subgame? This is what I described up here. It's the subgame that starts at history h, where the players have observed the path of play along this history h up to the current stage.

[0:27:31](https://youtu.be/iL3XRRI5rQs?t=1651) And then they can choose how to behave from here onward. What is s star h? This is notation for the restriction of s star to the h subgame.

[0:27:47](https://youtu.be/iL3XRRI5rQs?t=1667) Remember, we can't say that s star is a Nash equilibrium of the h subgame because s star is not a strategy profile in the h subgame. It specifies play at all these histories that may not be in that subgame. So here we just focus on the part of s star or the part of the strategy in s star that specify behavior within this particular subgame. So so far, we haven't really done anything. So far, we've just stated the definition of subgame-perfect Nash equilibrium in this particular context where the class of subgames has a certain structure.

[0:28:20](https://youtu.be/iL3XRRI5rQs?t=1700) Any questions? I think this lecture can be a little more abstract than some of the other ones, so I want to check in. Any questions about this, about what any of the words mean? It's great if people come after class and ask me, but if you ask me in class, you can help your classmates as well who might be confused too. Yeah.

[0:28:38](https://youtu.be/iL3XRRI5rQs?t=1718) AUDIENCE: What is the difference between these two statements?

[0:28:40](https://youtu.be/iL3XRRI5rQs?t=1720) IAN BALL: Oh, we're getting to the real results later. So this is just preparation. So, yeah, if you think this is obvious, that's fine. It's basically, once you understand the definitions, yeah, we haven't gone much-- so the heart of the theorem is going to be 3.

[0:29:04](https://youtu.be/iL3XRRI5rQs?t=1744) So we're, again, not going to do much in 0.3. We're just going to break down what it means for s star h be a Nash equilibrium of the h subgame. So this means that, for every history h-- I'll just use this--

[0:29:21](https://youtu.be/iL3XRRI5rQs?t=1761) no player has a profitable unilateral deviation from s star h in the h subgame.

[0:29:54](https://youtu.be/iL3XRRI5rQs?t=1794) So here, I still haven't really done anything. I've just written out the definition of Nash equilibrium. What does it mean for s star h to be a Nash equilibrium of the h subgame? It just means that in the h subgame, no player can profitably deviate unilaterally from the strategy profile. If a player starts at the strategy they're supposed to play, and they deviate to a different strategy, that can't be profitable within the subgame, just the definition of Nash equilibrium. So what makes this a theorem and what makes this hard is the final step. But I just want to make clear the structure of this.

[0:30:25](https://youtu.be/iL3XRRI5rQs?t=1825) So now comes the actual result. And what it says is, at every history, no player has a profitable unilateral one-shot deviation at history h.

[0:30:58](https://youtu.be/iL3XRRI5rQs?t=1858) One-shot deviation, maybe I'll say, from s star h at history h in the h subgame.

[0:31:12](https://youtu.be/iL3XRRI5rQs?t=1872) So this is where the theorem comes in. Up here, we're just saying no player has a profitable deviation of any kind. It could be a really complicated deviation where they deviate at many, many histories. Statement four says, in fact, no player has a profitable one-shot deviation at history h. And that means they're going to deviate-- we're only considering deviations to strategies that differ from their equilibrium strategy exactly at history h. And at every other history, it specifies the same behavior.

[0:31:45](https://youtu.be/iL3XRRI5rQs?t=1905) So let's just understand one direction of this is easy. So I think the equivalence of 1 through 3--

[0:31:59](https://youtu.be/iL3XRRI5rQs?t=1919) 1, 2, and 3 are equivalent by definition.

[0:32:08](https://youtu.be/iL3XRRI5rQs?t=1928) The equivalence of 1 and 2 follows is just a statement about the definition of subgame-perfect Nash equilibrium and the structure of the subgames. The equivalence of 2 and 3 is just the statement of the definition of Nash equilibrium. We're just breaking down what it means for something to be a Nash equilibrium. It means no player has a profitable unilateral deviation. And then the implication from 3 to 4 is also immediate. Why?

[0:32:40](https://youtu.be/iL3XRRI5rQs?t=1960) Why is this immediate?

[0:32:48](https://youtu.be/iL3XRRI5rQs?t=1968) Yeah.

[0:32:49](https://youtu.be/iL3XRRI5rQs?t=1969) AUDIENCE: Because a one-shot deviation is a type of unilateral deviation.

[0:32:52](https://youtu.be/iL3XRRI5rQs?t=1972) IAN BALL: Exactly, if no deviation is profitable, if I already know that there's no deviation that's profitable, well, certainly, there can't be a profitable one-shot deviation because that's just a special kind of deviation. So again, not a trick question, just really easy. If we know nothing is profitable, then certainly this special kind of thing can't be profitable, just logic. So the heart of the theorem, which we're not even going to prove because it's actually quite tricky is 4 to 3. This is I don't know, hard, important, substantive I'll say.

[0:33:26](https://youtu.be/iL3XRRI5rQs?t=2006) Because 3 to 4 is clear-- if no deviation is profitable, then clearly a one-shot deviation can't be profitable. But the other direction is not clear. Just because I can't profit with a one-shot deviation, how do I know that I can't profit with a much more complicated deviation? So the idea is if a simple deviation is not profitable, where I'm using simple colloquially, then a complicated deviation also can't be profitable. Notice that was exactly the issue when we talked here about complicated deviations. We observed that, in the last class, we actually only checked, quote, unquote, "simple deviations."

[0:34:01](https://youtu.be/iL3XRRI5rQs?t=2041) We didn't check that more complicated deviations weren't profitable. But I argued that, intuitively, if the simple deviations aren't profitable, then we know that the more complicated ones aren't profitable. And that's exactly the same logic here, but it's just mathematically a lot more tricky because the game can possibly go on forever. But the substantive idea is still there. So we're not going to prove this.

[0:34:25](https://youtu.be/iL3XRRI5rQs?t=2065) But it's true. And what it means is that it becomes much easier to check whether something is a subgame-perfect Nash equilibrium, so given this theorem, if I give you a strategy profile, say in a quiz or an exam or a game, and you want to understand, is this a subgame-perfect Nash equilibrium, you don't have to check that there's-- that no deviation is profitable. You only have to worry about the one-shot deviations. So it allows you-- so what's the upshot?

[0:34:56](https://youtu.be/iL3XRRI5rQs?t=2096) The upshot is that, to verify a subgame-perfect Nash equilibrium.

[0:35:03](https://youtu.be/iL3XRRI5rQs?t=2103) To verify that a strategy profile is an SPNE, it's enough to check the one-shot deviations.

[0:35:17](https://youtu.be/iL3XRRI5rQs?t=2117) That one-shot deviations are not profitable.

[0:35:27](https://youtu.be/iL3XRRI5rQs?t=2127) Because if I know that the one-shot deviations aren't profitable, the theorem tells me that no deviations are profitable, and then I'm done. Now, that sounds nice. We only have to check the one-shot deviations, but this is actually still quite complicated because we have to check the one-shot deviations still in every subgame. So checking one-shot deviations may not be so hard, but notice, we still have the quantifier for every history. So, yes, we only have to check one-shot deviations, but we have to check them at every single history of the game.

[0:35:57](https://youtu.be/iL3XRRI5rQs?t=2157) And in general, doing this may be complicated.

[0:36:04](https://youtu.be/iL3XRRI5rQs?t=2164) I should maybe be a little more precise. I don't think this is a big issue in this class. And maybe this demonstrates that this step 4 to 3 is not obvious. You need one technical assumption for this theorem to be true. So I said in any multistage game. I should really say in any multistage game-- I should really say in any continuous multistage game. We're not really going to deal with discontinuous multistage games in this class.

[0:36:37](https://youtu.be/iL3XRRI5rQs?t=2197) But you can look at the notes. I think they say a little bit about the definition of continuous. Continuous just means if the game is infinite, if everyone plays exactly the same way a billion periods into the game, then how they play beyond a billion periods shouldn't have a big effect on their payoffs? So basically, if we want to understand what people's payoffs are in the game, we just have to look at how they play, say, a billion periods into the game, and that'll give us a pretty good approximation for what their payoffs will be. It can't be that things, 2 trillion periods into the future determine everything.

[0:37:11](https://youtu.be/iL3XRRI5rQs?t=2231) So a classic example of what will satisfy this is if we just sum up the payoffs in each period with a discount factor because then things that happen really far in the future get discounted a lot. And therefore, they don't play much of a role. But this is kind of a technical thing that won't be covered in-- everything we study in this class will be continuous. You don't have to worry about this, but I'm happy to talk after if you have questions about this. OK.

[0:37:38](https://youtu.be/iL3XRRI5rQs?t=2258) If you like math, I think this is a nice theorem to think about and think about how you would prove it, but that's not going to be something we'll cover in this class. OK, so now the question is, how do we use this theorem? And we're going to use this theorem to analyze a classical game, just like we talked about before of price haggling but now with a potentially infinite horizon.

[0:38:17](https://youtu.be/iL3XRRI5rQs?t=2297) So now we're going to look at maybe what I'll call infinite-horizon alternating-offer bargaining.

[0:38:41](https://youtu.be/iL3XRRI5rQs?t=2321) So let's just go through the words. Let's start with alternating offer. This is just like we talked about before. We have two parties there negotiating or bargaining-- I'll use these terms interchangeably-- over something. One side comes to the table. They say, here's my proposal. The other side says either, great, we're done. Or the other side says, no, I'm rejecting your proposal. I'm going to come back with my proposal. And then the other side makes a proposal. Now, that can either be accepted or rejected, and then we go again. And we keep alternating our offers. You might see this if you're at a used car dealership.

[0:39:11](https://youtu.be/iL3XRRI5rQs?t=2351) You might say, I'll pay \$10,000 for the car. The dealer says, no, it's 15,000. You say, what about 12,000? He says, oh, what about 14,000? And you converge. Infinite horizon means, well, we don't literally think that negotiations will go on forever, but there's no clear last period. Whenever we're negotiating, it's always possible that the offer can be rejected, and our negotiations or bargaining could continue one more period. So the infinite horizon allows us to rule out these kind of weird last-period effects that we saw, where we know that this is our very last opportunity to negotiate, and whoever makes an offer in that period has a lot of bargaining power.

[0:39:44](https://youtu.be/iL3XRRI5rQs?t=2384) So we want to shut down that weird last-period effect by saying, in principle, there's no end date. The negotiations could just be ongoing. And we want a very abstract model of this that can capture not just haggling over the price of something, but also, say, in international relations. Right now, there's some negotiations going on in the Middle East. How can we have a very general model that can capture these negotiations? So what we want to do. Well, that's hard because what we're negotiating over may be very complicated. So maybe I'll say the terms of dispute may be complicated.

[0:40:25](https://youtu.be/iL3XRRI5rQs?t=2425) So how can we represent this? Well, what's relevant to the parties is not the details that we're negotiating over, but the utilities we get from those details. So we're going to reduce these complicated terms to utilities. We're going to reduce to maybe utility space. Now, of course, in reality, this is not how we negotiate. But we might say, instead of thinking of terms of negotiation as a 20-point plan, about the terms of some deal, we might say, well, let's represent that 20-point plan by the utility it gives to party 1 and the utility it gives to party 2.

[0:41:00](https://youtu.be/iL3XRRI5rQs?t=2460) Now, of course, in reality, we don't say this. We don't say, here's my offer. You get utility 7, and I get utility 3. No one actually speaks that way. But what we're saying is the strategically relevant aspect of the terms of some offer is simply how much utility each party gets. And we can just treat the offers-- in utility space, we can represent them by the utilities they give to each party. So what we're going to say is we're going to think of utility space. We have player 1's utility and player 2's utility.

[0:41:35](https://youtu.be/iL3XRRI5rQs?t=2495) And the set of all possible agreements we can reach we're simply going to represent by the set of all possible utility pairs we could get. So we're going to have some set x. Let me draw a very simple example of a set, but we could be much more general.

[0:41:56](https://youtu.be/iL3XRRI5rQs?t=2516) So maybe this is our set x and this represents feasible utility pairs.

[0:42:09](https://youtu.be/iL3XRRI5rQs?t=2529) So what do I mean by that? If you give me some point in this set x, there exists some agreement we could reach that would generate these utilities for the two sides. What that agreement looks like I'm not taking a stand on. We're being very abstract, but we just want to say it's possible to achieve this. And then if you give me a point out here that's not an x, what I'm saying is, there is no agreement that could give both sides this much utility. It would be great if there was some deal that makes us both really, really happy, but there's just nothing feasible that would give us this utility.

[0:42:41](https://youtu.be/iL3XRRI5rQs?t=2561) And then the one more thing we have to specify, well, negotiations can always break down. So we have to say what happens if negotiations break down, and we don't reach an agreement. If we don't reach an agreement, maybe we revert to the status quo. We have some disagreement outcome. But again, we want to reduce that disagreement outcome also to utility space. So we often normalize it to the origin. And this is going to be called the disagreement point.

[0:43:07](https://youtu.be/iL3XRRI5rQs?t=2587) And what this says is if we don't reach an agreement, something will happen. I don't know exactly what that is, but whatever it is, we can assign utility to it. And this is going to represent the utilities we get if we don't reach an agreement.

[0:43:23](https://youtu.be/iL3XRRI5rQs?t=2603) It's very standard to write our feasible set x to lie above the disagreement point. Why is that? What if there was some agreement we could make that would give utilities over here? I'm arguing that this is not going to be so relevant to our negotiation. Let me put one down here as well. Why is something like this, some point like this not really going to be relevant to our negotiations? Yeah, in the front.

[0:43:45](https://youtu.be/iL3XRRI5rQs?t=2625) AUDIENCE: Why bargain at all if you can just disagree and get better utility?

[0:43:48](https://youtu.be/iL3XRRI5rQs?t=2628) IAN BALL: I agree. But for whom? Let's be more specific. So if we disagree, we don't both get better utility. Who gets better utility in this example?

[0:43:58](https://youtu.be/iL3XRRI5rQs?t=2638) AUDIENCE: So it would be u2.

[0:44:01](https://youtu.be/iL3XRRI5rQs?t=2641) IAN BALL: Yes, so what I mean here is if we both disagree-- so disagreement-- and maybe I should be more clear. If we disagree and we don't reach a deal, this specifies the utilities that both of us get. So let's understand-- this deal is great for player 2. It's a lot better than the disagreement point for player 2. The problem is that it's worse for player 1. So the idea is player 1 would never accept something like this because player 1 can just walk away from the negotiating table, force the disagreement point, and get this point here. So I should say, when we talk about a disagreement point, there's kind of an implicit assumption here that each player has unilateral power to reach the disagreement point.

[0:44:38](https://youtu.be/iL3XRRI5rQs?t=2678) And this is usually true in negotiations. Any side can just say, I'm done. I'm walking away from the table. And we go down here. OK, what about this point? Why could we never agree on a point like this? Who would veto that? Yeah.

[0:44:51](https://youtu.be/iL3XRRI5rQs?t=2691) AUDIENCE: Player 2.

[0:44:52](https://youtu.be/iL3XRRI5rQs?t=2692) IAN BALL: Player 2 would veto that because player 2 can get this much utility by walking away from the table. And down here, they get a strictly lower utility. So maybe there's some feasible things down here. But we usually don't even include them because they're not relevant to the game. So what's relevant are the set of feasible utilities. They have to be feasible. Otherwise, they're certainly not relevant. And they deliver utility that's weakly better than the disagreement point for both players. So neither party would unilaterally reject this agreement. But the question is, where are we going to be within here?

[0:45:23](https://youtu.be/iL3XRRI5rQs?t=2723) That's the goal of the analysis to understand what's going to happen. In this game, I think it's pretty intuitive. You might think that we'll be somewhere along this line here. It seems intuitive that we don't want to throw away utility. But where? I mean, player 2 would love to be up here. Player 1 would love to be down here. And it's not clear where along this frontier we're going to be.

[0:45:48](https://youtu.be/iL3XRRI5rQs?t=2748) So today, we're going to analyze this special case. We could look at an arbitrary disagreement point and an arbitrary feasible set x and analyze that game. We're going to look at the special case described here. Now you might say, what does this represent? It turns out, this is exactly price haggling, this picture. This is exactly price haggling between a seller and a buyer where the buyer values the good at 1.

[0:46:26](https://youtu.be/iL3XRRI5rQs?t=2786) Let's see why this represents that? What is our disagreement point in price haggling? What can happen if we're a buyer and a seller come together, and they haggle over the price? Well, they can always just walk away and not exchange. So if there's no exchange, we each get utility 0. The buyer doesn't get the good. They don't pay anything. The seller doesn't get any revenue. So we get utility 0. And now I need to decide who's the seller and the buyer. Let's say this is the seller, and this is the buyer.

[0:46:56](https://youtu.be/iL3XRRI5rQs?t=2816) What about this point over here?

[0:47:03](https://youtu.be/iL3XRRI5rQs?t=2823) So at this point, the seller gets utility 1, and the buyer gets utility 0. What does this represent? How could, in our simple price haggling game, where we have value and we have risk-neutral agents get to this point? So this is the best thing for the seller? Yeah.

[0:47:21](https://youtu.be/iL3XRRI5rQs?t=2841) AUDIENCE: The buyer put the max they're willing to pay.

[0:47:24](https://youtu.be/iL3XRRI5rQs?t=2844) IAN BALL: And how much are they willing to pay?

[0:47:26](https://youtu.be/iL3XRRI5rQs?t=2846) AUDIENCE: 1.

[0:47:27](https://youtu.be/iL3XRRI5rQs?t=2847) IAN BALL: 1, so this would be the case where the buyer buys at a price of 1. So if the buyer buys at a price of 1, the seller's revenue is 1. So the seller's utility is 1. The buyer's utility is 0 because they value the good at 1, but they paid 1 for it. So what does it mean to value a good at something? It means if you pay that much for it, it's as if you don't have it. Your utility is exactly the same, so the buyer's utility is 0. What about up here? How do we get to this point? Yeah.

[0:47:52](https://youtu.be/iL3XRRI5rQs?t=2872) AUDIENCE: The buyer got it for free.

[0:47:53](https://youtu.be/iL3XRRI5rQs?t=2873) IAN BALL: The buyer gets it for free. So here, they exchange the good at a price of 0. The buyer's utility is 1 because they get the good for value of 1, but they pay nothing. And the seller's utility is 0 because they don't raise any revenue. Now, you can see we can then move along this line as we vary the price between 0 and 1. As we vary the price between 0 and 1, we're going to move along this line here. Now, I guess, technically, we can also get in this shaded region. How would we get in this shaded region? I mean, this is a bit strange.

[0:48:24](https://youtu.be/iL3XRRI5rQs?t=2904) No one would really do this, but what could we do? Well, technically I could, as the buyer, I could, say, pay you \$0.80 for the good and then burn \$0.10 and just throw it away. We probably wouldn't get that. But often, we want to think of this as a convex set. So technically, we could always burn money and just throw things away and go down here. We don't expect this will happen. But in the monetary context, this maybe seems crazy, but you certainly see in negotiations, people are sometimes willing to put forward proposals that would harm both parties as a threat.

[0:48:57](https://youtu.be/iL3XRRI5rQs?t=2937) So in the monetary context, maybe it doesn't seem reasonable, but it's important to think about the possibility of threats that hurt both parties, and that's why we want to fill this in. OK, so we've seen how this fits into our example. And now we just need to be a little more formal. So if we get an agreement, x1, x2, in X-- so let's go back to the terms.

[0:49:29](https://youtu.be/iL3XRRI5rQs?t=2969) Maybe player 1-- going to be a little more abstract. We can think of this in the price haggling context. But we'll be abstract and just think about players 1 and player 2. If we reach an agreement x1, x2 in X, then, well, we think that player 1 will get utility x1, and player 2 will get utility x2. But we also want to represent the cost of delay. Delay is costly. So suppose we reach an agreement here at time t, where t is going to go 0, 1, 2 on forever.

[0:50:03](https://youtu.be/iL3XRRI5rQs?t=3003) Then the utilities are going to be, well, player 1 gets utility delta to the t x1 and player 2 gets utility delta to the t x2. And here, where delta 0 is less than delta is less than 1. So this is just like we had before. We have a discount factor delta. The later on that we reach an agreement, the less utility each of us gets.

[0:50:32](https://youtu.be/iL3XRRI5rQs?t=3032) And as time passes, things get worse and worse for both of us. If we make an agreement in period 0-- remember, any nonzero number to the 0 is just 1. So if we agree immediately in period 0, we just get x1 and x2. If we agree in period 1, we get delta x1, delta x2. If we agree in period 10, we get delta to the 10 x1, delta to the 10 x2. And again, delta is going to parameterize our patients. So what happens if delta is very close to 1? What does that represent? Is that patience are impatience?

[0:51:03](https://youtu.be/iL3XRRI5rQs?t=3063) That's-- yeah.

[0:51:05](https://youtu.be/iL3XRRI5rQs?t=3065) AUDIENCE: Patience.

[0:51:05](https://youtu.be/iL3XRRI5rQs?t=3065) IAN BALL: Patience, right. If delta's close to 1, even if t is really large, it doesn't hurt us very much. If delta's close to 0, we're very impatient. And in the extreme case, if delta is 0, then, basically, the world ends tomorrow, and the only way we can reach a deal is if we reach a deal in the first period. OK, so this is our model. Now we have to specify the sequence in which we make offers. And I think the standard way we'll do it is we have time 0, 1, 2, 3, and so on.

[0:51:39](https://youtu.be/iL3XRRI5rQs?t=3099) And it's alternating offers, so let's say that, in the even periods, player 1 makes an offer--

[0:51:48](https://youtu.be/iL3XRRI5rQs?t=3108) so maybe I'll say makes an offer.

[0:51:55](https://youtu.be/iL3XRRI5rQs?t=3115) And then player two can either accept or reject that offer.

[0:52:07](https://youtu.be/iL3XRRI5rQs?t=3127) And then, in the odd periods, player 2 is the one who moves.

[0:52:23](https://youtu.be/iL3XRRI5rQs?t=3143) OK, so just to be clear, player 1 makes an offer. If player 2 accepts that offer, that's it. We implement the offer, and we stop. If player 2 rejects that offer, then player 2 comes back and makes their offer. Player 1 can either accept or reject. If they accept it, we make that offer, and we accept that offer, and we stop. If they reject it, we move on to the next period, and we go on like this. Now, what is an offer, though? Remember, in our price haggling game, an offer was a price. We said what price is offered? But more generally, an offer is simply a point in x.

[0:52:56](https://youtu.be/iL3XRRI5rQs?t=3176) Remember, that's how we're going to represent an offer. An offer is to say, let's stop now. Let's reach this agreement, and let's get that utility. So let's be clear that an offer is always a point in x. I've specified the payoffs. If we reach an agreement x in time t, and that's generally how the game will end. At some time t, the offer will be accepted. Some offer will be accepted, and now we can write down what the payoffs are. But I just have to be more careful here.

[0:53:26](https://youtu.be/iL3XRRI5rQs?t=3206) What happens if we just go on forever and nothing is accepted? That's one thing that could happen in the game? So I need to specify what the utilities are in that case? And if it's never accepted, well, then we just move to our disagreement point, which, in this case, is going to be 0, 0. It's never accepted. We go to our disagreement point.

[0:53:53](https://youtu.be/iL3XRRI5rQs?t=3233) And in this case, we'll say that utility is 0, 0. We have to specify that because that's one thing that could happen in the game. Yes.

[0:54:00](https://youtu.be/iL3XRRI5rQs?t=3240) AUDIENCE: So does this represent that because of time decay, we're heading towards 0, 0 asymptotically? Or do we actually hit the point 0, 0? Is it just like, both people are like, OK, let's give up, disagree.

[0:54:14](https://youtu.be/iL3XRRI5rQs?t=3254) IAN BALL: Yeah, I think interpreting what disagreement means in the infinite horizon is a little tricky because when does it actually happen? Yeah so I think there's a few different interpretations. I would maybe interpret the disagreement point-- so one interpretation-- yeah, I don't want to confuse things. One interpretation could be, in each period, until we reach a deal, we just get a flow utility of 0 in that period.

[0:54:50](https://youtu.be/iL3XRRI5rQs?t=3290) And then, once we reach an agreement, we get the agreement utility from that period onward. And then we discount in some R utilities. That would be one interpretation you could give. So it's, really, our payoffs would be like 0, 0, 0, 0, and then, once we reach an agreement of x here, then we get that agreement utility x, x, x, x forever. And then we do the average discounted payoff from that. I don't want to create complications, but if you did that, it would reduce to this model. So that would maybe be one way of thinking about it. Because you're right. If things go on forever, there's not a point where we say, oh, we've disagreed.

[0:55:21](https://youtu.be/iL3XRRI5rQs?t=3321) It just says, well, we didn't reach an agreement today, and that just happens forever. So maybe it's better to think of it in terms of flow payoffs. Good question. OK, so now we'd like to analyze this game. We did the whole one-shot deviation principle to help us study games like this. But it's not even clear that this is a multistage game because players are not moving simultaneously. Players are moving sequentially. Player 1 makes an offer, and then player 2 observes this. So why is this a multistage game.

[0:56:06](https://youtu.be/iL3XRRI5rQs?t=3366) Well, it turns out, it is a multistage game, but it's just the stages are not given by time. So technically, this is stage 0. Player 1 is the only player who moves. This is stage 1. This is stage 2. This is stage 3, 4, 5, 6, 7, 8, 9. So it is a multistage game. The multistage game just requires that, within each stage, a subset of players move simultaneously. Well, in this game, the subset who moves is just a single player, and the history of past play is always observed, so it's a multistage game. So we can now apply the one-shot deviation principle to try to understand subgame-perfect Nash equilibria of this game.

[0:56:43](https://youtu.be/iL3XRRI5rQs?t=3403) Now, just like with our finite case, the strategies in this game are really complicated. What is a strategy for player 2? It has to say, whatever offer I've given in period 0, will I accept or reject that? Then, based on that history, what offer will I make in period 1 for every possible history of play there? And on and on and on. So it's going to have to specify a complete contingent plan that says what I will either accept or offer at every history of past play. That's going to be really, really complicated, but we're going to, hopefully, be able to solve it. And then make a prediction about what's going to happen?

[0:57:15](https://youtu.be/iL3XRRI5rQs?t=3435) Any guesses, I guess, before we really get into it, if we look at our agreement space here, any intuition about where we might end up in terms of payoffs? So any predictions about what might happen? Yeah.

[0:57:31](https://youtu.be/iL3XRRI5rQs?t=3451) AUDIENCE: I mean, just looking at the graph over time, the line will move to the left.

[0:57:35](https://youtu.be/iL3XRRI5rQs?t=3455) IAN BALL: Exactly. That's right. So if we made an agreement in the initial period, we could-- that's the only way we can get on this line. And any intuition about where we might end up? I mean, we'll analyze it, but what do you think might happen in this game? Who's going to do better? Yeah.

[0:57:53](https://youtu.be/iL3XRRI5rQs?t=3473) Make a guess. It's fine. Yeah.

[0:57:54](https://youtu.be/iL3XRRI5rQs?t=3474) AUDIENCE: Well, my thought was, last time, you talked about how it's dependent on the fact that the negotiation was finite. So it would get sped up because both knew that they had to reach an agreement quickly. So I feel like it's going to devolve a little bit more. And maybe if it's infinite, then maybe the first person who has the offer has the advantage.

[0:58:13](https://youtu.be/iL3XRRI5rQs?t=3493) IAN BALL: That's true. So I agree. We still have complete information here. So we talked about, last time, one reason that we're not going to see actually delay and disagreement is that we all agree, for each offer, exactly what utilities that gives us. In reality, if you make a peace proposal in a negotiating plan, you don't know how much utility that's going to give to the other side. So there's been a big assumption here, and that's actually going to be a reason why we're going to see an agreement initially. And indeed, the first mover, player 1, is going to have an advantage in this game because they have the power of moving first.

[0:58:45](https://youtu.be/iL3XRRI5rQs?t=3525) And indeed, there's going to be a bit of a first-mover advantage, so we're going to end up along this line but a little closer to player 1. And the more impatient the players are, the closer we're going to be to player 1 because the more impatient the players are-- let's think of the extreme case just before we get into the analysis. I think this is good, whenever you approach a problem, to try to think through what might happen before you formally analyze it. Is that a question? Yeah, yeah, go ahead.

[0:59:09](https://youtu.be/iL3XRRI5rQs?t=3549) AUDIENCE: Does this only apply if we assume that they have the same level of patience?

[0:59:13](https://youtu.be/iL3XRRI5rQs?t=3553) IAN BALL: Good, so I'm assuming that they have the same level of patience. A very natural extension of the model would be to say that they have different discount factors. And indeed, that would change the results. And there may be a problem set on that or the problem is the algebra gets quite messy, so I often don't assign it. But that's a classic extension. Yes, exactly. Yes.

[0:59:31](https://youtu.be/iL3XRRI5rQs?t=3571) AUDIENCE: Yeah, I mean if delta were 0, then they would just make whatever deal you want.

[0:59:36](https://youtu.be/iL3XRRI5rQs?t=3576) IAN BALL: Exactly, so let's get to that. So let's say delta is 0. So the world ends after tomorrow. Player 1 is making a proposal. What do you think player 1's going to propose? And where are we going to be on this triangle? Yeah.

[0:59:49](https://youtu.be/iL3XRRI5rQs?t=3589) AUDIENCE: I mean, I think that we could make this 0.1 for player 2 but as long as it's above 0.

[0:59:54](https://youtu.be/iL3XRRI5rQs?t=3594) IAN BALL: Right, so we're basically-- maybe there's an issue with 0.1, 0.001, but we're basically going to be here. So player 1's going to have all the power. If the world ends tomorrow, then you as player 2 have to accept whatever I give you. So I'm going to make an offer way down here. I'm going to give you almost no utility, and we're going to be here. And then, intuitively, as the players become very, very patient. Well, now the first-mover advantage that player 1 gets is going to shrink. If we're very patient, the fact that you might have to wait till tomorrow isn't a big deal. And as we get more and more patient, we're going to move, and we're going to converge to the midpoint of here because, as delta goes to 1, the fact that player 1 moves first is not really an advantage at all.

[1:00:33](https://youtu.be/iL3XRRI5rQs?t=3633) And therefore, we expect the symmetric solution where we end up here. And that's exactly what we're going to see. But that's just a preview. Yes.

[1:00:40](https://youtu.be/iL3XRRI5rQs?t=3640) AUDIENCE: Does that mean we're never going to be on the left side of the midpoint where player 2 isn't getting \[INAUDIBLE]?

[1:00:46](https://youtu.be/iL3XRRI5rQs?t=3646) IAN BALL: That's correct. We are not because player 1 moves first. So if player 2 moves first, we would have the mirror image, where player 2 moves first and the discount factor was 0, we'd be up here, and then we would converge down to here.

[1:00:57](https://youtu.be/iL3XRRI5rQs?t=3657) AUDIENCE: So when we talk about converge, we actually hit the midpoint? Or do we just get--

[1:01:02](https://youtu.be/iL3XRRI5rQs?t=3662) IAN BALL: In this case, we will not literally hit it. We will just converge. We will get arbitrarily close as delta goes to S yeah. Because what would happen, you'd want to say we hit it when delta equals 1. Actually, if delta equals 1, we then have to be a little more careful about delta equals 1. So there could be multiple equilibria. So let's not-- we're not going to deal with the exact delta equals 1 case. Yeah. OK. Great, so let's try to analyze this. Let's go down here.

[1:01:46](https://youtu.be/iL3XRRI5rQs?t=3706) So as we said, specifying a strategy in this game is really, really complicated. There's so many histories. But what we're going to look for is a very, very simple strategy where people's behavior doesn't really depend on the details of the past, that, basically, when it's your turn to propose, you basically propose the same thing no matter what else happened. And we're going to check that this is an equilibrium. So here's our conjectured, maybe our candidate, SPNE. So we'll check that this is actually an SPNE. Well, as we said, we think that whoever proposes, whatever history we're at, has some advantage.

[1:02:19](https://youtu.be/iL3XRRI5rQs?t=3739) And what's going to happen is, at any history, well, someone's the proposer. It's either player 1 or player 2, depending on whether it's an even history or an odd history. But let's not worry about that. And any history, the proposer proposes-- well, I'd like to use a vector in x, but the problem is a vector in x tells me the payoffs for player 1 and player 2, and the proposer could be player 1 or player 2.

[1:02:53](https://youtu.be/iL3XRRI5rQs?t=3773) So I really want to think in terms of whoever's proposing and who's responding rather than the actual labels of the players. So let's say the proposer proposes some utility for herself and some utility for the other player, for the respondent. So they're going to propose-- well, let's think they're the proposer. So they have a bit more power. So we think they're going to propose more than 1/2 to themselves and less than 1/2 the other player. And in it's going to be exactly-- this is a formula we saw in class a few weeks ago-- 1 over 1 plus delta.

[1:03:27](https://youtu.be/iL3XRRI5rQs?t=3807) I'll write it this way.

[1:03:31](https://youtu.be/iL3XRRI5rQs?t=3811) Maybe I'll say for herself. And then delta over 1 plus delta for opponent.

[1:03:43](https://youtu.be/iL3XRRI5rQs?t=3823) So I'm proposing. I'm proposing a split. Remember, if we look at this line, along this line, the sum of the components of this vector is always 1. So we're basically thinking, how do we split the 1 unit of utility we have. And I'm going to propose to keep 1 over 1 plus delta for myself as the proposer, and delta over 1 plus delta for the opponent. And let's just understand how this fits in with our intuition that we said before, if delta gets very close to 1, then I'm basically proposing an even split.

[1:04:14](https://youtu.be/iL3XRRI5rQs?t=3854) It's very close to 1/2, 1/2. If delta gets very close to 0, then I'm basically taking everything for myself. So this comports with the intuition that we presented. So in any history, the proposal proposes this. And after any proposal, what does the respondent do?

[1:04:37](https://youtu.be/iL3XRRI5rQs?t=3877) Remember, we can't just say what the respondent does in response to this proposal because we have to specify the respondent strategy at every history, including some proposals that we don't expect to be actually be made. So let's say the respondent faces some proposal, any proposal, maybe I'll say an arbitrary proposal.

[1:05:06](https://youtu.be/iL3XRRI5rQs?t=3906) Any intuition about, well, certainly the respondent should accept if they're offered enough and reject otherwise.

[1:05:19](https://youtu.be/iL3XRRI5rQs?t=3919) So what is enough? Accept if-- it's always tricky with the notation, so what I mean is, accept if what's offered to the respondent-- the respondents component of that vector is large enough. Maybe I'll say, accept if she gets at least what? So meaning there's some offer that's made. The offer specifies a utility for both players. What the respondent cares about is the utility that she's offered, and she gets at least what? What is the critical value she's going to need to get?

[1:05:56](https://youtu.be/iL3XRRI5rQs?t=3956) Any thoughts? Well, in this equilibrium, the proposer is giving this much to the other player. So why are they giving the other player this much? That's the least they can give them that they'll still accept. So it's exactly going to be delta over 1 plus delta.

[1:06:16](https://youtu.be/iL3XRRI5rQs?t=3976) So let's just understand the structure. When I'm proposer, I anticipate that the other player, the respondent, will only accept my offer if I offer to give them at least this much utility. So what am I going to do? Well, I could give them more utility than this, and they'd accept it, but I don't want to do that. That leaves less utility for me. So I'm going to give them the smallest amount of utility that they will actually accept. And this is going to be our strategy. So remember, this specifies a lot. If we're at some really complicated history, whatever's happened, whoever's the proposer is going to propose exactly this split.

[1:06:51](https://youtu.be/iL3XRRI5rQs?t=4011) And if we're down here, whatever's offered, the respondent is going to follow up with exactly this strategy. So this is a complete contingent plan in the game. I didn't write it out really mathematically, but it's capturing a lot of complexity here. Yes.

[1:07:03](https://youtu.be/iL3XRRI5rQs?t=4023) AUDIENCE: I think as long as the proposer proposes \[INAUDIBLE] the first step, the game ends.

[1:07:07](https://youtu.be/iL3XRRI5rQs?t=4027) IAN BALL: Exactly right, which is exactly what we found in the finite horizon game as well. We said there's no disagreement, there's no delay. We immediately end in the first period, and we immediately end with these splits. And these splits exactly correspond to the points along the line that I was describing. So if these strategies are followed, then, in the first period, player 1 proposes this split. And therefore, player 1 gets 1 over 1 plus delta. And as we said, if delta is close to 0, this converges to 1/2. If delta is close to-- sorry, if delta's close to 0, this converges to 1.

[1:07:40](https://youtu.be/iL3XRRI5rQs?t=4060) And if delta converges to 1, this converges to 1/2. So it's exactly what we talked about down here. OK, so now we want to verify that this is a subgame-perfect Nash equilibrium.

[1:08:16](https://youtu.be/iL3XRRI5rQs?t=4096) So what do we need to do? We need to check that the proposer has no profitable one-shot deviation. And the responder has no profitable one-shot deviation. So we need to check. No profitable one-shot deviations.

[1:08:40](https://youtu.be/iL3XRRI5rQs?t=4120) So let's start with proposer. There's a few things they could do. They could, let's say, offer the responder more than delta over 1 plus delta. Or they could offer the responder less than delta over 1 plus delta. There's a lot of things they could do, but let me just organize them in this way. So suppose I'm the proposer, and I deviate, at some history, by offering the respondent less than delta over 1 plus delta.

[1:09:13](https://youtu.be/iL3XRRI5rQs?t=4153) Well, then what happens?

[1:09:18](https://youtu.be/iL3XRRI5rQs?t=4158) Well, according to the respondent's strategy, this offer is going to be rejected.

[1:09:31](https://youtu.be/iL3XRRI5rQs?t=4171) So if this offer is rejected, what payoff am I going to get as the proposer? Well, today, I'm the proposer. I've proposed an offer that's been rejected. That means, tomorrow, the person who rejected my offer is going to propose something tomorrow. And then I have to decide what to do. But we're only looking at one-shot deviations. So if I'm the proposer, I'm only deviating here. So once we get to tomorrow, I'm going to follow the strategy I'm supposed to follow. So if I follow the strategy I'm supposed to follow tomorrow, because that's what a one-shot deviation means, then, tomorrow, well, I'm going to be offered delta over 1 plus delta.

[1:10:10](https://youtu.be/iL3XRRI5rQs?t=4210) And I'm going to accept it because this is-- if I didn't accept it, that would be a multi-shot deviation, which we don't have to worry about. So it's going to be rejected. So tomorrow, I'm going to be offered delta over 1 plus delta. And I'm going to accept. Notice the trickiness. Whether I want to offer less than delta over 1 plus delta to the other player today depends on how I'm going to play in the rest of the game.

[1:10:42](https://youtu.be/iL3XRRI5rQs?t=4242) The one-shot deviation principle allows us to take as fixed how I'm going to play in the rest of the game and focus on a very specific kind of deviation. And it's easier to check that this very specific deviation's not profitable. There might be really complicated deviations where I offer too little today. Then I also do something weird tomorrow. Then I do something weird the next day. But we don't have to worry about these by the one-shot deviation principle. So what happens if tomorrow I'm offered delta over 1 plus delta, and I accept? What is my discounted payoff going to be?

[1:11:14](https://youtu.be/iL3XRRI5rQs?t=4274) Discounted into today's units, what will my payoff be? Well, I'm getting delta over 1 plus delta tomorrow, but I have to discount that back to today. So today it's going to be delta times that. So my payoff is going to be delta squared over 1 plus delta.

[1:11:33](https://youtu.be/iL3XRRI5rQs?t=4293) But if I don't deviate, I get 1 over 1 plus delta. So that's certainly better. So this deviation is not profitable. It's strictly unprofitable.

[1:11:47](https://youtu.be/iL3XRRI5rQs?t=4307) What if I offer more than delta over 1 plus delta to the other player? Can this be profitable? What's going to happen?

[1:11:58](https://youtu.be/iL3XRRI5rQs?t=4318) Yeah.

[1:11:58](https://youtu.be/iL3XRRI5rQs?t=4318) AUDIENCE: \[INAUDIBLE] to accept.

[1:12:00](https://youtu.be/iL3XRRI5rQs?t=4320) IAN BALL: Right, so it'll be accepted. And then what? But is that good for me?

[1:12:03](https://youtu.be/iL3XRRI5rQs?t=4323) AUDIENCE: No, because now you're only getting like-- now you're losing R minus delta over 1 plus delta.

[1:12:11](https://youtu.be/iL3XRRI5rQs?t=4331) IAN BALL: Exactly, so if I give the responder more than delta over 1 plus delta, I can't be getting more than 1 over 1 plus delta, so this must be worse for me and unprofitable. So you can write out all the math really carefully, but I think it's better to just think through the story. So what are the two kinds of deviations we're considering? We're considering one-shot deviations where the proposer proposed a different offer today but then follows their equilibrium strategy, or their candidate equilibrium strategy, forever after.

[1:12:45](https://youtu.be/iL3XRRI5rQs?t=4365) One kind of deviation is a deviation where they are very generous, and they offer to give more to the other player. The other player is going to accept this offer, but I'm hurting myself as the proposer by being too generous, more generous than I need to be, so this deviation won't be profitable. Another thing I could do is I could offer them less than what I'm supposed to in equilibrium. I could be too stingy. If I'm too stingy, the offer will be rejected. That means, tomorrow, the other player will make a counteroffer to me. The counteroffer to me is going to be delta over 1 plus delta.

[1:13:19](https://youtu.be/iL3XRRI5rQs?t=4399) And I'm going to accept it precisely because this is a one-shot deviation. And not accepting it would be a multi-shot deviation. But if I accept it, I actually get a lower payoff, and it's farther in the future, which means it's definitely worse for me. So we've checked that the proposer does not have any profitable one-shot deviations. We have not directly checked multi-shot deviations where the proposer is too stingy today and then also rejects the offer tomorrow and then does more complicated things. But the one-shot deviation principle says we don't have to worry about that. Any questions on that?

[1:13:51](https://youtu.be/iL3XRRI5rQs?t=4431) OK, now let's look at the other side. Let's look at the responder.

[1:14:16](https://youtu.be/iL3XRRI5rQs?t=4456) The responder is offered some amount.

[1:14:22](https://youtu.be/iL3XRRI5rQs?t=4462) And there's really two cases. So let's say the responder is offered at least-- so we're going to split the histories in two.

[1:14:40](https://youtu.be/iL3XRRI5rQs?t=4480) So there's some class of histories where a lot of things happen. We don't worry about it, but then, at the end of it, I'm the responder, and I'm offered some amount that's weakly greater than delta over 1 plus delta. And what I'm supposed to do is accept this.

[1:15:00](https://youtu.be/iL3XRRI5rQs?t=4500) And then there's other histories where I'm offered strictly less than delta over 1 plus delta. And again, I can choose whether to accept or reject. But I'm supposed to reject these. And we need to check that I don't have a profitable deviation, a profitable one-shot deviation as the responder.

[1:15:23](https://youtu.be/iL3XRRI5rQs?t=4523) So when I'm offered-- it's easy to compute what I get when I accept these offers. So let's write that down. If I'm offered more than this and I accept, then I'm going to get whatever this amount is that's greater than or equal to delta over 1 plus delta. And over here, I'm going to get this amount that's less than delta over 1 plus delta. Now, the harder case is figuring out what happens if I reject because if I reject, my payoff depends on what happens tomorrow. But again, we're only looking at one-shot deviations. So if I deviate here by rejecting, we can consider what happens tomorrow when I follow the strategy I'm supposed to.

[1:15:58](https://youtu.be/iL3XRRI5rQs?t=4558) So if I reject as the responder, tomorrow, I become the proposer. And now I follow my strategy. I propose to keep 1 over 1 plus delta for myself and give delta over 1 plus delta to the other player, and the other player has to accept because that's what their strategy says. So then, tomorrow, I'm going to get 1 over 1 plus delta as the proposer. But that's going to be-- the discounted value of that equals delta over 1 plus delta.

[1:16:38](https://youtu.be/iL3XRRI5rQs?t=4598) So 1 over 1 plus delta tomorrow is only worth delta times that to me today. But if I accept, I'm doing weakly better than that. So this deviation's not profitable. So we're good.

[1:16:58](https://youtu.be/iL3XRRI5rQs?t=4618) Let's look what happens if I reject the offer. Well, we know if I reject the offer, I'm going to become the proposer. Exactly the same argument goes through. In fact, this is not even a deviation. This is just saying, what happens if I follow my equilibrium strategy and reject? I'm exactly going to get a discounted value of delta over 1 plus delta. And if I deviate and accept this offer, I get strictly less than delta over 1 plus delta. So again, it's not profitable.

[1:17:29](https://youtu.be/iL3XRRI5rQs?t=4649) So we're good. Let's just go through it. I think the algebra can sometimes make it seem harder than it is. So let's try to understand. Suppose I'm offered an amount less than delta over 1 plus delta. My equilibrium strategy says, I should reject that offer, propose the equilibrium split tomorrow, and therefore get 1 over 1 plus delta tomorrow, which has discounted value delta over 1 plus delta. If I deviate, well, here we don't even have to worry about whether it's a one-shot deviation or not because if I deviate here, the game ends immediately, and I get strictly less than delta over 1 plus delta.

[1:18:00](https://youtu.be/iL3XRRI5rQs?t=4680) So my deviation is strictly unprofitable. And now we've confirmed that this constitutes a subgame-perfect Nash equilibrium. And in fact, you can show that this is the unique subgame-perfect Nash equilibrium. But that's a bit more involved. And I might show it next class. And I think I'm perfectly out of time. So let me stop there. Thanks.
