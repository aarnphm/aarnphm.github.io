---
title: 'lecture 8: backward induction'
description: 'In this lecture, Ian Ball talks about backward induction, which is the process of determining a sequence of optimal choices by reasoning from the endpoint of a problem or situation back to its beginning using individual events or actions.'
id: 'mit-14-12-fall-2025-resource-resources-mit14-12f25-lec08-1080p-mp4'
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
    n: 8,
    youtube: 'VHex0a2JFWI',
    duration: 4623,
    cues: 1587,
    captions: 'static_resources/191_SJ-IBUD14IRB1WLfMiDRj7zV-OYv9_transcript.webvtt',
    transcript: 'static_resources/191_SJ-IBUD14IRB1WLfMiDRj7zV-OYv9_transcript.pdf',
  }
layout: 'resource'
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
---

[0:00:10](https://youtu.be/VHex0a2JFWI?t=10) IAN BALL: All right, so we're going to start today with another MobLab game. Hopefully everyone can get logged in. This is going to be-- the game we're going to play is called the Ultimatum Game.

[0:00:24](https://youtu.be/VHex0a2JFWI?t=24) And this is probably the most famous game that's used in laboratory experiments about economics. So the way it's going to work is that you're going to be randomly assigned to two roles. One of you is going to be the proposer, and one of you is going to be the responder. And there's some pot of money. And the proposer just proposes how that money is split between them and the other person. And then the other person can choose to accept that, in which case you split the money in that way, or reject that, in which case no one gets any money.

[0:00:54](https://youtu.be/VHex0a2JFWI?t=54) So player 1 makes an ultimatum. This is how much money you get. This is how much money I get. Take it or leave it. Player 2 says, I'll take it, or I'll leave it and then we both get nothing. So that's the game. I think in this version you have \$100, maybe, to split. So let's play and see what happens.

[0:01:18](https://youtu.be/VHex0a2JFWI?t=78) So here we have a table of the results. What we can see is the blue circle, these are offers that were accepted. So this is the amount that the proposer offered the responder could receive out of the 100. And then the orange circles are offers that were rejected. And then we see the averages here. So we definitely see that, on average, at least especially in these early rounds, the higher offers were more likely to be accepted, and the lower offers were more likely to be rejected.

[0:01:54](https://youtu.be/VHex0a2JFWI?t=114) It looks like a lot of the offers were above zero, except maybe one down here. Someone offered zero. Anyone who played want to share their thoughts about how they approached this and how they thought about this game?

[0:02:10](https://youtu.be/VHex0a2JFWI?t=130) So what about when you were the-- yeah, go ahead.

[0:02:12](https://youtu.be/VHex0a2JFWI?t=132) AUDIENCE: I was the one who was receiving the offer. So as long as the amount was positive, I would receive it because otherwise I wouldn't \[INAUDIBLE].

[0:02:20](https://youtu.be/VHex0a2JFWI?t=140) IAN BALL: OK, so that's the ultra-rational way to play. And that's what game theory would propose. Often in practice we don't see that happening. But that's a great way of playing. So other people who rejected positive offers, what was your thinking about that? Yeah?

[0:02:35](https://youtu.be/VHex0a2JFWI?t=155) AUDIENCE: I guess I was less rational. If no one offered me 50, then I was like, well, I don't like you. So I'm not going to accept that.

[0:02:43](https://youtu.be/VHex0a2JFWI?t=163) IAN BALL: OK, great.

[0:02:45](https://youtu.be/VHex0a2JFWI?t=165) AUDIENCE: \[INAUDIBLE]

[0:02:46](https://youtu.be/VHex0a2JFWI?t=166) IAN BALL: So a lot of people have this reaction about fairness. So this captures people's idea of fairness. Some people are willing to sacrifice financial gains in order to punish people who you might feel have morally wronged you or done something that's fundamentally unfair. Again, it's a pretty cheap way to punish someone when there's no money on the line. If it was \$50,000 and they gave you 49,000, you'd probably still accept it, right? But yeah, that's useful. And then what about people who were proposing? What was your reasoning?

[0:03:20](https://youtu.be/VHex0a2JFWI?t=200) Did you propose an even split? Did you propose more for yourself? What did you do?

[0:03:27](https://youtu.be/VHex0a2JFWI?t=207) It's very quiet. Someone proposed. Some people proposed-- it looks like-- I guess, if we look at the game, it looks like no one gave more than 50 to the other person. So it looks like some people felt, let's be generous and give 50. And then other people wanted to keep a little more for themselves. Usually what happens in this game is people-- their reasoning is, well, I anticipate the responder might be a bit like you suggested and might not be happy with me if I give them nothing, but I don't have to give them all-- the full half. Maybe I'll give them 30 or 40.

[0:03:58](https://youtu.be/VHex0a2JFWI?t=238) That's a pretty common response. And then the hope is that the person will accept this. Here, you actually played it in multiple rounds. Did anyone think about the dynamics of this game? So I think that's an important consideration in this game. Today, we're going to think about a static version of this game. But if you're playing this repeatedly, you might think, well, depending on what I do in this round, that might affect the way the person will play in future rounds. If they know that I'm someone who rejects really low offers, maybe they'll be forced or motivated to offer more to me in the future.

[0:04:29](https://youtu.be/VHex0a2JFWI?t=269) And that's something we'll think about when we get into repeated games. So let's close this and move into more of a formal analysis of this game. So the topic for today is going to be what's called backward induction. But I think the best way to go about it is just to think through this example, and then, later in the class, I'll give a more formal definition. So let's look at an easier version of the ultimatum game.

[0:04:59](https://youtu.be/VHex0a2JFWI?t=299) I think in the game you played, you could choose any number between maybe 0 and hundreds. Let's make it a bit simpler. So let's see the ultimatum game with maybe \$3 at stake. And let's think about how this game works. So we have player 1, who's what we would call the proposer, and player 2, who's the responder.

[0:05:27](https://youtu.be/VHex0a2JFWI?t=327) And the proposer is proposing how to split the \$3 between themselves and the other person. But we're going to keep track of it as thinking about the proposer, P1, makes an offer.

[0:05:42](https://youtu.be/VHex0a2JFWI?t=342) So we'll think of the offer as what they offer to the other person. And that means that they will keep the remaining amount. So there's two numbers going on. But because the numbers always sum to \$3, we only have to keep track of one of those numbers. And we'll keep track of the number, which is the amount the responder would get. And then the responder, player 2, can either accept or reject this offer. And as usual, we're going to assume that people are risk neutral so that monetary payoffs coincide with utilities.

[0:06:20](https://youtu.be/VHex0a2JFWI?t=380) So let's try to represent this game in the extensive form. So first, we have player 1. And how much can they offer to player 2? Well, we'll assume that they can only make offers that are in even dollar amounts just to make it a bit simpler. So they can either offer 0, 1, 2, or 3.

[0:06:44](https://youtu.be/VHex0a2JFWI?t=404) And each of these offers, it's then player 2's turn to move.

[0:06:51](https://youtu.be/VHex0a2JFWI?t=411) Of course, these are all in separate information sets. We have to remember that player 2 observes the offer that is made by player 1. The responder observes the offer made by the proposer. So they can distinguish these offers. And then they choose accept or reject, which I'll call A/R. And now let's make sure we get the payoffs right. So whenever the second player rejects the offer, no matter what the offer was, both players get 0.

[0:07:25](https://youtu.be/VHex0a2JFWI?t=445) That's the rule of the game. If the offer is rejected, you both get 0. So we're going to have 0, 0; 0, 0; 0, 0; and 0, 0. Now, if the offer is accepted, then the number here is the offer that goes to player 2, which means player 1 gets the remainder of the \$3. So up here, it's going to be 3, 0. The accepted offer is an offer of 0 to player 2, and that means player 1 keeps 3. Here, it's going to be 2, 1.

[0:07:56](https://youtu.be/VHex0a2JFWI?t=476) The offer is 1 to player 2 is accepted. And that means the first player, the proposer, keeps 2. Then we have 1, 2, and 0, 3.

[0:08:10](https://youtu.be/VHex0a2JFWI?t=490) So here's the game. And the first approach to analyzing this game is let's think about how we can represent this game in strategic form. So what is a strategy? Let's write down what the strategies are for the players.

[0:08:34](https://youtu.be/VHex0a2JFWI?t=514) What about for player 1 and for player 2? So for player 1, what is a strategy in this game? Well, it's just an offer. So for player 1, a strategy is an element of the set 0, 1, 2, 3. They just choose, how much do they offer to player 2, to the responder? What about a strategy for player 2? This is a little trickier. How do we represent a strategy for player 2? Or what kind of mathematical object is it? Yeah?

[0:09:05](https://youtu.be/VHex0a2JFWI?t=545) AUDIENCE: \[INAUDIBLE] people?

[0:09:06](https://youtu.be/VHex0a2JFWI?t=546) IAN BALL: Right. So it's a function of four things, which we can represent as a four-tuple. Let's first write it as a function, and then I'll represent it that way. So it's a function that says, as a function of what you offer me, I have to choose whether to accept that offer or reject that offer. And here we're looking at pure strategies. So it's a function from 0, 1, 2, 3 to either accept or reject. And how are we going to represent that function? As you suggested, the way we normally write down strategies is we put one blank space for each of the information sets of the player.

[0:09:41](https://youtu.be/VHex0a2JFWI?t=581) In this case, the player has four information sets corresponding to the four possible offers. So we have 0, 1, 2, 3. And as an example, they might use a strategy like this. And this would be a very common strategy to use in this game. This would say, if you only offer me 0 or 1, I'm going to be upset with you. I'm going to find that unfair. So I'm going to reject that offer and punish both of us. If you give me 2 or 3, I'm very happy, and I'm going to accept that offer. I think it's fair.

[0:10:11](https://youtu.be/VHex0a2JFWI?t=611) In fact, you've given me a majority of the pot, so I'm very happy, and I'll accept it. So now let's see if we can find a Nash equilibrium of this game involving this strategy. So remember, Nash equilibrium is a strategy profile. I've specified one strategy. Let's see-- can we find a strategy for player 1 that makes this into a Nash equilibrium? So let's think this through. If you were player 1, if you were the proposer, and you expected or you believed that the responder was going to play in this way, how much would you offer the responder?

[0:10:45](https://youtu.be/VHex0a2JFWI?t=645) Yeah, over here?

[0:10:46](https://youtu.be/VHex0a2JFWI?t=646) AUDIENCE: 2.

[0:10:46](https://youtu.be/VHex0a2JFWI?t=646) IAN BALL: 2, great. So we're going to have exactly S2 equals-- S1 equals 2. That's confusing. And explain your reasoning. Why would you offer 2?

[0:10:55](https://youtu.be/VHex0a2JFWI?t=655) AUDIENCE: If you offer lower, both of you get 0. If you offer more, then you get nothing. So \[INAUDIBLE].

[0:11:02](https://youtu.be/VHex0a2JFWI?t=662) IAN BALL: Great. So another way of saying that-- exactly right-- is, I'm making the lowest offer that will be accepted. I don't want to offer any less because there I'm clearly doing-- if I offer them less, it gets rejected. And if I offer them more, it would be-- it will be accepted, but it will be worse for me. So just to keep track, let's say these offers-- these offers are better for the proposer if they're accepted because I'm offering less money to the respondent, but they're not accepted.

[0:11:42](https://youtu.be/VHex0a2JFWI?t=702) So when I say better for the proposer, I mean better for the proposer if they're accepted. The problem with these is they're not accepted. And then these higher offers are worse for the proposer.

[0:11:56](https://youtu.be/VHex0a2JFWI?t=716) So I make the lowest offer. That is the offer that's best for me if accepted. And I find the lowest one that's actually going to be accepted. So this indeed is going to be a Nash equilibrium.

[0:12:10](https://youtu.be/VHex0a2JFWI?t=730) And let's check this formally. Let's check that neither player has a profitable deviation. If player 1 is offering 2, let's check that player 2 cannot profitably deviate. Well, if you're offering me 2, there's only two possibilities. I accept the offer, and I get 2, or I reject the offer, and I get 0. So certainly I'm behaving optimally by accepting the offer. Now let's go the other direction. This is what we just reasoned through.

[0:12:41](https://youtu.be/VHex0a2JFWI?t=761) Given that this is the way that player 2, the responder, is playing, the best that player 1 can possibly do is to offer 2, and that means keep 1, because the only way they could get more than 1 is if they made one of these offers and it was accepted. But that's impossible given the strategy that the responder is using. So we found one Nash equilibrium. There's a lot of Nash equilibrium in this game. But there's something about this Nash equilibrium that seems wrong. Or maybe it seems problematic. So maybe I'll say, what's wrong with this Nash equilibrium?

[0:13:21](https://youtu.be/VHex0a2JFWI?t=801) And this is exactly what we want to capture today. So does anything feel off to you? Does it feel like there's a problem here? Yeah, in the front.

[0:13:31](https://youtu.be/VHex0a2JFWI?t=811) AUDIENCE: It seems like one person is always beating the other.

[0:13:34](https://youtu.be/VHex0a2JFWI?t=814) IAN BALL: OK, so that's true, but some games are unfair. So I think that's true. Maybe you're getting at it. But I would say let's add a little bit more to that. Yeah, so it is true that one player is always doing better, but I can certainly write down games where one player always loses. It doesn't necessarily mean they're playing irrationally. Any other thoughts in the front? Yeah?

[0:13:51](https://youtu.be/VHex0a2JFWI?t=831) AUDIENCE: The responders should \[INAUDIBLE] someone that's positive utility.

[0:13:54](https://youtu.be/VHex0a2JFWI?t=834) IAN BALL: Great. So this information set seems a bit puzzling. So this says, if I'm offered 1, there are two choices for me. I either accept 1, and I get my \$1, or I reject the offer, and I get 0. So it seems that in this contingency, accepting is strictly better than rejecting. And if you're not motivated by \$1, let's call it \$1 million. Then it makes it clear, I might be unhappy that you're only giving me 1 million out of the 3 million.

[0:14:24](https://youtu.be/VHex0a2JFWI?t=864) But if my choice is take the 1 million or reject it and get 0, it seems like accepting is much better. But why is this still consistent with Nash equilibrium? So I agree with you. It seems like clearly this is a-- we might call this a "mistake."

[0:14:41](https://youtu.be/VHex0a2JFWI?t=881) Or we might call this "suboptimal." But I thought Nash equilibrium said that players are playing optimal given the way their opponents are playing. So why is this allowed? It is a Nash equilibrium, but someone's playing suboptimally. So what's happening?

[0:15:02](https://youtu.be/VHex0a2JFWI?t=902) Yeah?

[0:15:02](https://youtu.be/VHex0a2JFWI?t=902) AUDIENCE: Because we're basing it off of the belief of how they're playing.

[0:15:05](https://youtu.be/VHex0a2JFWI?t=905) IAN BALL: Exactly. So player 2 believes that player 1 is making an offer of 2. If I believe that player 1 is making an offer of 2, I'm never going to reach this contingency. And because I'm never going to reach this contingency, how I play this contingency doesn't affect my payoff given the strategy of the other player. So what we've identified is kind of a crucial issue with Nash equilibrium, maybe a limitation of Nash equilibrium, that Nash equilibrium does not require optimal play at unreached contingencies.

[0:15:50](https://youtu.be/VHex0a2JFWI?t=950) We could be a little more precise-- contingencies that are not reached based on the play of the other player. But I'll be a little vague here-- "at unreached contingencies."

[0:16:08](https://youtu.be/VHex0a2JFWI?t=968) And if we want to dig a little deeper, when we analyze the Nash equilibrium of this game, we converted this game into a strategic form. And we said, well, what player 2 does is player 2 chooses a complete contingent plan. They just say, if I'm offered 0, I'll reject. If I'm offered 1, I'll reject. If I'm offered 2, I'll accept. And if I'm offered 3, I'll accept. And then they form beliefs about what player 1 is going to do. So this reasoning of the complete contingent plan, in some sense it's missing something.

[0:16:48](https://youtu.be/VHex0a2JFWI?t=1008) If I think about the complete contingent plan approach, it's missing the fact that a deviation might be observable. So it doesn't capture the observability of deviations.

[0:17:12](https://youtu.be/VHex0a2JFWI?t=1032) What do I mean by that? Well, the logic of Nash equilibrium says, I have a belief that my opponent is playing this strategy. And given that belief, I'm going to choose my contingent plan optimally. And that's it. I have my contingent plan, and then it gets executed. But if player 1 is going to deviate, well, I'm actually going to get to observe that because if they offer me 1, I get to see that. So what this strategic form reduction of the dynamic game seems to be missing is the fact that if player 1 were to deviate and play 1, well, now I know that they're not playing 2 anymore.

[0:17:48](https://youtu.be/VHex0a2JFWI?t=1068) But the complete contingent plan reasoning tells me, well, my play, even at this contingency, is governed by my belief that player 1 is always offering 2. And it's ignoring the fact that if we actually reach this contingency, I learn that my opponent has deviated and is not playing according to my belief. So the fact that information arrives over time is totally missed by the strategic form because the whole tree and the extensive form game are missing from the strategic form representation. And that's going to be what we're going to try to study more formally and more rigorously today.

[0:18:22](https://youtu.be/VHex0a2JFWI?t=1102) So today, we're going to define something called backward induction.

[0:18:35](https://youtu.be/VHex0a2JFWI?t=1115) And the basic idea is that backward induction is going to require optimal play even at unreached contingencies--

[0:18:54](https://youtu.be/VHex0a2JFWI?t=1134) obviously, at reached contingencies, as well. But what's new about it is that it requires optimal play at unreached contingencies.

[0:19:06](https://youtu.be/VHex0a2JFWI?t=1146) And I want to make one more point about this, that what we're doing today is going to depend crucially on the extensive form.

[0:19:19](https://youtu.be/VHex0a2JFWI?t=1159) So, so far our approach in the class has been to start with the extensive form to try to capture the application we're studying, reduce it to the strategic form, and then do all our analysis within the strategic form. And so far what we looked at, Nash equilibrium, these various solution concepts, rationalizability-- only dependent on the strategic form. But today, we're going to take the extensive form more seriously. And we're going to argue that something might be missing when we reduce the extensive form to the strategic form. So let's see how this algorithm works by looking at another simple example.

[0:19:56](https://youtu.be/VHex0a2JFWI?t=1196) Let's look at the Boston game again. But in this case, the first player chooses what sporting event to go, and that move is observed. So the Boston game with observed moves.

[0:20:11](https://youtu.be/VHex0a2JFWI?t=1211) So what happens, we have-- maybe I'll draw it-- we can draw it either way. I'll draw it up here. We have player 1 first chooses to go to the Celtics game or the Red Sox game. And then player 2 observes whether player 1 has chosen the Celtics or the Red Sox. And then they choose Celtics or Red Sox.

[0:20:37](https://youtu.be/VHex0a2JFWI?t=1237) So the original version of this game that we studied was a simultaneous-move version of this game. And in the simultaneous-move version, we would represent that by putting these two nodes in the same information set. Now we're studying a different version, where player 1's move is observed by player 2. And now let's remember our payoffs. I think the convention we've used is that player 1 prefers Celtics to Red Sox. So if they both go to the Celtics game, it's 2 1. If they both go to Red Sox, it's 1 2, and then 0 0, and 0 0. So when I write extensive form games this way, I'll put the first player's payoff on top instead of to the left, but it has the same meaning.

[0:21:14](https://youtu.be/VHex0a2JFWI?t=1274) Instead of going 0, comma 0, or 0, comma 1, I'll just do 2 and then 1. So first let's represent this game in strategic form.

[0:21:31](https://youtu.be/VHex0a2JFWI?t=1291) So we first have to understand the strategies for the two players. So player 1's strategy set is pretty simple. They're just choosing Celtics or Red Sox. So we just have C and R. But player 2's strategy set is a bit more complicated. Just like we had before, player 2 has two information sets. So their strategy should be a vector with two components, one component saying what they do here and one component saying what they do here. So we can write player 2's strategies as CC, CR, RC, and RR, where the meaning is that we've agreed that we're going to order our information sets from left to right.

[0:22:13](https://youtu.be/VHex0a2JFWI?t=1333) So the first letter says what player 2 does at this information set. And the second letter says what player 2 does at this information set. So now we've defined the strategies. We now need to write out the extensive form-- sorry, the strategic form. So we have CR. And we have CC, CR, RC, RR.

[0:22:42](https://youtu.be/VHex0a2JFWI?t=1362) So now we want to fill in the payoffs here. So if player 1 goes to the Celtics game, when we're trying to compute payoffs, what matters is the first entry of player 2's strategy because the first entry of player 2's strategy precisely says what player 2 does after they observe that player 1 goes to the Celtics game. So under these two strategies, they go to the Celtics game, as well. And therefore the payoff is 2, 1; 2, 1.

[0:23:17](https://youtu.be/VHex0a2JFWI?t=1397) Under either of these strategies, player 1 goes to the Celtics game. But player 2 responds by going to the Red Sox game. They're not at the game together. And they both get a payoff of 0.

[0:23:32](https://youtu.be/VHex0a2JFWI?t=1412) Now let's look at what happens if player 1 goes to the Red Sox game. Now it's the second component of the strategy that determines the payoffs. So when do we both go to the Red Sox game? It's actually going to be under these two strategies because these are the two strategies where player 2 goes to the Red Sox game after player 1 does. So we're going to have 1, 2, and 1, 2. And then over here, we're going to have 0, 0 and 0, 0.

[0:24:03](https://youtu.be/VHex0a2JFWI?t=1443) So let's now try to compute the Nash equilibria of this game. We're going to use our standard trick-- fixing a strategy of player 2. Oh, yes?

[0:24:14](https://youtu.be/VHex0a2JFWI?t=1454) AUDIENCE: Could you clarify what the first letter means \[? in ?] players too?

[0:24:17](https://youtu.be/VHex0a2JFWI?t=1457) IAN BALL: Yeah, absolutely. So this means-- so let me call these information sets maybe Ia and Ib. This is the first-- I'll use a so we don't get confused with the first player-- the first information set of player 2. And this is the second information set of player 2. And this is what they do at Ia. And this is what they do at Ib. So in words-- let me use this strategy.

[0:24:49](https://youtu.be/VHex0a2JFWI?t=1489) This is a better example since it's not constant. At the first information set, I play C. And at the second information set, I play R. That's my complete contingent plan. So now let's try to compute the Nash equilibria of this game. So as usual, we're going to fix the strategy of player 2 and think about how well player 1 does with their different strategies. So given that player 2 does this, player 1 is better off choosing C because 2 is greater than 0. Given here, player 2 is better off here.

[0:25:21](https://youtu.be/VHex0a2JFWI?t=1521) Here, it doesn't matter. And here, we go and get this. So I'm just going column by column, fixing player 2's strategy and computing player 1's best response. Now I'm going to do it the other way. I'm going to fix player 1's strategy and compute player 2's best response. So I'm going to get this and this.

[0:25:47](https://youtu.be/VHex0a2JFWI?t=1547) So here, I get three Nash equilibria. Remember, Nash equilibrium can be visualized here by looking for a cell that has both numbers underlined. So we have three Nash equilibria. Player 1 goes to the Celtics game. And player 2 always goes to Celtics game. Player 2 goes to the Red Sox game, and player 2 always goes to the Red Sox game. And then this one, where player 2 goes to the Celtics game, and what player 2 does depends on what they observe. So they go to the Celtics game if player 1 goes to the Celtics game, but they go to the Red Sox game if player 2 goes to the Red Sox game.

[0:26:22](https://youtu.be/VHex0a2JFWI?t=1582) So we have three Nash equilibria. Which of these-- I'll write this-- which of these seems most compelling to you? Which of these seems the most reasonable? Or do they all seem reasonable? Or do none of them seem reasonable? I guess I gave it away. Yeah?

[0:26:41](https://youtu.be/VHex0a2JFWI?t=1601) AUDIENCE: \[INAUDIBLE] C CR.

[0:26:44](https://youtu.be/VHex0a2JFWI?t=1604) IAN BALL: OK, this one. And tell me why.

[0:26:47](https://youtu.be/VHex0a2JFWI?t=1607) AUDIENCE: I mean, you're getting some sort of payoff either way or some positive payoff either way, whereas the other ones \[INAUDIBLE] always \[INAUDIBLE] the first player doesn't go to the Celtics game \[INAUDIBLE].

[0:27:00](https://youtu.be/VHex0a2JFWI?t=1620) IAN BALL: Right. So I think what you're thinking about is exactly, even at the contingency that's not reached, this second player is doing pretty well. So the issue, if we look at the first one or the last one, it is a Nash equilibrium, but it involves player 2 playing suboptimally at some contingency, namely in this Nash equilibrium, player 2 goes to the Celtics game even when they see that player 1 has chosen the Red Sox game. That's suboptimal play because they'd rather join them at the Red Sox game than be all alone at the Celtics game.

[0:27:35](https://youtu.be/VHex0a2JFWI?t=1655) And similarly, this strategy profile, while it is a Nash equilibrium, involves suboptimal play at the first contingency because player 2 observes that player 1 has gone to the Celtics game, and they've responded not by going to the Celtics game, but by going to the Red Sox game. So each of these involves suboptimal play at unreached contingencies. And it's only this Nash equilibrium that seems to have this special property of optimal play at every contingency. So let's see how we could have tried to identify this.

[0:28:08](https://youtu.be/VHex0a2JFWI?t=1688) What algorithm could we have used? If I have colored chalk-- maybe not. So I'll just highlight it. Well, it's pretty intuitive to work backwards. And I think this is how most people approach problems like this. Let's analyze this working backwards. Let's start here at these last information sets and say, how should player 2 behave optimally if this information set is reached? So we'll look at player 2. We'll say, well, if we're here, we're certainly better off getting 1 than 0. So I'm going to choose this edge.

[0:28:41](https://youtu.be/VHex0a2JFWI?t=1721) And then we'll come over here and say, if I reach this information set-- I don't know if we'll reach it. I don't know how player 1 is playing. But let's suppose that we get to this contingency. Well, if I choose C, I get 0. If I choose R, I get 2. So R is better. And then we can work our way up a step further. And now we can say, well, suppose I'm player 1 and I'm choosing how to play. I'm anticipating how player 2 is going to play at each of these contingencies. So if I choose C, I anticipate we'll go down here, and I'll get a payoff of 2.

[0:29:15](https://youtu.be/VHex0a2JFWI?t=1755) And if I choose R, I anticipate player 2 will do this, and I'll get a payoff of 1. 2 is higher than 1, so I'll go here. And sure enough, we've exactly identified this special Nash equilibrium. When you apply this procedure, it's very tempting to then-- you follow the path down and you say, ah, 2 1 is the answer. But no, 2 1 is just the payoff. The answer that you're looking for is a strategy profile. The strategy profile is player 1 plays C, and player 2 makes a contingent plan to play C at this information set and R at this information set.

[0:30:02](https://youtu.be/VHex0a2JFWI?t=1802) So now let's formalize this procedure. Yes?

[0:30:05](https://youtu.be/VHex0a2JFWI?t=1805) AUDIENCE: Since this is an \[INAUDIBLE] version of the \[INAUDIBLE] game, then would players here have no reason to choose the 0 0 payoffs in order to convince player 1 to switch strategies? Or is that only in the game where they don't know each other's choices?

[0:30:24](https://youtu.be/VHex0a2JFWI?t=1824) IAN BALL: Wait, so maybe what you're getting at is, let's look at player 1's payoffs in these equilibria. So we have this equilibrium, 2, 1, that gives-- it's an equilibrium that gives the payoffs 2 and 1. And we have another Nash equilibrium that gives the payoffs 1, 2. So you might say, if you're player 2, you do better in this equilibrium than this equilibrium. And indeed, this gets at a central issue, another way of thinking about backward induction, that how-- let's suppose I'm player 2 and I want to get to this equilibrium.

[0:31:00](https://youtu.be/VHex0a2JFWI?t=1860) What would I do? Maybe I would say something like, look, I'm always going to go to the Red Sox game. That's just who I am. Whatever I see you do, I'm going to go to the Red Sox game. And it's almost like a threat because the way I punish you is by just saying, I'm going to the Red Sox game whatever you do. The issue is that another way of saying it is that this involves a non-credible threat.

[0:31:25](https://youtu.be/VHex0a2JFWI?t=1885) So let's understand, why might player 2 want to make this threat? It would be great for player 2 if player 1 believes that player 2 is always going to go to the Red Sox game because player 2 loves the Red Sox. They love player one to believe that because then player 1 will choose the Red Sox themselves. The problem is if player 1 thinks it through, they're going to say, wait a second, you're threatening to go to the Red Sox game whatever I do. But if I actually go to the Celtics game, are you really going to find it optimal to follow through on that threat? And the answer is no, because given that I'm going to the Celtics game, you'd rather go to the Celtics than go alone.

[0:31:58](https://youtu.be/VHex0a2JFWI?t=1918) So one other way of thinking about backward induction is that it's a principled way of ruling out equilibria that involve non-credible threats at contingencies that are not reached. Great question. Yeah, any other questions on this? Yes?

[0:32:11](https://youtu.be/VHex0a2JFWI?t=1931) AUDIENCE: Then would you say offering 1 to a person then rejecting it is a non-credible threat?

[0:32:17](https://youtu.be/VHex0a2JFWI?t=1937) IAN BALL: Yes. So similarly, this was a Nash equilibrium that involves a non-credible threat, and therefore it's a Nash equilibrium, but it will not satisfy backward induction. So in both of these cases-- these are very symmetric. In both cases, we have many Nash equilibria. But some of the Nash equilibria involve non-credible threats. And we're going to introduce an algorithm called backward induction that will rule out those equilibria. But I think the point that could be confusing is even though I'm saying this is not credible, it's still an equilibrium.

[0:32:48](https://youtu.be/VHex0a2JFWI?t=1968) And similarly over here, this was exactly a non-credible threat, but it was still part of an equilibrium. Yeah. Any other questions? This is great. OK, so now let's be a bit more formal or just say what our algorithm is. So this is-- we'll call this the backward induction algorithm.

[0:33:19](https://youtu.be/VHex0a2JFWI?t=1999) So we already did it here. The algorithm is you work backwards. But let's say what that means. Well, first, what's the input to the algorithm? And formally, the input to the algorithm is a finite extensive-form game.

[0:33:42](https://youtu.be/VHex0a2JFWI?t=2022) Really, the important aspect of finiteness is that the players alternate moves only finitely many times. If one player at some node can choose among infinitely many moves, like choosing a number between 0 and 100, it's still going to be OK. But to be extra careful, we'll only-- we'll formally define the algorithm for finite games. But we will be a bit informal and then apply it to some infinite games where it won't create problems. The other important aspect is it's an extensive-form game. The whole point of this is that this algorithm is faithfully capturing the dynamics of the game, and we wouldn't be able to observe that if we started with a strategic-form game.

[0:34:20](https://youtu.be/VHex0a2JFWI?t=2060) So notice this is fundamentally different from Nash equilibrium. When we define Nash equilibrium, our starting point was a given strategic-form game. Here, our starting point is an extensive-form game. And the other final thing that we need is this is a game with perfect information.

[0:34:43](https://youtu.be/VHex0a2JFWI?t=2083) What do I mean by perfect information? I mean every information set is a singleton.

[0:35:01](https://youtu.be/VHex0a2JFWI?t=2101) And as long as information sets are singletons, you could even have moves by nature, that's OK, but only moves by nature that everyone gets to see. So poker, where you have a private hand, that's not going to be perfect information. But if nature flipped a coin and everyone saw that, that would be OK. Can you see what would go wrong? Let's say these two nodes were in the same information set and we tried to perform the same procedure. What would go wrong?

[0:35:33](https://youtu.be/VHex0a2JFWI?t=2133) Well, if we were at this information set and we tried to say how should player 2 play at this information set, we wouldn't know how to play because it would depend on which of the nodes in the information set we were at. And later in the course, we'll think about how to approach that. But for now, our algorithm-- "later in the course," just in two days-- but for now, we're only going to apply this to games of perfect information so that, at any node we get to, we know that we're at that node, and therefore we can compute the optimal play. If we had a non-singleton information set and we don't know which node we're at within that information set, it's not clear what optimal play is.

[0:36:08](https://youtu.be/VHex0a2JFWI?t=2168) So this is our input. And what is our output?

[0:36:14](https://youtu.be/VHex0a2JFWI?t=2174) We have to keep in mind that the output is always a strategy profile.

[0:36:22](https://youtu.be/VHex0a2JFWI?t=2182) It's not a pair of payoffs. It's the full strategy profile. And this is the most common mistake I see. People just list the payoffs they get at the end. They follow the paths down, and they say 2 1. But you really have to specify the full strategy profile. Now, I could formally define what the algorithm is, but I think you see how it works. I can introduce all this notation and say, you start at the end, and you play optimally, and you work your way backwards. But I think it's just easier to say, you see how it is. But let me talk about a few properties of this algorithm. So the output is not just any strategy profile.

[0:36:54](https://youtu.be/VHex0a2JFWI?t=2214) A first observation is that it is a pure strategy profile because, at each point, we just choose one action. We don't need any mixing. Now, if the player were indifferent at some point, you might have two choices. There might be two ways they could play optimally. And in that case, just choose one arbitrarily. So always break ties yourself because, at the end, we always want to get a pure strategy equilibrium when we apply this. And that kind of indicates another property of this, that, in fact, the output is going to be unique as long as there are no ties.

[0:37:39](https://youtu.be/VHex0a2JFWI?t=2259) So as long as we never get to a point where there's a tie, we never have any flexibility in what we do. So when we work backwards, we're always going to have a unique edge to choose. There's never going to be multiple choices.

[0:37:54](https://youtu.be/VHex0a2JFWI?t=2274) And this isn't just going to be any strategy profile. This is also going to be a Nash equilibrium.

[0:38:08](https://youtu.be/VHex0a2JFWI?t=2288) So what does this algorithm do? It gives us a pure-strategy Nash equilibrium. And unless there are ties, it always gives us the same pure-strategy Nash equilibrium. But let's keep in mind a warning.

[0:38:27](https://youtu.be/VHex0a2JFWI?t=2307) As we already saw, there may be other Nash equilibria.

[0:38:38](https://youtu.be/VHex0a2JFWI?t=2318) So it's true that the output of this algorithm is a Nash equilibrium, but it's a very special Nash equilibrium. And it's possible that there are other Nash equilibria that would not-- that cannot be outputs of this algorithm. And those would be the Nash equilibria that involve non-credible threats, as we've identified previously.

[0:39:08](https://youtu.be/VHex0a2JFWI?t=2348) Now, I think one nice implication of this that we shouldn't forget is that, recall--

[0:39:20](https://youtu.be/VHex0a2JFWI?t=2360) recall that some of the strategic-form games we looked at before did not have pure-strategy Nash equilibria.

[0:39:48](https://youtu.be/VHex0a2JFWI?t=2388) Can you remember an example of a game that doesn't have-- a finite game that doesn't have a pure-strategy Nash equilibrium?

[0:40:05](https://youtu.be/VHex0a2JFWI?t=2405) Yeah?

[0:40:05](https://youtu.be/VHex0a2JFWI?t=2405) AUDIENCE: Rock, paper, scissors.

[0:40:06](https://youtu.be/VHex0a2JFWI?t=2406) IAN BALL: Rock, paper, scissors is a great example. So it's unique. Nash equilibrium is we each mix one third, one third, one third. So wait a second. Rock, paper, scissors doesn't have a pure-strategy Nash equilibrium. But here, I have an algorithm that always gives us a pure-strategy Nash equilibrium. So what's going on here?

[0:40:27](https://youtu.be/VHex0a2JFWI?t=2427) Well, the input to our algorithm is a finite extensive-form game with perfect information. So notice, rock, paper, scissors is a simultaneous move game. If we wanted to represent rock, paper, scissors in extensive form, we'd have an information set with multiple nodes, and therefore this would not be a game of perfect information, and therefore algorithm would not apply. So one implication of this algorithm-- you can think of it as a theorem-- is that every extensive-form game with perfect information has a pure-strategy Nash equilibrium.

[0:40:59](https://youtu.be/VHex0a2JFWI?t=2459) So Nash's theorem told us that any game has a mixed-strategy Nash equilibrium. Some games only have mixed-strategy Nash equilibria. But for this special class of games with perfect information, we always have a pure-strategy Nash equilibrium.

[0:41:16](https://youtu.be/VHex0a2JFWI?t=2476) So maybe I'll say-- what's the answer here? Perfect-information games are special.

[0:41:28](https://youtu.be/VHex0a2JFWI?t=2488) And in fact, what's nice about this algorithm is it doesn't just show that a pure-strategy Nash equilibrium exists, it gives us an algorithm for finding it, whereas computing Nash equilibria in arbitrary games is generally much harder. And this actually formally confirms a claim I made in a previous class. If you recall, I said that the value of chess is plus 1, minus 1, or 0. Can anyone see-- I didn't actually check that claim.

[0:41:58](https://youtu.be/VHex0a2JFWI?t=2518) Can anyone see why this follows from what we've said so far? So we know chess has a value because chess is a zero-sum game. But in general, the value of a game could be something like 0.3 or negative 0.7. So we know chess is some value that's going to be between negative 1 and 1. But how do we know that it has a value that's either 0, negative 1, or 1, and not, say, 0.7? Yeah?

[0:42:26](https://youtu.be/VHex0a2JFWI?t=2546) AUDIENCE: Win, draw, \[? or loss. ?]

[0:42:28](https://youtu.be/VHex0a2JFWI?t=2548) IAN BALL: So those are the outcomes of the game, that's true. But I can give you examples of games-- let's take rock, paper-- well, let's take-- rock, paper, scissors is not a good example. The-- actually, hide and seek is also not a good example. There are games that can give you with 1's and 2's and only whole-number payoffs, where the value of the game is not a whole number because it involves mixing. So rock, paper, scissors, if I slightly change things, I'd be able to get that. So that shows that, indeed, the value can't be below negative 1 or above 1.

[0:43:01](https://youtu.be/VHex0a2JFWI?t=2581) But in principle, players could play mixed strategies. And maybe the value-- we often saw in zero-sum games that the security strategies involved mixing. And that meant that the value was a fraction. Yeah?

[0:43:13](https://youtu.be/VHex0a2JFWI?t=2593) AUDIENCE: I mean, I don't have the exact answer, but does it have something to do with the fact that one player has one step's worth of perfect information, \[? that ?] you always know what happens if you \[INAUDIBLE].

[0:43:25](https://youtu.be/VHex0a2JFWI?t=2605) IAN BALL: Yeah, I think it's even easier than that. They do have perfect information. So chess is a game of perfect information. Whenever I move, I see all the moves-- I see the board, and I know exactly how people have moved before. So let's understand this. Chess is a game of perfect information, PI, which means I can apply backward induction to chess, in principle. Of course, I can't actually do it because the game tree explodes. No computer-- all the NVIDIA chips in the world can't solve chess, but because it's a game of perfect information, I know it has a pure-strategy Nash equilibrium from backward induction.

[0:44:07](https://youtu.be/VHex0a2JFWI?t=2647) And because it has a pure-strategy Nash equilibrium-- well, because it's a zero-sum game, I know that in that pure-strategy Nash equilibrium, each player is actually playing a security strategy. So it must consist-- because it's zero-sum, that consists of security strategies. And the payoff in that pure-strategy Nash equilibrium consisting of security strategies is exactly the value of the game.

[0:44:41](https://youtu.be/VHex0a2JFWI?t=2681) So in general, the value of the game is always the payoff when each player plays their security strategy. But often, those security strategies involve mixing. Those are mixed strategies. And therefore the payoff isn't necessarily plus 1, minus 1, or 0. It could be something like 1/2. But this tells us that, in fact, the value of the game is plus 1, minus 1, and 0. And even more amazingly, it tells us that the security strategies are pure. So if I say, what's the optimal way to play chess, it's not, oh, I'm going to randomize, like it is in poker.

[0:45:11](https://youtu.be/VHex0a2JFWI?t=2711) There's literally a book. It would be bigger than the size of the universe, but there is a book I could, in principle, write down that says, given this move on the board, this is how you should play. Given this position on the board, this is how you should play. And that would ensure either that I always win, depending on if the value is 1 and I'm the first player, maybe I always draw, or maybe the other player has such a strategy. We just don't know in chess who has the advantage. In checkers, we do know. So what's the value of checkers? Does anyone know?

[0:45:42](https://youtu.be/VHex0a2JFWI?t=2742) Yeah?

[0:45:42](https://youtu.be/VHex0a2JFWI?t=2742) AUDIENCE: The first player has won.

[0:45:44](https://youtu.be/VHex0a2JFWI?t=2744) IAN BALL: No, I think it's actually 0. I think, checkers, if people play optimally, I think it ends in a tie. Though, there may be different variants of checkers, so I have to check that. But I think, in standard checkers, it always ends in a tie. Yeah. Yeah?

[0:45:55](https://youtu.be/VHex0a2JFWI?t=2755) AUDIENCE: When you say a game has a value, what are you referring to?

[0:45:59](https://youtu.be/VHex0a2JFWI?t=2759) IAN BALL: So this is something specific to zero-sum games. That's a lecture we did, I guess, on Thursday. Yeah, so it's referencing a previous lecture. Yeah, I'm happy to talk more after.

[0:46:10](https://youtu.be/VHex0a2JFWI?t=2770) Any other questions on this? So now let's do one final example of backward induction. And that will be another quantity competition game.

[0:46:35](https://youtu.be/VHex0a2JFWI?t=2795) So, so far we looked at Cournot competition. Here we'll look at something called Stackelberg competition. I mean, the names don't really matter.

[0:46:48](https://youtu.be/VHex0a2JFWI?t=2808) So in Cournot competition, remember, we had two firms that were each choosing how much to produce and how much to bring to market. But the assumption in Cournot was that the firms move simultaneously. And in Stackelberg, we're going to imagine that there may be a first mover. Maybe I harvest my crops first. I get to choose to bring them to market. And then the next farmer sees how much I bring to market and then chooses how much to bring themselves. So we have-- it's still quantity competition. I'll emphasize this.

[0:47:19](https://youtu.be/VHex0a2JFWI?t=2839) But it's-- sequential quantity competition would maybe be-- we often name these with people. But if you don't know the person, it's not a very meaningful name. So we might call this sequential quantity competition if we want a more descriptive title.

[0:47:37](https://youtu.be/VHex0a2JFWI?t=2857) So let's see how this works. We still have two firms, 1 and 2. And they are choosing quantities as before. And we have market demand-- maybe I'll say inverse demand P of Q equals to the maximum of 1 minus Q and 0. So this is exactly what we had before.

[0:48:08](https://youtu.be/VHex0a2JFWI?t=2888) The more quantity that's brought to market, the lower the price is. If quantity 1 is brought to market, the price is 0. If quantity 0 is brought to market, the price is 1. And the price can never go negative. So if tons is brought to market, we just say the price is 0. The market price is 0. And let's also assume marginal cost C. And as usual, we'll assume C is between 0 and 1.

[0:48:38](https://youtu.be/VHex0a2JFWI?t=2918) So remember, marginal cost C means whatever quantity Q of the good I produce, it costs me C times Q to produce that amount. So it's a linear cost-- constant marginal cost. But the key difference relative to Cournot competition is that firm 1 moves first.

[0:49:03](https://youtu.be/VHex0a2JFWI?t=2943) And then firm 2 observes firm 1's choice.

[0:49:16](https://youtu.be/VHex0a2JFWI?t=2956) So we can draw this. We can try to draw this in extensive form. It's a little tricky because they're making choices among infinitely many quantities. But if we wanted to draw it, we would say, here's firm 1. They're making an infinite choice over how much to produce. Maybe we'll call this Q1. Maybe I'll take a particular point here. And then we have firm 2, who's making an infinite choice over Q2.

[0:49:46](https://youtu.be/VHex0a2JFWI?t=2986) So it's a little hard to formally represent this in extensive form when we have infinitely many things. But what I mean here is player 1 chooses a quantity Q1 to bring to market, any non-negative number. Whatever quantity that is, player 2 observes it. I'll warn you that it's particularly hard to represent information sets with infinitely many players. So it's crucial here sometimes to write it in words that player 2 observes the quantity that's produced by player 1 because it's really hard to draw information sets. And then player 2 chooses the quantity Q2.

[0:50:26](https://youtu.be/VHex0a2JFWI?t=3026) So now let's write down payoffs. The payoffs are actually going to look exactly the same as they did in Cournot competition because the payoffs just depend on how much each firm brings to market. So this is kind of like our BoS game, where the payoffs are the same whether player 2 observes player 1's move or not, but the equilibria of the game are going to be very different depending on whether that move is observed. So what are our payoffs? Let's look at, say, firm 1.

[0:50:56](https://youtu.be/VHex0a2JFWI?t=3056) Well, as before, we want to say, what is firm 1's profit, pi 1, if they bring quantity Q1 to market and firm 2 brings quantity Q2 to market? So we always have to think of profit in two components. It's the profit per unit times the number of units you have. So it's going to be Q1 times what? Well, maybe I'll write it like this.

[0:51:30](https://youtu.be/VHex0a2JFWI?t=3090) Q1 is the amount I bring to market. I sell it at a price of P of Q1 plus Q2 because Q1 plus Q2 is the total quantity brought to market. That's what determines the price. And then, of course, I also have to pay my constant marginal cost per unit. But let me plug in a formula here, again with my little cheat that I'm going to assume Q is not so big that this is a negative number. So I'm just going to write this as Q1 times 1 minus Q1 minus Q2 minus C, where technically this formula only applies if Q1 plus Q2 is between 0 and 1, but in the end it will be.

[0:52:05](https://youtu.be/VHex0a2JFWI?t=3125) You might say, why are there two negatives? Well, remember, it's minus, in parentheses, Q1 plus Q2. And that's why I get two minuses. So now we've specified the payoffs. We still have to specify strategies, though. And this, I think, is where it gets a bit tricky. Any ideas? Let's start with player 1, firm 1. Maybe I'll call them F1 and F2 for firm. What is a strategy for firm 1?

[0:52:39](https://youtu.be/VHex0a2JFWI?t=3159) Yeah?

[0:52:40](https://youtu.be/VHex0a2JFWI?t=3160) AUDIENCE: Whatever value of Q1 would lead to the derivative of Q1 times \[INAUDIBLE].

[0:52:44](https://youtu.be/VHex0a2JFWI?t=3164) IAN BALL: Well, now you're talking about optimal. Sorry, I'm just saying, what is the set of all strategies, before we talk about optimum. So a strategy for-- I'm just saying, what is a strategy?

[0:52:52](https://youtu.be/VHex0a2JFWI?t=3172) AUDIENCE: \[INAUDIBLE] for any non-negative value.

[0:52:56](https://youtu.be/VHex0a2JFWI?t=3176) IAN BALL: Yeah, so it's just going to be Q1, which we have to choose exactly how much they can produce. Let's say it's a number between, say, 0 and infinity, so a non-negative number. Let's say that. The details don't really matter. Whether it's 0 to 1 or 0 to infinity, it doesn't really matter. But what about firm 2? This is where it gets quite subtle. What is a strategy for firm 2? Well, if we took our previous approach, we'd have to get a vector. And we'd have to have a different spot in the vector for every contingency.

[0:53:26](https://youtu.be/VHex0a2JFWI?t=3206) But now there's infinitely many contingencies. So that's going to become a bit messy. So we really need a function. What kind of function is this going to be? Well, firm 2 observes the quantity chosen by firm 1, and then they choose a quantity themselves. So sometimes it might be easier to do 0, 1, but I'll stick with it this way.

[0:53:58](https://youtu.be/VHex0a2JFWI?t=3238) So what does this say? What's the interpretation? Yeah?

[0:54:01](https://youtu.be/VHex0a2JFWI?t=3241) AUDIENCE: Is C going to be involved \[? in this one? ?] Is C going to be considered an input to the function? Or will C be implied?

[0:54:06](https://youtu.be/VHex0a2JFWI?t=3246) IAN BALL: No, so C is a commonly known parameter-- is known.

[0:54:11](https://youtu.be/VHex0a2JFWI?t=3251) AUDIENCE: So it is not considered an input to the function?

[0:54:14](https://youtu.be/VHex0a2JFWI?t=3254) IAN BALL: No, because C is like 2 here. It doesn't change-- C is a known parameter that affects the payoff functions, but it doesn't affect the definition of the strategies. So you think of it, in a game, there's a fundamental distinction between the set of strategies, what people can do, and the utility functions, which is how well they do when they choose those strategies. C is just-- we could call it a half if we wanted. It's just a known number that enters the utility functions but has no role in the strategies. One way of seeing this is if in this game, I made this a 3, it wouldn't change the set of strategies.

[0:54:48](https://youtu.be/VHex0a2JFWI?t=3288) It would just change the payoffs people get from those strategies. And I could call this alpha, and the same would be true. But that's important. If C was private information, then it would get tricky. But we're going to assume it's known and public. Everyone knows it, just like everyone knows this market function, big P. Great, so what's the interpretation of this? This says, for any Q1 that I observe, what is the amount Q2 of Q1 that I produce? So Q1 is the quantity that I observe for 1 producing hypothetically.

[0:55:23](https://youtu.be/VHex0a2JFWI?t=3323) It's not how much they actually produce, but it's if I were to observe them producing Q1, how much would I produce as firm 2? I would produce Q2 of Q1. OK, now, a big warning here. I think this is maybe the fault of sometimes the way we teach math.

[0:55:43](https://youtu.be/VHex0a2JFWI?t=3343) Q2 is a function. Q2 of Q1 is a number. And this creates a lot of confusion because, in math class, we often write f of x to mean a function. But really, f of x is a number. It's the value of the function at x. So it's really important in this class that we understand f by itself is a function, f of x is a number-- the value of the function evaluated at x. So in problem sets, it's really important that we understand that the strategy for firm 2 is the function Q2.

[0:56:17](https://youtu.be/VHex0a2JFWI?t=3377) People often say the strategy is Q2 of Q1, but no. Q2 of Q1 is a number. It's the quantity produced at a particular contingency, Q1. So it's really important that you understand function notation. If in doubt, you need to, on problem sets or on quizzes, be very clear about what you mean by things because if you say the strategy is Q2 of Q1, we're not going to really know if you mean a number or a function because of this ambiguity that we have. So try to be precise. Great. So now let's try to actually analyze this.

[0:57:07](https://youtu.be/VHex0a2JFWI?t=3427) So now what we'd like to do is apply backward induction to this game. Now, I know technically I said it only applies to finite extensive form games. But we are going to be able to apply it here, and it won't be a problem. So let's apply backward induction.

[0:57:25](https://youtu.be/VHex0a2JFWI?t=3445) And what are we going to do? Well, we're going to work backwards, like we always do. So let's start here. And we have to say, as a function of the amount Q1 that firm 1 has produced, what should firm 2 do? So let's consider maybe what I'll call contingency or infoset Q1. So this says, put yourself in the position of firm 2. You've observed how much firm 1 has brought to market.

[0:57:56](https://youtu.be/VHex0a2JFWI?t=3476) And you've observed that the amount they've brought to market is Q1. And now you're choosing Q2, how much you should bring to market yourself. So what are you choosing? Well, you're going to maximize over Q2-- Q2 is what you're choosing-- pi 2 of Q1, Q2. So this is essentially just what we did before when we talked about computing a best response. Given that Q1 is the amount produced, I want to choose Q2 to maximize my payoff where Q1 appears in here because this is fixed.

[0:58:32](https://youtu.be/VHex0a2JFWI?t=3512) I can't control this. This is already chosen. And this is what I can control. And what do we get? Well, this is Q2. I'm just going to copy this formula but switch it because we're now focusing on firm 2. 1 minus Q1 minus Q2 minus C. And if we use our trick, this is something times a constant minus that something. It's always optimal to just do the something the constant-- sorry, this is Q2 times the constant minus Q2.

[0:59:07](https://youtu.be/VHex0a2JFWI?t=3547) It's always optimal to set Q2 to be the constant divided by 2. So what we're going to get is Q2, maybe I'll say star of Q1 is equal to--

[0:59:22](https://youtu.be/VHex0a2JFWI?t=3562) so we're almost there. I just mechanically took a derivative. But there is one issue I have to be a little careful about. What if Q1 is really, really big? So now my assumption that, oh, we don't have to worry about negative prices, we don't have to worry about max or min, turns out we have to be a little bit careful. If Q1-- let's just think it through. If Q1 is a billion, what should firm 2 produce? Nothing, because if Q1 is a billion, then the market price is 0.

[0:59:53](https://youtu.be/VHex0a2JFWI?t=3593) My cost is C. So I definitely don't want to produce anything. And unfortunately, it turns out that if we just compute this number and check if it's positive, we'll get the right answer. So if it's positive, we're good. If it's negative, we set it to 0. And we often-- the notation I'll use for that is I'll put a little plus at the bottom of it. Or-- let me just write-- I like that notation, but I think that might just cause confusion. So let's just write max of this is 0. Need a little more space.

[1:00:25](https://youtu.be/VHex0a2JFWI?t=3625) Getting too cute here. So it's the-- we have 1 minus Q1 minus C over 2. But if this is negative, I just want to produce 0. So I'm going to write the maximum of this and 0. I could use the plus, but I think this is clearer.

[1:00:49](https://youtu.be/VHex0a2JFWI?t=3649) And I wrote Q2 star just to indicate that this is going to be part of my special backward induction strategy profile. And let's understand, Q2 star at a particular Q1 is exactly this number. But by going through this procedure, I'm going to obtain a function Q2 star. So Q2 star-- the function is firm 2's strategy.

[1:01:20](https://youtu.be/VHex0a2JFWI?t=3680) It's the function where, whatever Q1 you plug into it, this is what you get out.

[1:01:29](https://youtu.be/VHex0a2JFWI?t=3689) OK, so now we go one step further.

[1:01:35](https://youtu.be/VHex0a2JFWI?t=3695) So we work backwards. We figured out what happens at the second stage with firm 2. And now we have to figure out how firm 1 should behave.

[1:01:57](https://youtu.be/VHex0a2JFWI?t=3717) So now we're at firm 1. Technically, we're at the initial contingency. Nothing has really happened so far in the game. And what is firm 1's problem? They're maximizing over Q1 what? They're choosing Q1 to maximize what? Any thoughts here? Well, I think the first step is it's going to be a profit. So let's put pi 1 here. And if they choose Q1, well, definitely Q1 should be the first component in here.

[1:02:30](https://youtu.be/VHex0a2JFWI?t=3750) But the hard thing is, what do we put in the second component? If I choose Q1, what will firm 2 do? Yes?

[1:02:37](https://youtu.be/VHex0a2JFWI?t=3757) AUDIENCE: \[INAUDIBLE]

[1:02:39](https://youtu.be/VHex0a2JFWI?t=3759) IAN BALL: Exactly. So what I'll put in here is Q2 star of Q1. So notice what I'm doing here as firm 1. I'm choosing how much quantity to produce, recognizing that when I choose Q1, that's going to directly affect my payoff. But the Q1 I choose is also going to affect how much the other firm brings to market. And specifically, we can see that this is a decreasing function of Q1. So the more that I bring to market, the more the next farmer sees, wow, this person brought a lot to market.

[1:03:11](https://youtu.be/VHex0a2JFWI?t=3791) That's really going to bring down the market price. I'm going to bring less to market. So I can influence how much firm 2 brings to market through my choice of Q1. And now we're just going to write this, substitute everything in. And we'll use the trick that let's hope that we never deal with the 0. So let's assume everything's positive and solve it and see what happens. So we're going to get Q1 is the amount that I produce. Then we're going to get-- maybe I'll write it really carefully-- Q1 plus Q2 star of Q1 minus C because for each unit I produce, I have to pay a cost of C. And I'm going to get the market price.

[1:03:54](https://youtu.be/VHex0a2JFWI?t=3834) What's the market price? Well, it depends on the market production level, which is the sum of what I produce and what I'm going to induce firm 2 to produce. So this is Q1 times 1 minus Q1 minus Q2 star of Q1 minus C. And here, there's kind of a trick. We have 1 minus Q1 minus C minus, again, 1 minus Q1 minus C but over 2. So we have 1 minus Q1 minus C minus half of it.

[1:04:26](https://youtu.be/VHex0a2JFWI?t=3866) So we have half left. So we're going to get Q1 times 1 minus Q1 minus C over 2.

[1:04:36](https://youtu.be/VHex0a2JFWI?t=3876) The Q2 star is gone because I substituted in the expression for Q2 star above.

[1:04:47](https://youtu.be/VHex0a2JFWI?t=3887) And it turns out the half here doesn't actually affect the optimum, because we could just pull out the half to the front. And we're going to get Q1 star equals 1 minus C over 2 if we do the algebra. It can be tricky to see this half here. But remember, if we just put the half in the front, you can see-- if we multiply the objective by a half, that doesn't change the optimum. It changes the optimal value, but not the optimum. So we'll just get this. Now let's be really clear about this. The algorithm has told us a strategy profile.

[1:05:19](https://youtu.be/VHex0a2JFWI?t=3919) It said firm 1 is producing this amount, and firm 2 is using this complete contingent plan that specifies what they produce as a function of every Q1. But now we'd like to understand, well, how much is actually produced? Because all we know is firm 2's complete contingent plan. Let's figure out the actual production level for firm 2. What is that going to be?

[1:05:44](https://youtu.be/VHex0a2JFWI?t=3944) Well, we need to plug in-- what I want to look at is Q2 star is the strategy that backward induction gave us for firm 2. But now I'm going to plug into that function the strategy that backward induction gave us for firm 1, which is just a number, the quantity. And this is going to tell me how much is actually produced as opposed to firm 2's complete contingent plan. And I have to-- little careful here. It's 1 minus C over 2 minus quarter.

[1:06:18](https://youtu.be/VHex0a2JFWI?t=3978) And I think it should be 1 minus C over 4.

[1:06:25](https://youtu.be/VHex0a2JFWI?t=3985) So now we've analyzed the game. But now we can ask-- we've analyzed it mathematically. Let's ask some econ questions about it. And in particular, we've looked at how much each firm produces in the equilibrium. Does anyone recall, if we did Cournot with two firms, how much each firm produced? Anyone remember? We had the same-- were the same demand function, and the same costs.

[1:06:56](https://youtu.be/VHex0a2JFWI?t=4016) No? I'll just say it. It's 1 minus C over 3.

[1:07:03](https://youtu.be/VHex0a2JFWI?t=4023) So can anyone comment on the differences here? So in Cournot, Q1 star equals Q2 star equals 1 minus C over 3. So when the firms move simultaneously, they both produced 1 minus C over 3. When the firms move sequentially, firm 1 produced 1 minus C over 2, and firm 2 produced 1 minus C over 4. Any intuition for why we get this? What's the key difference here?

[1:07:33](https://youtu.be/VHex0a2JFWI?t=4053) Let's think of it from firm 1's perspective. What's the key difference in firm 1's optimization problem between the simultaneous-move Cournot game and the sequential-move game? Remember, we already saw in the Boston game that simultaneity versus sequential play can have a big difference. Yeah?

[1:07:51](https://youtu.be/VHex0a2JFWI?t=4071) AUDIENCE: Maybe in the Cournot game, they both know that they're choosing the same quantity. So they have to keep in mind that the other person could be doing the same thing while the other \[INAUDIBLE].

[1:08:03](https://youtu.be/VHex0a2JFWI?t=4083) IAN BALL: Stackelberg or sequential quantity competition? Yeah.

[1:08:06](https://youtu.be/VHex0a2JFWI?t=4086) AUDIENCE: \[INAUDIBLE] player 1 knows \[INAUDIBLE] player 2's going to know. Player 2's obviously not going to \[INAUDIBLE].

[1:08:13](https://youtu.be/VHex0a2JFWI?t=4093) IAN BALL: So I think the one word you said there was crucial. You said firm 1 can influence firm 2, and I think that's the crucial thing. So let's think about the Nash equilibrium reasoning in Cournot. In Cournot, I'm taking as given the amount that my opponent is producing. And I'm saying, given the amount that they're producing, the amount that I'm producing is optimal, taking as fixed the amount they produce. Here, it's fundamentally different because, because I move first, I know that my choice of quantity affects how much they produce. So I know if I increase my production, I can cause them to decrease their production.

[1:08:46](https://youtu.be/VHex0a2JFWI?t=4126) So that makes more production for firm 1 more attractive. In general, I don't want the other person to produce because that brings down the price. So if I know that by raising my production, I can decrease their production, that gives me a further incentive for me to raise my production level. So that's why firm 1 produces more. Firm 2's choices actually looks very similar. Firm 2 takes firm 1's quantity as given. So firm 1's incentives are essentially the same-- sorry, firm 2's incentives are essentially the same. But the issue is firm 1 is producing so much more.

[1:09:17](https://youtu.be/VHex0a2JFWI?t=4157) It's optimal for them to produce a lot less than in Cournot.

[1:09:24](https://youtu.be/VHex0a2JFWI?t=4164) So what's key? The key word you used is "influence."

[1:09:30](https://youtu.be/VHex0a2JFWI?t=4170) Can my quantity influence the quantity that you choose? All right, so now we've found-- we've applied backward induction. We found a particular Nash equilibrium that seems reasonable to us. It doesn't involve any credible-- any non-credible threats. But it's still good to keep in mind, could we find any other Nash equilibria that do involve credible threats?

[1:10:03](https://youtu.be/VHex0a2JFWI?t=4203) So could we find other Nash equilibria that maybe we don't find compelling, but just to understand what's going on here? What would be a threat that firm 2 might be tempted to make? If we think back to our reasoning about-- we gave the example of, I'll threaten to always go to the Red Sox game whatever you do. What might be a threat that firm 2 might want to impose to try to benefit themselves, if we had a very smart firm 2? Yeah?

[1:10:30](https://youtu.be/VHex0a2JFWI?t=4230) AUDIENCE: Make a threat to produce a lot to drive the price down so that firm 1 won't make any money, and then, yeah, they're both just worse off.

[1:10:37](https://youtu.be/VHex0a2JFWI?t=4237) IAN BALL: Great, exactly. So firm 2 can make a threat. Firm 2 might threaten-- there's even a term for this-- to flood the market, meaning produce a huge amount of quantity to pull down the market price.

[1:11:02](https://youtu.be/VHex0a2JFWI?t=4262) But let's say a little more. When would they want to impose this? When would they flood the market? Under what conditions or what circumstances?

[1:11:13](https://youtu.be/VHex0a2JFWI?t=4273) What does firm 2 want firm 1 to do? Produce a lot or a little? You can answer or someone else can answer. Yeah?

[1:11:21](https://youtu.be/VHex0a2JFWI?t=4281) AUDIENCE: Well, I guess firm 2 would want firm 1 to produce 1 minus C over 3. But--

[1:11:25](https://youtu.be/VHex0a2JFWI?t=4285) IAN BALL: Or in general, they actually might want them to produce either even less. But in general, they want them to produce not very much. So when would they threaten-- when would they flood the market? Under what--

[1:11:33](https://youtu.be/VHex0a2JFWI?t=4293) AUDIENCE: When they produce too much.

[1:11:34](https://youtu.be/VHex0a2JFWI?t=4294) IAN BALL: When they produce too much. So what firm 2 might say is, look, you better keep your production level low because if you produce too much, I'm going to flood the market. So firm 2 might threaten to flood the market if firm 1 produces too much.

[1:11:55](https://youtu.be/VHex0a2JFWI?t=4315) And this actually gets into dynamics that we'll study later. Oil cartels essentially function exactly in this way. I might say, you need to keep production low, and if you don't, I'm going to punish you by flooding the market Yes?

[1:12:05](https://youtu.be/VHex0a2JFWI?t=4325) AUDIENCE: How would this be a credible threat, though? Because firms 2 loses money by flooding.

[1:12:10](https://youtu.be/VHex0a2JFWI?t=4330) IAN BALL: So this will not be a credible threat. That's why it didn't survive our backward induction procedure. But it will be a Nash equilibrium. So what I want to illustrate is how a Nash equilibrium can involve non-credible threats. Backward induction rules out those non-credible threats, but Nash equilibrium can allow them if we design them carefully. Other question? Yeah?

[1:12:29](https://youtu.be/VHex0a2JFWI?t=4349) AUDIENCE: If you remove some of the simplifying assumptions we make for this, can you make it a credible threat with things like infrastructure and capital investments in production capacity?

[1:12:39](https://youtu.be/VHex0a2JFWI?t=4359) IAN BALL: I think the best way to make it credible would be to say it's a repeated game, and that's what we'll study later on. We're assuming it's a one-off thing. But really what might happen is, every day we come to market, and I might say, well, if day after day you keep doing this, then I'm going to do this. And then we can get these complicated rewards and punishments over time. And that's going to be a big focus of the second part of our course. So that's a good way to make it credible. Yes?

[1:12:59](https://youtu.be/VHex0a2JFWI?t=4379) AUDIENCE: If it were a Nash equilibrium-- so this is a Nash equilibrium. Is it the idea that firm 1, knowing firm 2 can threaten the market, will deviate?

[1:13:07](https://youtu.be/VHex0a2JFWI?t=4387) IAN BALL: Exactly, so let's actually construct our Nash equilibrium because we haven't constructed one yet. So let's fix some number, Q1, which is, let's say, really small. So let's say-- it doesn't really matter exactly, but we do--

[1:13:32](https://youtu.be/VHex0a2JFWI?t=4412) yeah, let's say 1 minus C over 4. It doesn't matter. And now what is Q2 star going to be?

[1:13:50](https://youtu.be/VHex0a2JFWI?t=4430) Well, let's use our idea that-- there's a few different ways we can do it, but let's use an idea that they're going to threaten to flood the market if production is too high. So let's look at two cases. If Q1 is less than-- let me call this Q1 bar. If Q1 is less than Q1 bar and if Q1 is greater than or-- less than or equal to and greater than Q1. So if Q1 is greater than Q1 bar, firm 2 is going to flood the market.

[1:14:20](https://youtu.be/VHex0a2JFWI?t=4460) Again, the exact amount doesn't matter. Let's say 100. It doesn't matter.

[1:14:27](https://youtu.be/VHex0a2JFWI?t=4467) What if Q1 is less than or equal to Q1 bar? We actually have to be a little careful to make sure it's still a Nash equilibrium. So let's follow our idea here, and let's make this 1 minus Q1 minus C over 2. There's many examples. And we could figure out exactly how big Q1 bar can go. But I think this one should work. And let's just see that we understand that this is a Nash equilibrium. So given Q2's strategy, why is this the optimal choice of Q1?

[1:14:58](https://youtu.be/VHex0a2JFWI?t=4498) Can anyone explain in words? I mean, we see the math, but can we explain it in words?

[1:15:06](https://youtu.be/VHex0a2JFWI?t=4506) Well, why don't I want to produce more than this? Yeah?

[1:15:10](https://youtu.be/VHex0a2JFWI?t=4510) AUDIENCE: That's the most you can produce without the market getting flooded.

[1:15:12](https://youtu.be/VHex0a2JFWI?t=4512) IAN BALL: Exactly. So if I produce-- it's the most I can produce without flooding. And you can check that, when the market's not flooded, my payoff is going to be increasing at least until this point. So if I produce less than this, the market doesn't get flooded. But I'm producing so little, I'm not doing very well. If I produce more than this, that would be good if the market didn't get flooded. But if I produce more than this, the market gets flooded, so I don't want to do it. So player 1 is behaving optimal given firm 2's strategy. How can we see that firm 2 is behaving optimally given firm 1's strategy?

[1:15:43](https://youtu.be/VHex0a2JFWI?t=4543) What's the one-- the only one aspect of this strategy that's important if we want to check that it's a Nash equilibrium? What's important is, when firm 1 chooses Q1 bar, which is in this case-- firm 2 responds optimally because that's the only thing that affects firm 2's payoff, what they do in the contingency that's actually reached. So you can check that, given that firm 1 is producing this amount, firm 2 is going to produce 1 minus Q1 bar minus C over 2. And it turns out that's optimal for firm 2 given firm 1's behavior.

[1:16:15](https://youtu.be/VHex0a2JFWI?t=4575) So this will be a Nash equilibrium. But it doesn't survive backward induction. What is a contingency at which firm 2 is behaving suboptimally?

[1:16:29](https://youtu.be/VHex0a2JFWI?t=4589) Well, really anything up here-- actually, not if Q1 is really, really high. But if Q1 is anywhere between Q1 bar and 1 minus C, firm 2 could respond and make positive profits, but they're instead flooding the market, causing the price to drop to 0, and making profits of 0. So at unreached contingencies, firm 2 is behaving suboptimally. That's why this doesn't survive backward induction. But at the reached contingency, they are behaving optimally. And that's why it constitutes a Nash equilibrium. And let me stop there, and I'll see you on Thursday.

[1:17:00](https://youtu.be/VHex0a2JFWI?t=4620) Thank you.
