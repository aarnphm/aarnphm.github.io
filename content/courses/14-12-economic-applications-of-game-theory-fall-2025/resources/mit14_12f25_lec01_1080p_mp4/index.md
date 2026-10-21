---
title: 'lecture 1: introduction to individual decision making'
description: "In this first lecture of the course, Ian Ball explains the basics of game theory, a mathematical framework for analyzing decision-making when the outcome depends on the actions of other people. In simpler terms, it's interactive decision-making."
id: 'mit-14-12-fall-2025-resource-resources-mit14-12f25-lec01-1080p-mp4'
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
    n: 1,
    youtube: 'WRibE2nt8wM',
    duration: 3433,
    cues: 1353,
    captions: 'static_resources/1ujkUiHKqtcJRKF_1szj0FgHPBPE1rXeC_transcript.webvtt',
    transcript: 'static_resources/1ujkUiHKqtcJRKF_1szj0FgHPBPE1rXeC_transcript.pdf',
  }
layout: 'resource'
license: 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
license_url: 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
---

[0:09](https://youtu.be/WRibE2nt8wM?t=9) IAN BALL: OK, great. So let's now say a little bit about what this course is. This is a course on game theory or on the economic applications of game theory. So I think a natural starting point is to ask what is game theory. Well, here's two definitions, both from people who've won the Nobel Prize for their work in game theory. The first is by Roger Myerson, who says that game theory is the study of mathematical models of conflict and cooperation between intelligent, rational decision makers.

[0:40](https://youtu.be/WRibE2nt8wM?t=40) Robert Aumann, also a Nobel laureate, gives just a briefer definition, which I think encapsulates the key ideas. It's interactive decision theory. It's about people making decisions when their decisions interact. What does it mean for people's decisions to interact? Well, I think the best way to see it is just an example. So we're going to play a game as a class to see what it means to be a hopefully rational and intelligent interacting decision maker. And then after that, I'll get into the formal material on the board. So to play games in the class, we're going to use a platform called Moblab.

[1:10](https://youtu.be/WRibE2nt8wM?t=70) So let me just describe the game and then I'll hit Begin. You're each going to be assigned a random number between 1 and 100. It's going to be an integer. And then you're each going to make a guess, which is also a number between 1 and 100. The winner of the game-- I'm sorry, there's no real prizes but pride-- the winner of the game is the person who guesses closest to 2/3 of the average of everyone's number in the class.

[1:30](https://youtu.be/WRibE2nt8wM?t=90) AUDIENCE: The number they guess or the number they were assigned?

[1:33](https://youtu.be/WRibE2nt8wM?t=93) IAN BALL: The number they guess. So you're assigned a number. Only you know that. No one else knows that number. You're going to make a guess. And the winning guess is the guess that's closest to 2/3 of the average of everyone in the class's guess. Yes.

[1:44](https://youtu.be/WRibE2nt8wM?t=104) AUDIENCE: Is everyone's number drawn from a uniform distribution?

[1:46](https://youtu.be/WRibE2nt8wM?t=106) IAN BALL: Yes. Any other questions? Yes.

[1:49](https://youtu.be/WRibE2nt8wM?t=109) AUDIENCE: What's the range again?

[1:49](https://youtu.be/WRibE2nt8wM?t=109) IAN BALL: 1 to 100. Yeah. Any other questions? I think these details should be there, but sometimes they're a little imprecise in the way they write it. OK, so let's begin.

[2:04](https://youtu.be/WRibE2nt8wM?t=124) All right. Here we are. So let's see. So here on the horizontal axis are the guesses that people made. Again, to clarify, not the numbers you got, but the guesses that people are making. And on the vertical axis, we're showing the percentage of people who made that particular guess. So we have a few. It's a bit maybe hard to see with the color, but we have a few summary statistics here. We see that the average was 36 roughly. And that's the dotted purple line here. And the winning guess was the guess that was closest to 2/3 of 36, so about 24.

[2:38](https://youtu.be/WRibE2nt8wM?t=158) And we see that guess is here. Does anyone want to share their thought process? Maybe someone who was quite high up here, what they were thinking. Or low? Anyone who's willing to share what they were thinking about? What about the winner? They should be able to come forward. Who won? Yes. You won. OK, tell me, what was your reasoning?

[3:00](https://youtu.be/WRibE2nt8wM?t=180) AUDIENCE: I wouldn't replicate the strategy. I think it's not the best idea. But what I did was put the mean of a uniform distribution from 0 to 100 would probably be around 50.

[3:09](https://youtu.be/WRibE2nt8wM?t=189) IAN BALL: Yeah.

[3:10](https://youtu.be/WRibE2nt8wM?t=190) AUDIENCE: I'd take 2/3 of that. But I thought probably other people would be thinking the same thing. So I downscaled it a bit more by 2/3.

[3:17](https://youtu.be/WRibE2nt8wM?t=197) IAN BALL: Great. Great. So you're showing the foundation of game theoretic reasoning, what we talked about, interactive decision making. You didn't just think about your number. You had to think about other people's numbers. And then you had to think about what other people will do given the numbers that they got. And you adjusted your strategy accordingly. And in this case, it worked out pretty well. Anyone else want to share maybe a bid they're happy with, a bid they feel they made a mistake on? Any other thoughts? Yes.

[3:43](https://youtu.be/WRibE2nt8wM?t=223) AUDIENCE: I did 26.

[3:44](https://youtu.be/WRibE2nt8wM?t=224) IAN BALL: 26, OK. So you were pretty close, yeah?

[3:46](https://youtu.be/WRibE2nt8wM?t=226) AUDIENCE: Yeah. I thought maybe something like 30, and then figured that I was probably wrong in whatever reason I would have, so I just lowered it.

[3:52](https://youtu.be/WRibE2nt8wM?t=232) IAN BALL: OK that's fair. I mean, maybe they don't want to out themselves, but I'm kind of curious. People that guessed very high, any reasoning about that? No, maybe they don't want to say anything. OK, let's play it one more time and see what happens.

[4:06](https://youtu.be/WRibE2nt8wM?t=246) Ah, so quite different this time. So now the average was 21 and the winning choice was 14. That's interesting. So this happens every year. So people are very consistent. So the distribution has changed a bit. So some people must have decided to bid less this time. Anyone who bid less this time, do they want to share why? Very quiet. Anyone?

[4:33](https://youtu.be/WRibE2nt8wM?t=273) AUDIENCE: It's basically tho idea that you would keep getting 2/3 and 2/3 and 2/3 of what you previously thought like would be a good guess.

[4:42](https://youtu.be/WRibE2nt8wM?t=282) IAN BALL: OK. Can you say a little more? So you thought something was a good guess, and now you're bidding even less. And why is that?

[4:48](https://youtu.be/WRibE2nt8wM?t=288) AUDIENCE: Like I initially thought something was a good guess. And then I was like, OK, but if I think this, then what if other people think the same? So I'm going to give 2/3 of that. But then 2/3 again of that. And then I'm going to give 2/3 of that.

[4:59](https://youtu.be/WRibE2nt8wM?t=299) IAN BALL: Right. So what we see is that interactive decision making sometimes creates this regress. So first, I have to think about my number. That's certainly relevant. Then I have to try to think about what other people are going to do. So I have to think about what number they have. But now I have to put myself in the shoes of those other people and say, what are they bidding? What are they going to guess? Well, they're thinking about what I'm doing. But when they try to think about what I'm doing, they have to think about what I think about what they're doing. So I have think about what would they think about what I think what they think. And all of a sudden, it gets very, very hard. And this is exactly what game theory is trying to address. I think for a while historically, if you go back to the history of thought, people thought, it's just an infinite regress.

[5:33](https://youtu.be/WRibE2nt8wM?t=333) There's no way we can try to analyze games like this. And game theory is a formal way to try to think about this kind of reasoning process. This also happens every year. A lot of people shift to 100 after we talk about it. Why? I think some people are kind of mischievous. They want to try to shift the results and get it wrong. So I think this highlights a really important thing about game theory, that when we study game theory and when we analyze a game, a really important modeling choice is writing down what people's preferences actually are, what people want. So if you analyze this game assuming that people's goal was to win, your predictions might be wrong if some people's goal is not actually to win.

[6:10](https://youtu.be/WRibE2nt8wM?t=370) So I think some people's goal might be to make my prediction look a little silly or to make their friends laugh or to mess with things. That's fine. That's their preference. But we need to make sure we accurately capture people's preferences when we model games. Because if we don't, our predictions are obviously going to be wrong. Another thing, though, that we have to keep in mind is, especially when comparing the results we get in a lab or in a classroom setting to the real world, is that the stakes really matter. So we have a lot of students-- maybe a few students who are kind of mischievous. If you won \$10 for guessing, well, I think some of those students wouldn't be so mischievous.

[6:42](https://youtu.be/WRibE2nt8wM?t=402) And if you won \$100,000 for winning, I think very few students would be mischievous. And I think once we get high enough, essentially no students would be mischievous. So the stakes and what you actually get for winning are going to affect the preferences that people have in this game. So let's move on. The plan is to move on to the formal analysis, unless there are any questions about this game. I should say this is sometimes called the Keynesian beauty contest. So if you want to google it or read about it, that's what it's often called. Yes.

[7:09](https://youtu.be/WRibE2nt8wM?t=429) AUDIENCE: So the average wasn't the average of numbers randomly assigned by the game to everyone, but the average of the choices?

[7:18](https://youtu.be/WRibE2nt8wM?t=438) IAN BALL: Exactly right. So with this many people, the average of the numbers that people got is almost certain to be very close to 50. But this average is the average choice, so the average guess of people in the class. And that's why when we move from the first round to the second round, I didn't see all the realizations, but it's likely that the average number that people received was basically the same in the two rounds. But the average guess changed dramatically because people behaved differently in the second round after we had this discussion. And we could keep doing this. We have limited class time. But my guess if we played this 10 times, we might see people getting very, very close to 0.

[7:51](https://youtu.be/WRibE2nt8wM?t=471) And my guess is then some people at 100 to mess with things. Was there a question here or comment? Yeah.

[7:58](https://youtu.be/WRibE2nt8wM?t=478) AUDIENCE: I didn't get a number. It was just guest. I don't know if there was a big number assigned to it.

[8:03](https://youtu.be/WRibE2nt8wM?t=483) IAN BALL: Oh, maybe I'm-- oh yeah, you're right. Sorry. There's no number. Yeah, good point. I'm getting ahead to the next game we play. You're right. Sorry. You weren't assigned a number. Yeah, sorry. That was very confusing. We play another game where you get a number, and fair enough. Yeah, you don't get a number. There's no uniformity. Yeah, fair enough. OK, I'm getting ahead of myself. Thanks for the clarification. OK, let us get to the board.

[8:50](https://youtu.be/WRibE2nt8wM?t=530) There we go. So on the problem set, you'll analyze a variant of this game where you'll get a number. So that'll be coming.

[9:10](https://youtu.be/WRibE2nt8wM?t=550) Let's open this up.

[9:32](https://youtu.be/WRibE2nt8wM?t=572) OK, so let's get started. So first let me say a little bit more about what we said was the definition of game theory which we said was the study of interacting rational decision makers.

[10:03](https://youtu.be/WRibE2nt8wM?t=603) So let's break down each of the terms in this definition. Let's go backwards from easiest to hardest. So I think the basic thing is that we're studying decision makers. We're studying people who make decisions. And that means people are making choices. So another good word for decision that we'll use a lot in the course is people are making either choices or we might also call it actions. And game theory is a very abstract, general mathematical theory that applies to a huge range of choices and actions.

[10:33](https://youtu.be/WRibE2nt8wM?t=633) So in the game you just played, where you chose a number to guess, that was an example of a choice. Whether you go to MIT is another example of a choice or action that we could analyze in game theory. What price a firm sets is another example of a choice. Which country you move to, what job you choose, how many hours a week you work, these are all these different questions-- what platform a political party chooses coming up to an election, who someone votes for, these are all actions or choices that people make. And the generality and abstractness of game theory is what allows it to be applied to so many different settings.

[11:08](https://youtu.be/WRibE2nt8wM?t=668) And throughout the course, we're going to talk about various applications of game theory, mainly within economics, but also to some things beyond economics, like political science and political bargaining. What does rational mean? So this is certainly a sticking point of game theory. If you've mentioned game theory to someone, whenever I'm on a plane and people ask me what I do and I say I'm a game theorist, I always get the same response. But aren't people not rational? You probably don't get asked about game theory on planes as much as I do, but maybe someday you will. And I want to be cautious about how we interpret rationality.

[11:41](https://youtu.be/WRibE2nt8wM?t=701) In some contexts, this is a very bad assumption. There's no question about that. But in other contexts, I think it's a better assumption than people might think. And the reason for that is that the way we use the term rationality within game theory is a bit different than the way we use it colloquially in everyday language. So what you might often hear someone on the street say is, it's so irrational for this person to like this political candidate. In our model of rational choice, we would never say someone's preferences are irrational.

[12:15](https://youtu.be/WRibE2nt8wM?t=735) If that's what they like, that's what they like. So rationality, I think a better way to think of it is it's really about consistency.

[12:24](https://youtu.be/WRibE2nt8wM?t=744) The agents in the models that we study are going to act towards achieving some consistent objective. You might disagree with the objective they have. They might like a different flavor of ice cream or a different political candidate or put different weight on income and leisure. That's fine. We don't all have to have the same preferences, but the assumption is that people consistently act with an eye towards these preferences that they do have. So sometimes, there's a phrase that we say and I like to say is that preferences cannot be irrational.

[13:01](https://youtu.be/WRibE2nt8wM?t=781) What can be rational in the context of game theory choices giving preferences. So liking chocolate ice cream is not irrational. Even if you really like vanilla, if your friend likes chocolate, that's fine. But if your friend likes chocolate and they choose vanilla, we would call that choice irrational because they're not maximizing their objective. Now, consistency-- now I've shifted the goalposts. Instead of trying to defend rationality as a game theory assumption, I'm now trying to mention consistency. Again, sometimes it applies, sometimes it doesn't. My guess is your preferences today as 20-year-olds are very different from the preferences you had 18 years ago.

[13:37](https://youtu.be/WRibE2nt8wM?t=817) So if we're looking at a model of people's choices over life, this probably isn't a very good description, or at least we might need to enrich the model to capture that. But if we're looking at your choices or preferences today and tomorrow, it probably is a pretty good assumption. And then interacting. This is the key thing that distinguishes game theory from a lot of other decision problems that we study in economics. And it's exactly what we saw in this first game. It's that multiple people are making decisions. But it's more than that. There are a lot of situations where we all make decisions, but there's no interaction between the decisions.

[14:10](https://youtu.be/WRibE2nt8wM?t=850) So we don't really have to take into account what decisions other people make. What distinguishes game theory and what I mean by interacting decisions is that when I'm choosing what decision to make myself, I have to take into account what other people are going to do. Because what other people do affects my preferences. So this is often called-- the situation is often called strategic interdependence.

[14:38](https://youtu.be/WRibE2nt8wM?t=878) What's best for me depends upon what other people do. So a classic example where this is used would be penalty kicks or tennis serves. Should the shooter shoot left or right? Well, there's no inherent reason to shoot left or right. But whether they should shoot left or right depends upon which way the goalie is going to dive or which way the tennis serve receiver is planning to play. The best action for one player depends upon what other people are doing. And that's going to be what we study today. So that's this course. What about this class today?

[15:09](https://youtu.be/WRibE2nt8wM?t=909) We're actually going to drop interacting and just study a single rational decision maker. That's going to be our baseline for today. Because before we can study how rational decision makers interact, we have to study how rational decision makers act in isolation. So today-- and then the rest of the course will be about interaction-- today we're going to study what's called individual decision making. Of course, up here, we still have individuals. But the key is that they're interacting with other individuals.

[15:41](https://youtu.be/WRibE2nt8wM?t=941) So maybe individual is not the best term, but this is what's used. We're going to study individual decision making.

[15:50](https://youtu.be/WRibE2nt8wM?t=950) I want to make one final comment about the scope of game theory, which is the question of who's taking these actions. We often think of it as an individual making choices. But it can be applied much more broadly. So there's a lot of applications of game theory to evolution, where we think of animals or species kind of collectively taking actions. We might also think about firms or organizations, governments taking actions, committees taking actions. And I think now with the rise of artificial intelligence, there's a lot of interest in thinking of the agents in our models as being an LLM or some algorithm that's taking an action and interacting with people.

[16:25](https://youtu.be/WRibE2nt8wM?t=985) So one view that is kind of becoming common, we can debate about it, is that the rationality assumptions of game theory become more useful and more compelling in a world of artificial intelligence. We can debate about that, but that's a view that some people express. OK, so here we are, individual decision making. Let's start with a very simple problem--

[16:48](https://youtu.be/WRibE2nt8wM?t=1008) a simple what I might call decision or choice problem.

[16:54](https://youtu.be/WRibE2nt8wM?t=1014) Nothing deep here. Nothing that interesting about the problem. But I just want to show how we can set it up. So let's say that you're choosing between three things. Let's say you're at a cafe and it's coffee, espresso, and tea. So we have three choices-- Coffee which I'll label C, espresso which I'll label E, and tea, which I'll label T. And let's not think about prices. Of course, maybe coffee is cheaper. But for now, let's not think about prices. You just get a choice. Your friend's paying.

[17:25](https://youtu.be/WRibE2nt8wM?t=1045) You're choosing between these three things. The first component that we need to think about in this decision problem is, what are these preferences? Remember, we said that we're studying decision makers who consistently maximize some of preference. So we need to specify this agent's preferences.

[17:44](https://youtu.be/WRibE2nt8wM?t=1064) And in this case, their preferences are going to specify some ordering among these three choices. So a great way to represent this in this context would be C better than E better than T. So these are probably my preferences. What does this mean? It means I prefer drinking coffee to drinking espresso. I prefer drinking espresso to drinking tea. And the way I've written this suggests transitivity, which you will assume that if I prefer coffee to espresso and espresso to tea, then I must also prefer coffee to tea.

[18:16](https://youtu.be/WRibE2nt8wM?t=1096) I won't write that out explicitly, but that's implied by this. And there's an implicit transitivity assumption that we're making. But I think this actually raises kind of a difficult philosophical question. What does it mean to prefer coffee to espresso? What is a preference? Philosophers talk about this. Maybe people who study cognitive sciences go to the lab and they try to measure people's brain activity and say, oh, when you drink coffee, a different neuron activates. Philosophers will get into these long debates about what it means to have a preference. In economics, we take a much more operational view of preferences.

[18:50](https://youtu.be/WRibE2nt8wM?t=1130) What does it mean? And it's almost tautological. What does it mean that I prefer coffee to espresso? It means when I'm faced with a choice between coffee and espresso, I choose coffee. That's all it means. So this is the decision based view of preferences that's predominant in economics. So let's say what is the meaning of this?

[19:12](https://youtu.be/WRibE2nt8wM?t=1152) The meaning is that if I'm given a choice between coffee and espresso, I choose coffee. I'm not going to be too precise here. We might argue about strict preferences versus weak preferences. And maybe we can be indifferent between things. But for now, let's just think about strict preferences. And preferences are a pairwise object, right? So a preference says for any two things, which do I prefer between them? So one way of thinking about what we mean by preferences is we're describing what an agent would choose from any pair or any menu containing two items.

[19:46](https://youtu.be/WRibE2nt8wM?t=1186) And then from that primitive preference, we can talk about what they would choose in richer problems. Now, an issue is that these little squiggly inequality signs are kind of messy to deal with. So it often becomes more convenient in economics to use a utility representation of preferences. So let me go to a new board for this. And this didn't go all the way up. Let's see. There we go.

[20:12](https://youtu.be/WRibE2nt8wM?t=1212) And I'll define the word ordinal in a second. But I'm going to talk about an ordinal utility representation of preferences.

[20:31](https://youtu.be/WRibE2nt8wM?t=1231) Why do we use this? Well, it's just more convenient. Especially when we have a lot of things, specifying all my pairwise preferences between these things can get kind of cumbersome. So what we do is we just specify a function called a utility function. So let's give just an example here. So maybe my utility function is u of C equals 5, u of E equals 4, and u of T equals 1. This should be one utility representation of these preferences.

[21:03](https://youtu.be/WRibE2nt8wM?t=1263) Why do these utilities represent these preferences? Well, do I prefer C to E? Yes, because the utility assigned to C is higher than the utility assigned to E. Do I prefer E to T? Well, yes, because the utility assigned to E is strictly greater than the utility assigned to T. And if we have a utility function like this, you can check that our induced preferences are going to satisfy transitivity as well. Because if one number is bigger than another and that other number is bigger than a third number, then the first number is bigger than the third number.

[21:35](https://youtu.be/WRibE2nt8wM?t=1295) But we have an issue here that this is only a representation. So I can also write down an alternative utility function. Maybe I'll call this u1. Let me now write down a utility function u2 that's this.

[21:59](https://youtu.be/WRibE2nt8wM?t=1319) What about these preferences? Are these the same preferences or are these different preferences than the first utility function? What do people think? I don't think there's really one right answer.

[22:11](https://youtu.be/WRibE2nt8wM?t=1331) AUDIENCE: In a marginal sense, it's the same.

[22:13](https://youtu.be/WRibE2nt8wM?t=1333) IAN BALL: The same, right? So it is true that if this were my utility representation, it still says I prefer C to E, I prefer E to T, and I prefer C to T. So this is an alternative representation of the same preferences. And this is what we mean by an ordinal utility representation. Only the order matters. Let me say this.

[22:46](https://youtu.be/WRibE2nt8wM?t=1366) And because only the order matters, that has a few implications. The first is that there's many utility representations of the same preferences.

[22:58](https://youtu.be/WRibE2nt8wM?t=1378) Many ordinal representations.

[23:12](https://youtu.be/WRibE2nt8wM?t=1392) And a second I think more subtle point is that the units and the values in the utility function don't have meaning.

[23:27](https://youtu.be/WRibE2nt8wM?t=1407) In fact, what are the units of utility functions? They don't really have a unit that makes sense. Sometimes we call the units utils.

[23:37](https://youtu.be/WRibE2nt8wM?t=1417) But from a purely preference standpoint, you might look at this and you might say, oh, I like C more in this world than in this world. But no, the way we think about ordinal representations is that what matters are your preferences. And there's a convenient mathematical way to represent them like this. And this is another convenient mathematical way to represent them. And they represent the same preferences. And there's no meaning associated with these utility functions in this ordinal world. Yes.

[24:03](https://youtu.be/WRibE2nt8wM?t=1443) AUDIENCE: Would you have to specify a null option, then? Let's say if the utility is negative 100 for T, would you have to specify that they could also choose nothing?

[24:12](https://youtu.be/WRibE2nt8wM?t=1452) IAN BALL: Great, that's a good point. So when we write a choice problem, it's really crucial-- and it's implicit in what I did, but I didn't say it explicitly, so thanks for bringing this up-- that I can make only one choice, and every possible choice is included. So if you have four options, if only three of them are included, that's not going to capture your preference problem-- your choice problem. So really, if I were being careful, probably this is actually a more complicated problem. And there's a third option, which is not drinking any drink. And I should really specify the utility here. So in this simple thing I'm imagining for some reason you have to make one of these three choices.

[24:44](https://youtu.be/WRibE2nt8wM?t=1484) But you're right. In general, when we model a choice problem, it's important to capture every possible choice the agent could make. On the other side, indeed, if it's possible for me to choose both coffee and espresso, then that needs to be formally modeled as another choice, the bundle where I get coffee and espresso, because that's a separate choice that I can make. Good point. OK, so this is ordinal utility representations. And now I'm going to move to the opposite, cardinal representations. And everything I said here is going to go the other way. So let's be really clear. Everything I just said is going to be wrong in a different context.

[25:16](https://youtu.be/WRibE2nt8wM?t=1516) So let's be very careful. So this is ordinal. The issue is that often in game theory, we face decisions under uncertainty. And when we face decisions under uncertainty, we need to model preferences a bit differently. So let's now move to decisions under uncertainty.

[25:41](https://youtu.be/WRibE2nt8wM?t=1541) What is a decision under uncertainty mean? It means when I make a choice, I don't know exactly what the consequence of that choice will be. When I buy an index fund or buy a stock, I don't know whether that stock is going to go up or down. I can't choose-- of course, if I could choose the stock that made more money, I would choose that one. But that's not my choice. I'm choosing one stock that has some distribution over outcomes that I don't know what the outcomes will be. And then I might choose another stock. And again, I don't know exactly what the outcome of that stock will be. And that's what we mean by uncertainty. So let's look at, again, a very, very simple problem, not economically interesting, but just to convey the ideas, which is the choice of how to get home.

[26:18](https://youtu.be/WRibE2nt8wM?t=1578) Let's say you're choosing whether to walk home or to take the subway. So you can either walk home or take the subway. Maybe we'll call that and we'll call this W. And again, here we're assuming you have to get home. Maybe there's a third option where you stay at work all day. But we're not going to model that.

[26:41](https://youtu.be/WRibE2nt8wM?t=1601) And now why is this a decision under uncertainty? Well, you don't know what the weather is going to be like. You have a long walk home, so the weather could change as you walk home. And we're going to assume there's two ways the world could be again. Weather it's more complicated, but this is a simplification of reality. It could be sunny or it could be rainy.

[26:58](https://youtu.be/WRibE2nt8wM?t=1618) Maybe we should really model every possible temperature and every possible level of precipitation, but we're not going to do that. And the way things work is when it's sunny, you'd rather walk home and when it's rainy, you'd rather take the subway. So maybe I'll write this over here. If sunny, you prefer-- and I think sometimes I call this the T to avoid the S. Let's call this the T. Maybe TA, just so we don't see the two S's.

[27:33](https://youtu.be/WRibE2nt8wM?t=1653) Sunny, you prefer to walk home. And if rainy, you prefer to take the T. So how would you approach this problem? And this is a very simple problem, a simple choice people make all the time. How would you approach this?

[27:53](https://youtu.be/WRibE2nt8wM?t=1673) What do you do when you-- do you just randomly go home or do you think a little bit? What's relevant? What do you need to know to approach this problem? Yeah.

[28:04](https://youtu.be/WRibE2nt8wM?t=1684) AUDIENCE: You check the weather app.

[28:05](https://youtu.be/WRibE2nt8wM?t=1685) IAN BALL: You check the weather app. And what does that tell you?

[28:08](https://youtu.be/WRibE2nt8wM?t=1688) AUDIENCE: If it's sunny or rainy.

[28:09](https://youtu.be/WRibE2nt8wM?t=1689) IAN BALL: OK. And it might tell you-- and really, it might give you some-- or do you want to go ahead? Yeah.

[28:14](https://youtu.be/WRibE2nt8wM?t=1694) AUDIENCE: It'd be how likely it is.

[28:15](https://youtu.be/WRibE2nt8wM?t=1695) IAN BALL: Yeah. It's going to tell you-- it might say sun or rain, but also it might give you a likelihood of sun or rain. And oftentimes, you're not certain if it's going to be sunny or rainy. You have some uncertainty. That's the key thing that we're having here. So this problem, if you knew the weather, would be very easy. If you knew for certain it was going to be sunny, we know you'd walk home. And if you knew for certain it was going to rain, you'd take the T. The issue is that you have uncertainty. And indeed, when you check the weather app, what you do is you form beliefs. And this is going to be kind of a foundational idea in this course, that in the face of uncertainty, what do you do?

[28:53](https://youtu.be/WRibE2nt8wM?t=1733) You form beliefs.

[28:57](https://youtu.be/WRibE2nt8wM?t=1737) So in game theory, we take a fundamentally Bayesian view of the world. When we have uncertainty, we say, well, we don't know whether it's going to be sunny or rainy. But we might say something like, there's a 20% chance that it's sunny. So we'll assign beliefs. And in this case, let's represent our belief by the probability p that it's sunny. And then correspondingly, we have the probability 1 minus p that it's raining. So these are my beliefs. OK? And now I'm going to fill in some utilities here. I'm going to say 7, 2, 5, 5.

[29:35](https://youtu.be/WRibE2nt8wM?t=1775) So this is a representation of my preferences. And indeed, when it's sunny, I prefer walking to taking the T. And when it's rainy, I prefer taking the T to walking. So now we've written things down. Now I'd ask again. Now we've specified our beliefs, which would you choose? How would you approach this choice?

[29:55](https://youtu.be/WRibE2nt8wM?t=1795) AUDIENCE: Find an expected utility.

[29:56](https://youtu.be/WRibE2nt8wM?t=1796) IAN BALL: So in this course, we're going to assume expected utility. But I want to point out that that's not so obvious. One thing you might say you might say, look, I'm an optimistic person. If I walk, the best thing that can happen is I get 7. If I take the T, the best thing that can happen is I get 5. 7 is higher than 5. I'm going to walk. I'm an optimistic person. You might say, I'm a pessimistic person. I'm really afraid of the worst-case outcome. So when I walk, I could get 2. When I take the T, the worst I can get is 5. That's better, so I'm going to take the T.

[30:27](https://youtu.be/WRibE2nt8wM?t=1827) I think the view in game theory is that optimism and pessimism are better reflected in your beliefs. And we're generally going to assume in this course expected utility.

[30:42](https://youtu.be/WRibE2nt8wM?t=1842) But I want to point out that this is an assumption. There's an axiomatic foundation. So decision theorists think a lot about the question, why should we use expected utility? Or more concretely, if we assume that agents use expected utility, what assumption are we making? What does that mean about their behavior when we assume expected utility? How restrictive is that assumption? And there's an appendix in the lecture notes that's posted online where you can look at this axiomatic approach to expected utility theory. But in this course, we're just going to take it as given that people always use expected utility. So what does that mean?

[31:13](https://youtu.be/WRibE2nt8wM?t=1873) It's two steps. First, they form beliefs.

[31:27](https://youtu.be/WRibE2nt8wM?t=1887) Now, how reasonable is it to think that people form beliefs about the thing they're uncertain about? Well, I think it depends on the context. In this problem, I think it's very reasonable. You open your weather app. It gives you some probability of rain or sun. You might think, oh, I don't really trust the weather app. It always overestimates or underestimates. You might adjust. But you'll form your belief. But if I said, form your belief about what is the probability that in 15 years your income will be between these two numbers. That's pretty hard to form beliefs about. So I think this assumption that you start by forming beliefs about what you don't know is a very realistic assumption in very simple problems.

[32:03](https://youtu.be/WRibE2nt8wM?t=1923) But as we get to very, very rich problems and complicated problems, you might worry that this is not a very well-founded assumption. When we get to complicated choices, maybe your choice about what job to take today depend on your belief about how likely it is that you have a grandchild in 40 years who's short on money and needs the money that you saved based on your job today. Or let's go deeper. What's the probability that in 10 generations, your 10th degree grandchild is going to be short on money? Wow, that's really hard to form beliefs about. So in that context, maybe this is an unrealistic assumption, but it's going to be the assumption we make in this class.

[32:37](https://youtu.be/WRibE2nt8wM?t=1957) And we're going to apply this to models where we think it's a reasonable assumption. And then the next thing that you do, given that you formed your beliefs, is you compute your expected utility from each choice.

[32:55](https://youtu.be/WRibE2nt8wM?t=1975) But now we have a bit of an issue. Because I set up here when I talked about ordinal utility representations that the numbers didn't matter. But now all of a sudden, the numbers will matter. Because when I compute the expectation, my expected utility from walking, if 7 were 700, that's going to change my expected utility. So we have to really crucially understand that in this model, these utilities are different. These are no longer ordinal utilities. So in the decision making under uncertainty-- so in this context, utility is cardinal as opposed to ordinal.

[33:43](https://youtu.be/WRibE2nt8wM?t=2023) Ordinal means only the order matters. That's where the word ordinal comes from. Cardinal means the size or the actual numbers matter. And the way we distinguish this is we say that this utility function here is a Von Neumann-Morgenstern utility function.

[34:12](https://youtu.be/WRibE2nt8wM?t=2052) And basically, throughout this course, we're going to focus on this case of uncertainty. And essentially, all the utility functions we work with are going to be cardinal utility functions or Von Neumann utility functions rather than ordinal utility functions. And just to understand the landscape, up here, we didn't make any expected utility assumptions. This was a very abstract approach to any preference problem. When we're in the world of uncertainty and we restrict attention to expected utility preferences, in that more restricted environment, we're always going to work with what are called Von Neumann-Morgenstern utility functions, where the numerical value actually matters.

[34:51](https://youtu.be/WRibE2nt8wM?t=2091) So let's see how we would approach this problem in this VNM context. So we have two choices. We have walking and taking the T. And what I'm generally going to use is this Von Neumann-Morgenstern utility function u. This is a small u. But then when I compute expected utilities, your expected utility is going to be a big U. So I'm going to now compute here U of walking and U of taken the T. And this is a big U.

[35:22](https://youtu.be/WRibE2nt8wM?t=2122) And of course with handwriting, it's not obvious what it is. So if you're ever unsure, you can ask me. But let's make sure we understand that these functions are defined for different objects. My cardinal utility tells me what is my utility from walking when it's sunny? What is my utility from walking when it's rainy? And it gives me these four numbers. My big U, my expected utility, isn't assigned to a weather-choice pair. It's assigned just to the choice. So for my big U, I can say, what is my expected utility from walking?

[35:57](https://youtu.be/WRibE2nt8wM?t=2157) My little you tells me my utility from walking in each state of the world. So let's just compute this. It's going to be quite easy. The math here is not the point. It's about conceptually. So if I walk, well, with probability p I get 7 and with probability 1 minus p I get 2. So my expected utility is p times 7 plus 1 minus p times 2. And if I take the T, well, it's p times 5 plus 1 minus p times 5.

[36:30](https://youtu.be/WRibE2nt8wM?t=2190) Well, this is easy, right? The expectation of this is just going to be 5. It's 5 either way. So this is 5. And then here what do I get? I get 2. Minus 2p, so plus 5p.

[36:48](https://youtu.be/WRibE2nt8wM?t=2208) So which do I prefer? Well, it depends on p, right? If p is really, really small-- if p gets close to 0, then this becomes 2. And I prefer 5 to 2. So I prefer taking the T. That makes sense. If I look outside and it looks really, really rainy, I'm going to take the T. If p is really, really high-- well, let's say p is 1. I know it's going to be sunny. Then if I walk, I get 7. Yes, if I walk, I get 7. If I take the T, I get 5. 7 is more than 5.

[37:19](https://youtu.be/WRibE2nt8wM?t=2239) So therefore, I prefer to walk. When will I walk rather than take the T? Well, I have to compare these two numbers. So when is 2 plus 5p greater than or equal to 5? Let's ask that question.

[37:37](https://youtu.be/WRibE2nt8wM?t=2257) Well, it's if 5p is greater than or equal to 3. So if p is greater than or equal to 3/5, right? All right? Yeah. I'm sure someone will check the math.

[37:54](https://youtu.be/WRibE2nt8wM?t=2274) Yeah.

[37:56](https://youtu.be/WRibE2nt8wM?t=2276) AUDIENCE: Why is it greater than 3/5?

[37:59](https://youtu.be/WRibE2nt8wM?t=2279) IAN BALL: Ah, good point. So when p is exactly 3/5, I'm indifferent between the two. So I need some word for that. So I guess I would say I weakly prefer walking to taking the T if p is greater than or equal to 3/5. And I strictly prefer walking to taking the T if p is strictly greater than 3/5.

[38:18](https://youtu.be/WRibE2nt8wM?t=2298) AUDIENCE: But wouldn't that make sense that the equal part would be for the T, not for walking?

[38:25](https://youtu.be/WRibE2nt8wM?t=2305) IAN BALL: Sorry, say it again.

[38:27](https://youtu.be/WRibE2nt8wM?t=2307) AUDIENCE: So since it's greater or equal, when this happens, we walk, right?

[38:32](https://youtu.be/WRibE2nt8wM?t=2312) IAN BALL: Exactly. Yeah.

[38:33](https://youtu.be/WRibE2nt8wM?t=2313) AUDIENCE: So why are we putting equal in walking? Is there like-- do we say like when it's equal, it's indifferent? Do we decide which one?

[38:43](https://youtu.be/WRibE2nt8wM?t=2323) IAN BALL: You're right. So there's a bit of ambiguity there. So maybe I'll put-- if this makes it-- let's say this, yeah. I would say, let's focus on the case where p is not 3/5. Then it's pretty clear. When p is exactly 3/5, you're right, we wouldn't have an inherent preference for walking over taking the T. In that case, we would be indifferent between walking and taking the T. So maybe I'll make a note up here and sometimes use a squiggle for that. So we might say W squiggle T, I'm indifferent between walking and taking the T, exactly if p equals 3/5.

[39:17](https://youtu.be/WRibE2nt8wM?t=2357) And in this case, it's a strict preference. So here what this says, which I think is an intuitive model of decision making, that I'm going to choose to walk home when I check my weather app and I think it's sufficiently likely that it's sunny. How sufficiently likely? Well, in this case, it's exactly 60% chance of sun that makes it worth it for me to walk home. Now, where did we get this number from? It depended on the numerical values over here. If these numerical values were different, if I made this 700, well, then I'm basically always going to walk home unless I'm absolutely certain, basically, it's going to rain.

[39:49](https://youtu.be/WRibE2nt8wM?t=2389) So that again shows us that the cardinal values of these utility functions are going to affect the decisions that we make. Any other question on this? Yeah.

[39:58](https://youtu.be/WRibE2nt8wM?t=2398) AUDIENCE: So in the case when it's equal, is there-- like, now we use expected utility. So you can see which expected utility is bigger. But in the case it's equal, is there a rule we should follow? Or based on the numbers we have in the grid--

[40:12](https://youtu.be/WRibE2nt8wM?t=2412) IAN BALL: No, I would say the right way to think about preferences-- I was being a little loose over there to make it quicker. But I'd say really, there's three possibilities when you compare two things. You can strictly prefer one to the other. You can strictly prefer the other to the first one. Or you can be indifferent. There's really three possibilities. And when I compare my expected utilities, there's again three possibilities. Either one is strictly higher, the other is strictly higher, or we're exactly indifferent. And in general, when people are indifferent, it's harder to make predictions about what they'll do. They might randomize. They might choose one. They might choose the other.

[40:43](https://youtu.be/WRibE2nt8wM?t=2443) We're not going to really take a strong stance on what people do in that case. Thanks for clarifying. Yeah. Any other questions? All right. So now let's be a bit more formal. This was a very simple example, but let's formalize this a bit. I guess I have one more board.

[41:07](https://youtu.be/WRibE2nt8wM?t=2467) So maybe I'll say VNM. This is Von Neumann-Morgenstern that I said before. And I'll call this the VNM setting.

[41:17](https://youtu.be/WRibE2nt8wM?t=2477) So generally, this is the approach we're going to take in this course. We're going to start with a really simple example to motivate the theory. We're then going to present a more general abstract model. And then we're going to go back to more concrete and more interesting applications that are more substantive. So that's the three steps we'll take. So what's the setting here? Well, there's going to be a set Z, which we'll say is the-- and I'll call this Z1 through Zm. And this is the set of maybe outcomes or consequences.

[41:49](https://youtu.be/WRibE2nt8wM?t=2509) And we have to be really careful about how we define outcomes and consequences. So in this game that we described, there are actually four outcomes-- I walk home in the sun, I walk home in the rain, I take the T in the sun, and I take the T in the rain. It's important to understand the outcome has to capture everything about my choice. It's not directly my choice. I'm not directly choosing outcomes. I'm choosing effectively lotteries over outcomes. And the outcome has to capture everything that's relevant in the situation. So we have this set of outcomes or consequences.

[42:20](https://youtu.be/WRibE2nt8wM?t=2540) And then we have the Von Neumann-Morgenstern utility function little u. Now, just as we saw over here, little u assigned a number to each of our four outcomes. So U is going to be a function from Z to R. So I'm going to use this function notation a lot in the course. When I put a letter and then a colon, that means this is a function that assigns to every item in this set, every object in this set, something in this set. So in this case, it assigns to every outcome Z is-- I'll write it like this underneath-- what we'll call u of Z. And u of Z is the cardinal Von Neumann-Morgenstern utility I get from the outcome Z.

[43:04](https://youtu.be/WRibE2nt8wM?t=2584) Now, given the set of outcomes and my Von Neumann-Morgenstern utility, we can consider my expected utility. Let me write it this.

[43:20](https://youtu.be/WRibE2nt8wM?t=2600) So my expected utility is the big U. And what is this going to be defined over? It's really crucial that this is defined over a different set. So what do I assign expected utilities to?

[43:32](https://youtu.be/WRibE2nt8wM?t=2612) Yeah.

[43:32](https://youtu.be/WRibE2nt8wM?t=2612) AUDIENCE: Would it be a set of actions?

[43:34](https://youtu.be/WRibE2nt8wM?t=2614) IAN BALL: So in this context, it was a set of actions. But in this abstract model-- good question. Yeah.

[43:40](https://youtu.be/WRibE2nt8wM?t=2620) AUDIENCE: Like the state you're in.

[43:41](https://youtu.be/WRibE2nt8wM?t=2621) IAN BALL: So I would say-- may be a bit of a vague question. I would say the set of lotteries over outcomes. So let's be clear. Here, you're right, we were choosing actions. We chose whether to walk or take the MBTA. But the way we think about those actions in this abstract setting is that each one corresponds to a lottery. Walking is the lottery where with probability p, I walk in the sun, and with probability 1 minus p, I walk in the rain. And similarly, the MBTA is a different lottery where with probability p, I get this outcome, and with probability 1 minus p, I get this outcome.

[44:13](https://youtu.be/WRibE2nt8wM?t=2653) So we need some notation for lotteries. I'm going to use throughout the course delta of Z. I don't want the notation to scare people. It's just going to make it easier for us to talk about things. So Z is our set of outcomes. Delta of Z is the set of lotteries over these outcomes. So in this context where we only had two outcomes, a lottery just specified two numbers, p and 1 minus p. And because those numbers sum to 1, it was really only one number that mattered. Once we knew p, we were able to calculate 1 minus p.

[44:44](https://youtu.be/WRibE2nt8wM?t=2684) But in general, we're looking at a situation where there are m outcomes instead of just two. So what is a lottery?

[44:53](https://youtu.be/WRibE2nt8wM?t=2693) So it contains what I'll call lotteries. And again, I should be clear the use of lottery colloquially is different from in econ. We're not literally talking about buying lottery tickets. We're using this more abstractly. It's a vector p that specifies the probability of each of these outcomes. So p1 to pm. The lottery says with probability p1, you get consequence Z1. With probability pm, you get consequence Zm.

[45:24](https://youtu.be/WRibE2nt8wM?t=2724) These are probabilities. So what has to be true about them? They have to be non-negative. You can't have negative probabilities. And they have to sum to 1. So say p1 up to pm are greater than or equal to 0. And p1 plus pm equals 1.

[45:50](https://youtu.be/WRibE2nt8wM?t=2750) So delta of Z is the set of all of these probability vectors or lotteries that satisfy these properties. And now our expected utility function u is a function from delta Z to R. It says, for any lottery, what is my expected utility from that lottery? So again, using the notation above, we would call a lottery p, and that would go to u of p. So this distinction is crucial, right?

[46:21](https://youtu.be/WRibE2nt8wM?t=2781) A VNM utility says for each outcome, what is my utility from that outcome? And expected utility big U says for each lottery over outcomes, what is my expected utility from that lottery? Yes, question?

[46:34](https://youtu.be/WRibE2nt8wM?t=2794) AUDIENCE: Just to clarify, when we say lotteries, that's what we're modeling your decision as is opting into a specific lottery.

[46:41](https://youtu.be/WRibE2nt8wM?t=2801) IAN BALL: So in the face of uncertainty, we're going to imagine that each choice that you make is going to induce some lottery of outcomes. And we're going to represent the choice by that lottery. This is an important point, because it could be that-- yeah, there's an embedded assumption there. But I'll just say yeah, that's what we're assuming. Yes.

[46:58](https://youtu.be/WRibE2nt8wM?t=2818) AUDIENCE: Can you just say what u would be in this case?

[47:01](https://youtu.be/WRibE2nt8wM?t=2821) IAN BALL: Yes. So let me write the definition and then I'll give you an example. Exactly.

[47:04](https://youtu.be/WRibE2nt8wM?t=2824) AUDIENCE: The little one.

[47:05](https://youtu.be/WRibE2nt8wM?t=2825) IAN BALL: Yeah. Yeah, exactly. So let's first give the general definition. Then I'll go to this example. So u of p equals what? Well, I'm just going to sum over all the possible outcomes. There are m outcomes. So I'm going to take-- and let me not use summation notation, because I think that makes it just seem harder than it is. So let me write p1 u of Z1.

[47:34](https://youtu.be/WRibE2nt8wM?t=2854) So I'm saying with probability p1, I get outcome Z1. And this is my utility. And I sum it all the way up until I get to pn. This is the probability of outcome n. And I get utility this. So if we wanted to really formally write this in the example over here-- so in the example, walking corresponded to a particular lottery, right? Now, if I really want to write out the example, I need to write out all four of these things. So maybe I'll call these Z1, Z2, Z3, Z4.

[48:11](https://youtu.be/WRibE2nt8wM?t=2891) OK? These are the four consequences of my actions. Now, what is walking here? We need to specify four numbers-- p1, p2, p3, p4. What are these four numbers in this example? Well, if I walk, I either get this box with probability p or this box with probability 1 minus p. So of these four numbers, two of them are 0. p3 and p4 are both 0.

[48:44](https://youtu.be/WRibE2nt8wM?t=2924) Why? Because it's impossible for me to get one of those consequences. If I choose to walk home, there's no way-- oh, I got unlucky. I'm now on the T. We're not imagining that can happen. What can happen is you can either walk home when it gets wet or you walk home and you don't. So we have Z1 is going to be p.

[49:05](https://youtu.be/WRibE2nt8wM?t=2945) 1 minus p here. And if we compute it, what is u of this particular p going to be? Well, we go through and we get pu of Z1 plus 1 minus pu of Z2 plus 0 plus 0. I'm not going to include the last two terms because they disappear. And this is going to be exactly what we said before, p times 7 plus 1 minus p times 2.

[49:38](https://youtu.be/WRibE2nt8wM?t=2978) Which we already computed. Any questions on that?

[49:43](https://youtu.be/WRibE2nt8wM?t=2983) Yes.

[49:44](https://youtu.be/WRibE2nt8wM?t=2984) AUDIENCE: So when we're actually getting the value that goes from p to a real number, there's already an implicit choice, like here implicitly we're walking and we're not calculating this particular T.

[49:55](https://youtu.be/WRibE2nt8wM?t=2995) IAN BALL: Yeah. So what I would say is in principle, you can evaluate-- and this is the key distinction between preferences and choices. In principle, you could say, if I were offered a lottery between these outcomes, this lottery versus this lottery, what would I prefer? Those are your preferences. But in reality, you're given a limited menu of lotteries that you can actually choose from. And you're going to make a choice between those lotteries. So I could say in my head, what would I prefer between a stock that always pays a return of 1,000% and a stock that always pays a return of negative 1,000%? I know which one I'd prefer. And I can compute my preferences.

[50:26](https://youtu.be/WRibE2nt8wM?t=3026) But there may not actually be a company listed on the stock market that is going to give me that lottery. In reality, I'm going to face a limited choice set, a limited menu of options. Each of those options will induce a lottery, and I'll make a choice among this smaller collection of lotteries. In this case, that collection had two lotteries, the walking lottery and the taking the T lottery.

[50:43](https://youtu.be/WRibE2nt8wM?t=3043) AUDIENCE: When you're calculating this function, you've already chosen a lottery. Or you calculate this function based off of that.

[50:50](https://youtu.be/WRibE2nt8wM?t=3050) IAN BALL: I would say it's a function, so it's defined for every single lottery. I can say for every single lottery what my expected utility is, even for lotteries that I may not have the option to choose in a given situation. Yes.

[51:02](https://youtu.be/WRibE2nt8wM?t=3062) AUDIENCE: So even though p is determined on our beliefs, when it's a function of big U, do we just take it as exogenous and optimize based off of the things that we can control?

[51:12](https://youtu.be/WRibE2nt8wM?t=3072) IAN BALL: Good question. So this is going to be a big difference between individual decision making and interactive decision making. In individual decision making, all of this uncertainty is just going to be exogenous. And that's the model I've written down. But we'll see when we get to interactive decision making, well, how do I form beliefs about the choices that other people make? And then we're going to have to impose more structure on these beliefs. But for the individual context, we think of them as exogenous. Through introspection, you form your beliefs about what nature does, not what other players do. Yep. Thank you. OK, so one final thing in the last five minutes.

[51:42](https://youtu.be/WRibE2nt8wM?t=3102) There's a special case, which is of a lot of importance. And I'll say the key special case is going to be what we call money lotteries.

[52:00](https://youtu.be/WRibE2nt8wM?t=3120) So these are more like the lotteries that you hear about in reality. And why do I call it a special case? Well, it's the special case where Z, the set of outcomes, is just a set of monetary rewards. So I'll say Z is equal to R. And this means the real number line. Now, before I said Z was finite. Here I'm going to be a little loose and allow Z to be infinite. So the outcome here is I get \$1, I get \$10, I get a billion dollars. These are all outcomes. So what is the lottery? Let's consider two lotteries.

[52:31](https://youtu.be/WRibE2nt8wM?t=3151) One lottery maybe pays \$10.

[52:55](https://youtu.be/WRibE2nt8wM?t=3175) So let's now consider two lotteries. Now here, with infinitely many things, it could be more complicated. But I'm going to look at lotteries where there's a small number of things that could actually happen. So under this lottery, there's only two outcomes that could happen. Every other outcome has probability 0. Under this lottery, with probability 0.99, I get \$10. And with probability 0.01, I get nothing. I get 0. With this lottery, I get \$1,000 with probability 0.01 and I get 0 with probability 0.99.

[53:26](https://youtu.be/WRibE2nt8wM?t=3206) So two different lotteries. They fit within our framework. And we can compute our expected utility of these lotteries. How? Well, to do that, we need our VNM utility function, which is a function, remember from Z to R. But in this case, Z is just R. But let's understand, this is money and this is utility.

[53:53](https://youtu.be/WRibE2nt8wM?t=3233) So which of these lotteries would you prefer? There's no right answer. Again, preferences are not irrational. But what's your preference between these?

[54:06](https://youtu.be/WRibE2nt8wM?t=3246) Yeah.

[54:07](https://youtu.be/WRibE2nt8wM?t=3247) AUDIENCE: I like the first one.

[54:08](https://youtu.be/WRibE2nt8wM?t=3248) IAN BALL: You like the first one? OK. I think most people would. And why is that?

[54:12](https://youtu.be/WRibE2nt8wM?t=3252) AUDIENCE: Because I don't know, it seems like 1,000 divided by whatever seems to be about 10, and then it feels more certain because it's 0.99. But I don't know.

[54:25](https://youtu.be/WRibE2nt8wM?t=3265) IAN BALL: Right. So that's a great way of thinking about it. So the first thing you might say is let's compare the expected values of these lotteries. The expected value of this is exactly \$10. 100th of 1,000 has expected value \$10. What's the expected value of this? It's \$9.90. So this has a lower monetary expected value. But despite it having a lower monetary expected value, it might have a higher expected utility. And many people would choose this lottery over this lottery exactly because of what you said, it's safer.

[54:57](https://youtu.be/WRibE2nt8wM?t=3297) So what's key is through our Von Neumann-Morgenstern utility function u, when we evaluate lotteries, we don't just compare the expected monetary amount of the lotteries, we compare the expected utility between the lotteries. And what you're describing is what a lot of people feature in this context is, I'll say often, u is a concave function. So if I were to draw u for a lot of people, their u is something like this.

[55:29](https://youtu.be/WRibE2nt8wM?t=3329) And when u is concave, it means people are what we call risk averse.

[55:37](https://youtu.be/WRibE2nt8wM?t=3337) So in this case, you exhibited risk aversion. You preferred this lottery to this lottery, even though it had a smaller expected value. And if we want to formally define what does risk aversion mean, a risk averse person is someone who always prefers a certainty at the expected value of a lottery to the lottery itself. So let me just write this mathematically and then explain it.

[56:12](https://youtu.be/WRibE2nt8wM?t=3372) Someone is risk averse if deciding between a lottery p-- let's take this as an example. This is one lottery p. And another lottery that pays the expected value of p with certainty. In this case, that would be exactly \$10. Someone who's risk averse would always prefer getting \$10 for sure to getting a lottery whose expected monetary amount was \$10. And mathematically, that's captured by the concavity of the utility function here, u. For this to be a strict preference, we need p to be non-degenerate.

[56:43](https://youtu.be/WRibE2nt8wM?t=3403) 1p would be you get something with probability 1. So as long as it's non-degenerate, this is the meaning of risk aversion. And this kind of answers a puzzle, I mean, going back to the 1700s where people said, why would you choose this lottery over this one? This one has a higher expected value. And the answer is, well, your utility function is not the same as money. You don't maximize expected money, you maximize expected utility. And some people's utilities can be concave like this. So let me stop there and I will see everyone next week.
