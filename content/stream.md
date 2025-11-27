---
date: "2025-10-24"
description: a microblog within a blog, at the middle of the night.
id: stream
metadata:
  ebnf: |-
    stream        = block , { separator , block } , [ separator ] ;
    separator     = newline , ( "---" | heading ) , newline ;
    heading       = "##" , ws , title ;
    block         = newline, meta_block , newline, body ;
    meta_block    = "- [meta]:" , newline , meta_entry , { meta_entry } ;
    meta_entry    = ws , "- date:" , ws , timestamp
                  | ws , "- tags:" , newline , tag_item , { tag_item }
                  | ws , "- " , key , ":" , ws , value ;
    tag_item      = ws , ws , "- " , tag ;
    body          = markdown_line , { newline , markdown_line } ;
    markdown_line = text_line | quote_line | list_line | embed_line | image_line ;
    quote_line    = ">" , ws , text_line ;
    list_line     = "- " , text_line ;
    embed_line    = "![[" , text , "]]" ;
    image_line    = "![" , { character - "]" } , "](" , uri , ")" ;
    timestamp     = year , "-" , month , "-" , day , ws , hour , ":" , minute , ":" , second , ws , "GMT" , offset ;
    offset        = ("+" | "-") , hour , ":" , minute ;
    tag           = identifier ;
    key           = identifier ;
    value         = text ;
    title         = text ;
    text_line     = { character - newline } ;
    text          = { character - newline } ;
    identifier    = letter , { letter | digit | "-" } ;
    year          = digit , digit , digit , digit ;
    month         = digit , digit ;
    day           = digit , digit ;
    hour          = digit , digit ;
    minute        = digit , digit ;
    second        = digit , digit ;
    ws            = { " " } ;
    letter        = "a".."z" ;
    digit         = "0".."9" ;
    character     = ? any printable ascii except newline ? ;
modified: 2025-11-27 16:27:22 GMT-05:00
tags:
  - fruit
  - evergreen
title: stream
---

- [meta]:
  - date: 2025-11-27 15:57:58 GMT-05:00
  - tags:
    - alignment

Anthropic found that natural emergent misalignment stems from [_reward hacking_](https://www.anthropic.com/research/emergent-misalignment-reward-hacking). Though, I suspect that ablating these "bad behaviour" wouldn't necessary make the model more aligned. What if having certain malicious intent is actually helpful?

---

- [meta]:
  - date: 2025-11-25 13:15:26 GMT-05:00
  - tags:
    - love
    - emotions

there's a towel i used for when L stayed over. still haven't washed it. keeping it means keeping the rot—letting those feelings decay until the whole thing becomes unbearable enough that throwing it away becomes a necessity rather than being a choice.

---

- [meta]:
  - date: 2025-11-22 14:18:52 GMT-05:00
  - tags:
    - life
    - relationship

I got a cup of hot chocolate today. It reminded me of L, and somehow, we carry fragments of them within you without knowing so.

---

- [meta]:
  - date: 2025-11-21 12:43:25 GMT-05:00
  - tags:
    - relationship

I'm have no desire of making new friends in Toronto anymore. Everything felt so superficial here.

---

- [meta]:
  - date: 2025-11-20 13:07:46 GMT-05:00
  - tags:
    - love

```quotes
To find someone worth fighting for is a beautiful thing.

James, _[All At Once](https://jameslin.bio/jolie)_
```

---

- [meta]:
  - date: 2025-11-17 14:55:06 GMT-05:00
  - tags:
    - love

![[quotes#^hopeless]]

---

- [meta]:
  - date: 2025-11-15 12:36:29 GMT-05:00
  - tags:
    - life

Got on a call with middle school friends, seeing my old Vietnamese teacher, B—there's warmth in it. Good memories, honestly. But I'm glad I left when I did.

---

- [meta]:
  - date: 2025-11-15 01:58:17 GMT-05:00
  - tags:
    - love

Why does this hurts so bad? We aren't even in a relationship.

---

- [meta]:
  - date: 2025-11-13 11:33:37 GMT-05:00
  - tags:
    - love

I'm completely broken down, seeing the physical letter L left on the counter. I'm loss for words.

---

- [meta]:
  - date: 2025-11-13 05:44:12 GMT-05:00
  - tags:
    - love
    - feeling

Love sucks. love hurts. love prevails. I want to be loved. and yet, all I got are heartbreaks. I'm once again, perplexed by my own feelings, a sense of _saudade_

---

- [meta]:
  - date: 2025-11-13 01:08:39 GMT-05:00
  - tags:
    - love

```quotes
It's possible to love men without rage. There are thousands of ways to love men.

Lidia Yuknavitch, The Chronology of Water
```

## on suicide

- [meta]:
  - date: 2025-11-13 01:01:11 GMT-05:00
  - tags:
    - philosophy
    - death

I was reading [On suicide](https://docs.google.com/document/d/14XZJtJcMGzD4ZY6AgaTzobunndRvXZMGUybA6JIQOj4/edit?tab=t.0) by [Alexey Guzey](https://guzey.com/)

> When I was in high school, I spent a year trying to kill myself. I just couldn't do it. At some point I decided to make my life as bad as I possibly could, but nothing worked. No matter how much I tried, I still ==wanted to be alive more== than I [_wanted to be dead_].

The [[thoughts/ethics#deontology|deontological]] arguments of life ::presupposes the feeling of interconnectedness{h5}::. I think that if one acknowledges that no one wants one in life, then one might conclude that one's life is not worth living. [[thoughts/Philosophy and Kant|Kant]] would then argue against the notion of "kill oneself" given that it is an act in violoation of the [[thoughts/moral|moral]] law, and deemed it as **wrong**.

> I wonder how much of this is due to the lack of imagination. If you're suicidal, it's very difficult to imagine life getting better.

You're depressed, sure, so your cognition is impaired, such that you realised life isn't worth living anymore, which then concludes that suicide is the real solution. So your mind convinced yourself to just **gave in**

```quotes
The impulse to end life and the impulse to further life contradict each other.

Kant, Groundwork _§4:422_
```

But I'm troubled by how close this gets to "just shake it off" or "try something new!" as if suicidal depression were boredom with your routine. The difference, maybe, is that the conventional advice assumes you can white-knuckle through by force of [[thoughts/Will|will]] or positive thinking.

Guzey's argument focuses on a darker path, where you can't think your way out, and you probably can't will your way out either. You need an external disruption large enough to short-circuit the entire system. Which is less hopeful than it sounds—e.g: you're not in control of your own recovery. You're waiting for something to happen that breaks the pattern, or you're throwing yourself at random experiences hoping one of them will.

The wanting to live that exists underneath reasons, that persists even when you've eliminated every reason you can name. Whether that's the body asserting itself, or some pre-rational attachment to [[thoughts/being|being]], or just biochemistry doing its thing.

What brought me back wasn't a good argument for living. It was N's voice, L's conversations, the way light looked one morning, the prospect of missing something I hadn't experienced yet. The stuff that doesn't make sense in the ledger of reasons but somehow weighs more than the whole column of evidence against continuing.

I think, subconsciously, we all have a moral duty to continue to {{sidenotes[live,]: go outside, look at trees, eat an onion sandwich, buy some sourdough}} and only when you have exhausted all possible solutions, then suicide is the last reasonable solutions to end the absurd life.

I'm once again, thinking about suicide.


---

- [meta]:
  - date: 2025-11-12 21:33:51 GMT-05:00
  - tags:
    - life

```quotes
But Kurt Vonnegut writes about the difference between two kinds of teams. A granfalloon is a team of people pushed together for some ordinary human purpose, like learning medicine or running a hospital psychiatry department. They may get to know each other well. They may like each other. But in the end, the purpose will be achieved, and they’ll go their separate ways.

Scott Alexander, [To the Great City](https://perma.cc/G5UP-PD2N)
```

---

- [meta]:
  - date: 2025-11-10 04:30:10 GMT-05:00
  - tags:
    - pattern

> Each #pattern describes a problem which occurs over and over again in our environment, and then describes the core of the solution to that problem, in such a way that you can use the solution a million times over, without ever doing it the same way.
>
> —[[library/A Pattern Language]], p. x

---

- [meta]:
  - date: 2025-11-09 01:34:25 GMT-05:00
  - tags:
    - technical

I've been thinking about connectionist networks lately (or for the past two years or so), and there's something deeply unsettling about how we talk about them. Not unsettling in a bad way—more like that productive discomfort you get when you realize the categories you've been using don't quite map onto reality.

The whole connectionist project started as a rejection, really. A rejection of the idea that intelligence is symbol manipulation all the way down. Back in 1986, when Rumelhart and McClelland dropped their PDP volumes [@rumelhart1986parallel], they weren't just proposing a new computational architecture—they were making an ontological claim about what cognition _is_.

![[thoughts/Connectionist network#{collapsed: true}]]

---

- [meta]:
  - date: 2025-11-05 05:48:09 GMT-05:00
  - tags:
    - fruit

My journal is my [[posts/index|blog]]—not because I want to become a blogger, but because it’s a permanent state of [[thoughts/Eldrich horror|eldritch horror]] etched into [[thoughts/LLMs|GPT‑X]]’s compressed mind, all about *myself*. Til the day I’m plugged into the mainframe, it’s as if I never left.

---

- [meta]:
  - date: 2025-11-04 21:11:21 GMT-05:00
  - tags:
    - life

I find myself using my mechanical keyboards less and less nowadays, using my laptop keyboard instead. This might have to do with the mode of focus the laptop keyboard puts me into—something about focusing on the work itself rather than the tools. Partially because of the wrist pain from long sessions of working at my desk 🐙

---

- [meta]:
  - date: 2025-10-31 18:24:43 GMT-04:00
  - tags:
    - life

Writing kernels sounds way more fun the whoring on the streets of Toronto. Happy Halloween 🎃 though.

---

- [meta]:
  - date: 2025-10-30 03:18:38 GMT-04:00
  - tags:
    - writing

```quotes
Writing essays, at its best, is a way of discovering ideas.

An essay should ordinarily start with what I'm going to call a question, though I mean this in a very general sense: it doesn't have to be a question grammatically, just something that acts like one in the sense that it {{sidenotes[spurs some response.]: When you find yourself very curious about an apparently minor question, that's an exciting sign. Evolution has designed you to pay attention to things that matter. So when you're very curious about something random, that could mean you've unconsciously noticed it's less random than it seems.}}

Paul Graham, [The Best Essay](https://paulgraham.com/best.html)
```

![[thoughts/writing#as a journey for exploration]]

---

- [meta]:
  - date: 2025-10-29 05:18:28 GMT-04:00
  - tags:
    - productivity
    - philosophy

Late night work listening to Dreyfus' lectures hits like smoking a good joint on a Friday night.

![[https://youtu.be/usxvyf3xqcQ?si=wrria3i7tSqYvGQk]]

---

- [meta]:
  - date: 2025-10-28 16:38:29 GMT-04:00
  - tags:
    - math

```poem
Try as you may,

you just can't get away,

from mathematics

—Tom Lehrer
```

---

- [meta]:
  - date: 2025-10-28 16:15:23 GMT-04:00
  - tags:
    - writing

```quotes
The reason I've spent so long establishing this rather obvious point <dfn>[that [[thoughts/writing]] helps you refine your thinking]</dfn> is that it leads to another that many people will find shocking. If writing down your ideas always makes them more precise and more complete, then no one who hasn't written about a topic has fully formed ideas about it. And someone who never writes has no fully formed ideas about anything nontrivial.

It feels to them as if they do, especially if they're not in the habit of critically examining their own thinking. Ideas can feel complete. ==It's only when you try to put them into words that you discover they're not==. So if you never subject your ideas to that test, you'll not only never have fully formed ideas, but also never realize it.

Paul Graham, [Putting Ideas into Words](https://paulgraham.com/words.html)
```

---

- [meta]:
  - date: 2025-10-27 15:01:07 GMT-04:00
  - tags:
    - life

```quotes
I wanted to eat life by the mouthful, to devour it, to be swallowed up in its dizzying vertigo, to be both actor and spectator, to possess and be possessed, to discover and to create, to make of my life a work of art.

Simone de Beauvoir, _Memoirs of a Dutiful Daughter (1958)_
```

---

- [meta]:
  - date: 2025-10-27 09:34:45 GMT-04:00
  - tags:
    - fruit

![[quotes#^violent]]

---

- [meta]:
  - date: 2025-10-27 07:37:15 GMT-04:00
  - tags:
    - journal

I find myself the most productive while procrastinating other tasks.

---

- [meta]:
  - date: 2025-10-26 01:41:59 GMT-04:00
  - tags:
    - death
    - philosophy

[[thoughts/Camus]] begins with suicide in [[library/The Myth of Sisyphus]], more or less a demonstration of how absurd life really is. He used suicide as a scape goat of people who don't have enough courage to deal with the hardship of life being thrown at them, and considered suicide as cowardice.

I have exercised this thought _many times_, in the way one considers a standing appointment. Not with desire of actually doing the deeds or the fear of death, but rather a procedure to keep my mind sharp. The world offers no meaning, yet I require meaning, therefore, it doesn't seem reasonable to be alive.

If I could leave, what makes me stay? When I actually sit with this question, as a real possibility, the answers surprise me. I want to see how this conversation with N resolves on Thursday. I'm still waiting for L to send that letter my way. There's a problem on my desk I'm halfway through understanding, and I need to know if my intuition about it is right. I think about my parents getting the phone call, my friends having to sort through my things, and something in me recoils not from death but from inflicting that particular grief.

Turns out the results are mostly attachments. They are somewhat very small, specific to my life, but accumulate gradually.

I watch people in coffee shops, on trains, in office buildings. They seem to continue, mindlessly. Felt likes aimless drones just letting time pass through their body. Pour a cup of coffee, send that emails, making plan for next week. Perhaps they have solved something I have not. Perhaps they have simply never filed the paperwork for the question. It is possible everyone considers this and we have agreed, collectively, not to mention it. Like a standing meeting no one enjoys but everyone attends.

My work is not that special, my contributions to the collectives projects are temporary. Life won't change a lot if I disappear. Heat death will erase everything eventually. Still, I'm here though. Maybe because of a particular arrangement of attention and time, maybe because I still care enough, maybe the thought of striving another day for N, parents, L, J, C, S are good enough of a motivators to keep ones going.

> Meaning does not require permanence. This seems important but I am not certain why.

[[thoughts/Camus]]'s revolt: Continuing with full knowledge of the absurd. Choosing again each day, not from habit but from decision. I am not certain I achieve this. Most days feel like habit.

The question becomes dangerous when it stops being theoretical. When it moves from mind to body. When the weight becomes physical rather than philosophical. Then one must interrupt the process. Call someone. Leave the room. The distinction is administrative: one is philosophy, the other is emergency. Emergency requires different procedures entirely.

## Why Superhuman AI Would Kill Us All

- [meta]:
  - date: 2025-10-25 23:26:00 GMT-04:00
  - tags:
    - llm

--[Eliezer Yudkowsky](https://www.youtube.com/watch?v=nRvAt4H7d7E)

Yudkowsky's full argument on eschatology isn't productive whatsoever. Feels a lot more science fiction writing, where he claims that this systems will end up "wanting to do their stuff without wanting to take the pills that [we offer to] makes them to do stuff that we _wants_ them to do instead."

Yudkowsky's claim relies on [[thoughts/AGI|superintelligent]] optimizer + misaligned goals = human extinction. Not because it hates us. Because we're made of atoms it can use for something else. Nanotech grey goo. Designer pathogens. Trees that grow computer chips. All physically possible, therefore inevitable once you have sufficient optimization power.

But intelligence in actual systems is jagged, domain-specific, constrained by architecture. AlphaFold is brilliant at protein folding and useless at everything else. [[thoughts/LLMs|GPT-4]] can write code but can't reliably count letters. The jump from "good at prediction" to "can design novel molecular machinery from first principles" assumes transfer learning we haven't seen. These very much resembles the old GOFAI vs NFAI arguments. Maybe the argument here is to have a _composition of multiple domain-specific superintelligence systems_ that amplify our life.

The "foom" scenario requires explosive recursive self-improvement, which is obtuse. GPT-6 builds GPT-7, capabilities doubling weekly until godlike intelligence. Architecturally speaking, maybe we figure out something that scales with attention, but it has to be beyond just Transformers, maybe in conjunction with something like JEPA. The argument assumes breakthroughs on demand.

He did mention about a recursive needs of hydrogen, but fwiw physical constraints matters a lot more here. Building nanoassemblers needs: labs, materials, energy, time for experiments. Biology took billions of years of parallel search to reach cells. You can speed that up with intelligence – how much? The argument assumes "enough."

The frame requires that one assume worst case at every branch, assume maximum capability, assume minimal constraints, therefore _doom_. I just don't think that's not how you build things. Real systems fail in boring ways. Scaling laws break. Architectures saturate. The chain from "AI breaks up marriages" to "superintelligence converts biosphere to computronium" requires assumptions that would be rejected in any engineering domain.

> don't build, don't experiment, don't iterate, because any mistake might be the last. That's **not** how we've solved any complex safety problem. Treating everyone who continues working as equivalent to cigarette executives isn't engaging with technical disagreements.

[[thoughts/Alignment]] is hard. I do think that capabilities scales faster than safety. But the response can't be "stop everything and hope treaties hold." It has to be: build better systems, understand current systems deeply, develop alignment that might work. You need feedback loops. You need to learn from failures at scales where failure isn't extinction.

## how we talk about god

- [meta]:
  - date: 2025-10-24 01:23:00 GMT-04:00
  - tags:
    - theology
    - life

"God exists" – we say it like we're saying "the table exists." Same grammatical structure, same verb, same declaration of being. But [[thoughts/Wittgenstein|Wittgensteinians]] would tell you these are entirely different language games. Different forms of life enacted through identical syntax.

When someone tells me "I feel God's presence," I wonder what they're actually describing. Not the [[thoughts/Metaphysics|metaphysical]] claim – that's the easy dismissal. But the phenomenology itself. The feeling of being held when no one's holding you. Of mattering when the universe gives no indication you do. Of not being alone in the dark when, empirically, you are.

I've had these feelings. Walking home at 3 AM and suddenly feeling like the street lights are watching over me. Finding a book at exactly the moment I needed it. That uncanny sense of alignment, of pattern, of something speaking directly to you through the noise of existence.

Theism has this ready explanation: providence, divine orchestration, God moving through your life. Clean. Comforting. Complete.

Strip that away and you're left with something harder to name. Not "coincidence" – that's too dismissive of the actual experience. But this persistent feeling that events _mean_ something. Your pattern-matching brain encountering random noise and insisting – _insisting_ – there's signal there.

"God is good" sounds like "water is wet" – a property of an existing thing. But maybe it's closer to "justice is sacred." Not a description but an orientation. A way of being in the world, encoded in grammatical structures that trick you into thinking you're making claims about reality when you're actually describing how you move through it.

Kierkegaard said faith requires [[thoughts/Camus|absurdity]] – you leap beyond reason into belief. But we're all leaping anyway. Every morning. Into relationships we know will end. Into projects that will be forgotten. Into futures that terminate, always, in death.

The absurdity isn't believing without evidence. The absurdity is pouring coffee at 7 AM when you know – really know, not just intellectually but viscerally – that entropy wins. That nothing persists. That the heat death of the universe is coming for everything you've ever loved.

And yet here we are. Making breakfast. Answering emails. Planning for Tuesday.

This isn't Kierkegaardian faith – no transcendent leap into divine arms. But there's still a leap. Acting _as if_ things matter despite having no cosmic guarantee they do. Building despite impermanence. Loving despite loss. Creating despite destruction.

Maybe religious language expresses this fundamental human insistence on continuing. On treating temporary patterns as if they're eternal truths. On finding meaning in noise because meaning-making is what conscious matter does.

The [[/posts/feelings|feeling]] of being held, of mattering, of not being alone – these persist even after you've dismantled the theological framework. They're built into consciousness itself. This need for connection, for significance, for something beyond the merely material. Not because there's a God responding, but because you're the kind of system that generates these feelings when confronting its own existence.

Strip away providence and you're left with pattern-matching systems encountering randomness and desperately trying to read it as text. As message. As meaning.

We can't help it. It's how we're built.

The grammar of theism gives us vocabulary for these experiences. But the experiences themselves? They're what happens when consciousness encounters itself in a universe that doesn't care. When the meaning-making machine meets the meaningless void and keeps making meaning anyway.

Not despite the void. Through it. Because of it.

Generation without guarantee. Creation without cosmic permission. Love without eternal preservation.

---

- [meta]:
  - date: 2025-10-25 16:50:00 GMT-04:00
  - tags:
    - fruit

The visceral feeling of returning to a place you once lived. Those familiar streets you'd walk daily without really noticing. The tiny brick that used to catch your shoe—gone now. New shops sprouting up, the decommissioned church turned taco spot. The coffee shop with its familiar faces, where you just walk by and smile at each other, no words needed. That bar you used to haunt.

What gets me is seeing the old faces, the same constellation of people I used to orbit. Five months in my new Toronto apartment (which I love, truly), but it feels like years have passed. Time does this strange thing – warps around you when you leave. For me, whole epochs have unfolded. For them, Tuesday follows Monday follows Sunday. The uncanny valley of temporal perception: you've moved through time at different speeds, yet here you both are, occupying the same present moment.
