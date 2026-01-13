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
modified: 2026-01-12 17:42:06 GMT-05:00
tags:
  - fruit
  - evergreen
title: stream
---

## cat and the state of homeostasis

- [meta]:
  - date: 2026-01-12 17:41:48 GMT-05:00
  - tags:
    - life

---

- [meta]:
  - date: 2026-01-11 10:17:56 GMT-05:00
  - tags:
    - agents

I mean we all heard about [RLM](https://alexzhang13.github.io/blog/2025/rlm/), but path to [[thoughts/AGI]] is to use recursively many large models, i.e Kimi + Qwen3 + MiniMax + Gemini 3 Pro + ChatGPT 5.2 Pro + Claude 4.5 Opus + Claude 4.5 Haiku. Subagent et al. and all that

![[thoughts/images/elmo-fire.gif|just use them all.]]

_update: oops, it seems like Nathan just released a [post](https://www.interconnects.ai/p/use-multiple-models) about this ahaha_

---

- [meta]:
  - date: 2026-01-11 07:53:01 GMT-05:00
  - tags:
    - moral

I'm infatuated by the root of {{sidenotes[all evil]: I just finished [[movies/Nuremberg]] by Vanderbilt, where they dramatized the historical Nuremberg trials, through the lens of US psychatrists Dr. Douglas Kelley.}}, as supposed to the root of happiness. I say happiness here instead of good because I would argue universal good is intrinsic towards human nature, and how they became to be (i.e in the case of Nazi) has nothing to do with this innate good. The degree to which goodness is portrayed is then driven by how happy they are in life. A few line of questioning, circa [[library/Beyond Good and Evil|BGE]]:
- is "universal good" claim a synthetic a priori judgment, an empirical generalization, or a regulative ideal, given [[thoughts/Philosophy and Kant|Kant's]] limits on metaphysical knowledge in [[library/The Critique of Pure Reason|CPR]]? cf. BGE §11
- if i treat good as intrinsic to human nature, am i covertly positing a thing in itself about the will that CPR says i cannot know? cf. BGE §2
- does BGE's suspicion of dogmatism imply that "universal good" is just a refined prejudice, a metaphysical comfort smuggled in under psychological terms? cf. BGE §1
- how does BGE's master and slave morality distinction reorder my good and evil axis, am i confusing good bad (noble vs base) with good evil (altruism vs harm)? cf. BGE §260
- if values track types and rank, what would count as a universal good that does not erase difference, and is that coherent on [[thoughts/Philosophy and Nietzsche|Nietzsche]]'s anti realist view of value? cf. BGE §257
- can Kant's universality test justify a universal good without presupposing a substantive end, or does any substantive end slide into heteronomy? cf. BGE §6
- if reason seeks the unconditioned and produces antinomies in CPR, is "universal good" better treated as a regulative ideal rather than a constitutive truth claim? cf. BGE §3
- if goodness scales with happiness in my claim, what ratio of happiness to goodness am i assuming, and does that ratio survive Nietzsche's herd morality critique? cf. BGE §202

_ig it is time to revist BGE argument once again_

---

## accumulation of jagged taste

- [meta]:
  - date: 2026-01-07 06:50:11 GMT-05:00
  - tags:
    - philosophy
    - pattern

I usually go to school relatively early nowadays, given that commute time is sub-optimal for any given {{sidenotes[day]: i.e. me living in Toronto going to school in Hamilton}}. I'm reading Ramsey's [Truth and Probability](https://fitelson.org/probability/ramsey.pdf) while waiting for the compiler to finish. On p.176 he wrote:

```quotes
i am at a cross-roads and do not know the way; but i rather think one of the two ways is right. i propose therefore to go that way but keep my eyes open for someone to ask; if now i see someone half a mile away over the fields, whether i turn aside to ask him will depend on the relative inconvenience of going out of my way to cross the fields or of continuing on the wrong road if it is the wrong road.

but it will also depend on how confident i am that i am right; and clearly the more confident i am of this the less distance i should be willing to go from the road to check my opinion. i propose therefore to use the distance i would be prepared to go to ask, as a measure of the confidence of my opinion.

Ramsey, _Truth and Probability (p. 176)_
```

Ramsey then formalized with _the degree of belief_ {{sidenotes[to be]: here the disadvantages of going $x$ yards is $f(x)$, advantage of arriving right $r$, arriving wrong is $w$, and you are willing to go distance $d$ to ask}} $p = 1 - \frac{f(d)}{(r-w)}$:

> By proposing a bet on $p$ we give the subject a possible course of action from which so much extra good will result to him if $p$ is true and so much extra bad if $p$ is false

By this logic Ramsey implied that the distance you'd walk to verify a statement IS the belief, i.e. the willingness to stake time and cross a field and possibly be wrong later. He then followed with on p.183:

> Whenever we go to the station we are betting that a train will really run, and if we had not a sufficient degree of belief in this we should decline the bet and stay at home. The options God gives us are always conditional on our guessing whether a certain proposition is true. Secondly, it is based throughout on the idea of mathematical expectation; the dissatisfaction often felt with this idea is due mainly to the inaccurate measurement of goods.

I first encountered Ramsey's work while reading [Galvin's entry](https://www.gleech.org/frank) and got curious about his contributions to economics. Ramsey was 22 when he wrote T&P, four years before his death. Ramsey was a close friend of [[thoughts/Wittgenstein|Ludwig Wittgenstein]] and instrumental towards convincing Wittgenstein to return to Cambridge for #philosophy. Ramsey is loved by many, in the way that people would call him Frank, where we would never call Wittgenstein "Ludwig".

```quotes
I don’t feel the least humble before the vastness of the heavens. The stars may be large, but they cannot think or love; and these are qualities which impress me far more than size does…

My picture of the world is drawn in perspective, and not like a model to scale. The foreground is occupied by human beings and the stars are all as small as threepenny bits… In time the world will cool and everything will die; but that is a long time off still… Nor is the present less valuable because the future will be blank.

Humanity, which fills the foreground of my picture, I find interesting and on the whole admirable. I find, just now at least, the world a pleasant and exciting place. You may find it depressing; I am sorry for you, and you despise me. But I have reason and you have none; you would only have a reason for despising me if your feeling corresponded to the fact in a way mine didn’t. But neither can correspond to the fact. The fact is not in itself good or bad; it is just that it thrills me but depresses you. On the other hand, I pity you with reason, because it is pleasanter to be thrilled than to be depressed, and not merely pleasanter but better for all one’s activities.

Ramsey, _aged 22_
```

Ramsey's crossroads metaphor emphasizes the pattern of bets IS the belief you create towards a _gut feeling_, _conviction of sort_. A thing that you deem to be beautiful predicates upon a set of bets that you consider aesthetically pleasing. For example, I would consider Rembrandt to be the best painters because his drawing speaks towards the absurdism as the facticity of life. In a way that Rembrandt individuated my liking towards more expressionist painters, such as Egon Schiele, Oskar Kokoschka, etc. The way that we take bets in aesthetics has the same form as taking bets elsewhere. One would live with consequences of choosing a direction, willingly sacrifice the alternatives knowing that such commitments will eventually [[thoughts/emergent behaviour|formulate]] your own [[thoughts/taste]].

[———]

So why jagged? The term actually came up in a recent conversation with my therapist. we were talking about living in a {{sidenotes[hyperabundance]: i.e post-scarcity, unprecedented [[posts/hyperabundance|interconnectedness]], post-singularity to an extent}} world, and i found myself circling the same worry, a la the tension between filtering and creating. [JZ writes](https://jzhao.xyz/posts/aesthetics-and-taste) that we need to "do things without recipes more often." this tracks with Ramsey's formulation of bets.

Such grief arose, yet again, whilst i was fine tuning a [Qwen-based ghost-of-mine](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8). there's this moment when the loss curve suddenly drops, .i.e. _grokking_, and the model starts enforcing traits it learnt in my writing. watching this felt strange. my style is still evolving, still unfinished, yet here i was compressing a snapshot of myself into a [[thoughts/Eldrich horror|Lovecraftian horror]] of my own shortcomings. the thing it kept turning to: _i'm not able to write like Camus yet. i write Camus-adjacent. but i don't want to write like Camus. i need to find something Aaron-shaped._

I found this fascinating, as I realized the gap started to feel unbridgeable. on one side holds my rigorous sense of what i like. yet, here, on the other side resides the curation speed, how fast i can filter so the _ghost_ generates queries that actually emulate how i think. (Or I have one better for you, the ghost can also do the filtering as well!!)

[———]

the main problem I observe with this approach, at large over-used of [[thoughts/LLMs|LLMs]] in general, is that one will never have to walk across any _fields_ to verify anything, if one actually like Camus. Ramsey assumes distance has some intrinsic cost. _but what happens when_ $f(d) \to 0 \forall d \in \mathbb{N}^{*}$? what happens when you can sample every possible aesthetic direction simultaneously, and a Camus-adjacent or a Dostoevsky-adjacent, or a Bakhtin-adjacent just, _poof, appear out of thin air_? In this case, the degree of belief then collapsed into a _preference-simulation_ instead of subjective conviction.

the jaggedness comes from _having to commit_. Dostoevsky couldn't just pivot to write like Turgenev when Notes from Underground wasn't landing like his previous work. He had to tunnel deeper into his own neurotic verbal tics, with run-on sentences that felt like you are traversing a vast landscape, so that the _underground man who won't shut up because shutting up means losing the only voice he possess_ can live on perpetually. Bakhtin's polyphony emerged from the made-up fantasy of Soviet's society (where socialism is more of an idea rather than practical decisions), having to encode his own philosophical dialogue disguised inside {{sidenotes[literacy criticism]: otherwise he would probably get shot or capture by the secret police}} since that is the only [[thoughts/forms of life|form]] it can exist.

As such, the jaggedness stems from this **individuated friction**, because we need friction to formulate such shape. When you are essentially time-boxed and constrained at which books you can read in the local library, versus what your friends recommends, and what you can afford, you develop the _taste_ through ==enough encounters== such that it form _priors_ within the same limited set, which in turns allows you to form your own worldview:

- One like Rembrandt because that's what's is displayed in the museum that you can reach to.
- You reread the same version of the books because you want to memorize exactly, **word for word** what Camus wrote about how Husserlian intentionality embodied this absurd spirits despite its contradictory appearance:
  > Thinking is not unifying or making the appearance familiar under the guise of a great principle. Thinking is learning all over again how to see, directing one's consciousness, making of every image a privileged place. In other words, phenomenology declines to explain the world, it wants to be merely a description of actual experience. It confirms absurd thought in its initial assertion that there is ==no truth, but merely truths.== [...] Consciousness does not form the object of its understanding, it merely focuses, it is the act of attention, [...] it resembles the projector that suddenly focuses on an image [to borrow Bergsonian image]

Per Ramsey's framework, the _crossroads_ is the wager that induces this friction, insofar as choosing a path/direction. Hyperabundance somewhat blurs this distinction, as in when you can sample both roads simultaneously, when the {{sidenotes[lovely robot]: I want to figure out better term to call these beings, because Shoggoth sounds mean and evil.}} can generate infinite variations of each path, when you can A/B test your life in real-time, the bet stops being a bet, and it _becomes_ a ==simulation of betting==, a performance of conviction without the underlying structure that made conviction possible.

The fucked up thing is this is why we build recommendation systems in the first place, i.e. to accelerate convergence towards a generic-optimal. We are universally {{sidenotes[curved into engagement]: from a philosophy perspective, A.J. Ayer's positive emotivism can be used to describe this phenomenon, largely found through his seminar book [[library/Language, Truth & Logic|Language, Truth & Logic]]}}, rage-bait, click-through, time-on-page, which means they are selecting for writing that doesn't require you to cross any fields, writing that just meet you where you are, because _they know what you like even before you know it_.

In a way, these systems have the same infrastructure with those of {{sidenotes[gambling]: I don't find anything morally wrong towards gambling, other than it is built on top of our worst nature, _power-seeking_}}. Therefore, it creates this evolutionary pressure towards styles that are considered maximally accessible and frictionless. In a world of social pressure full of acted-up performance, the jaggedness gets smoothed out because those edges are considered turn-off.

[——]

I can't and will never be able to write like Camus. I acknowledge that there are a sets of technical skills that I lack. I also acknowledge that there are certain barriers for me to write in English (as a non-native speaker)
- Though the underlying issue about "becoming Camus" or adapting to any tone of writers you aspire to be is that it will require you to make ==irreversible bets== about what matter in your life.
- Camus staked his life on certain ideas about absurdism and resistance, and wrote in a particular historical moment where those stakes were real.
- His style emerged from constraint, from resistance, from the friction between what he wanted to say and what the world allowed him to say.

For me, I'm trying to develop a style in an environment of infinite accommodation, where the "machine of loving grace" will happily generate Camus-adjacent prose for me, where i can retrieve every POSSIBLE variation of absurdist philosophy ON EARTH , where nothing commits because everything's reversible. _I don't want this_

_So you must resist, or might as well die._

---

## being held against lonesome

- [meta]:
  - date: 2026-01-06 06:58:46 GMT-05:00
  - tags:
    - o/eschatology
    - w/fiction
  - socials:
    - substack: https://open.substack.com/pub/livingalone/p/being-held-against-lonesome?r=1z8i4s&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true

_tw: death, gore, body horror_

> What prepares men for totalitarian domination in the non-totalitarian world is the fact that loneliness, once a borderline experience usually suffered in certain marginal social conditions like old age, has become an everyday experience[...]
>
> —From _The Origins of Totalitarianism (1951)_, Hannah Arendt

_the human nervous system registers absence of touch as a form of chronic low-grade stress, measurable in cortisol levels and immune function degradation_

in the progress of building a lossless silhouette of someone i recently came across, i found myself lying here, in this room, a kind of coffin but with better lightning, wondering: _what is it exactly that i want? what is the emergent feelings that my brain trying to resist here?_ to be held, yes, but by ::whom::? by anyone? no, that obscene, that's the logic of the animal, and i am not an animal, i read books, i try to articulate and put it down on paper, i've my own opinion on building [[thoughts/llms|my ghost]], critiques against the frankfürt school, i have refused to download tinder on principle, i resisted getting a new phone number to create a new hinge account. yet here i am, at, <time>07:23:34am</time> with my arms wrapped around my own chest like a man trying to hold his organs inside after a wound, which is precisely what it is, isn't it, a wound, except there is no blood, there is nothing to point to, a doctor would find nothing wrong with me and that is the worst part, that i am entirely healthy and entirely dying at the same time, avoiding working on an interview in 4 hours.

the finns have a word, _kalsarikännit_, which loosely means to _drink alone in your underwear with no intention of going out_, and they have made this into a virtue, and i think perhaps i should move to finland, i should go to the forests where the wolves are, and i would lie down in the snow and the cold would be a kind of touch, wouldn't it, the cold touches everything, it is promiscuous with its attention, it does not ask whether you've earned it, it simply arrives and holds you and does not let go.

but that's not what i want either. _maybe in 20 years_

> Loneliness is the experience of being deserted by all human companions
>
> —Hannah Arendt

[waking up...]

> [...] loneliness is a subjective internal state. It's the distressing experience that results from perceived isolation or unmet need between an individual’s preferred and actual experience
>
> —Our Epidemic of Loneliness and Isolation, _U.S Surgeon General's Advisory, (2023)_

what i want, really, (and here is where it becomes humiliating, where i must avert my eyes from myself), is something so specific that it cannot be named without losing it. if saying it out loud, i afraid i will lose it forever. i want a particular weight of a particular arm wrapped across my chest. i want to hear breathing of the specific being, that is not my own. i want that part of computation to work, for once, in a while, for the feeling to find its proper substrate, for the thing i built in my mind to correspond to something external that does not flinch, that does not relocate to another city, that does not send the message that begins "i'm sorry for misleading you but still i want what is best for you."

DOSTOEVSKY'S UNDERGROUND MAN CLAIMS HE DOES NOT WANT THE CRYSTAL PLACE. he wants to want, _which is different, which is worse,_ because wanting-to-want means you are outside even your own [[thoughts/desire|desires]], watching them as if a man watches fish swimming in an aquarium. I am watching my need for touch and I'm disgusted by it and I am also INSIDE it, drowning in it, and this is the contradiction that Socrates or Kierkegaard or Merleau-Ponty or Simone Weil cannot address.

[in my dream...]

i'm in finland, and i've been walking aimlessly for hours. (_my body has begun its long negotiation with temperature_)

_first stage initialized:_ shivering, which is to say my muscles are burning glycogen in small desperate contractions, 200-250 per minute, calculating heat-loss exceeds heat-production and activating the ancient mammalian subroutines, and i think: _this is what it means to be held by biology, to have something inside you that wants you to live, even when you have stopped wanting it yourself._

the grey ones are watching. i can see them between the birches, which are white like bones, like those i will soon become, and the wolves are patient bc patience is what 40,000 years of evolution has taught them, that the cold does most of the work, and the two-legged things eventually stop moving if you wait long enough. i'm not afraid of them (_this is either the self-made courage or the first symptom of cognitive decline from reduced cerebral blood flow. i suspect it to be the latter_)

_by the second stage_ i have completely forgotten why i came here. the shivering has stopped, as if my body has abandoned the last arithmetic it is programmed to do, and deemed the equation unsolvable. i'm conserving what remains for the core organs, the heart, the lungs, the brain, that is still producing these sentences though it has no reason to, though no one will read them, though they are being written in a medium that does not exist to the [[library/Our Knowledge of the External World|external world]], which is to say that i am thinking, still, for no one, into [[library/Being and Nothingness|nothingness]], and this is the underground man's final joke, that [[thoughts/Consciousness|consciousness]] persists past the point of utility, that i am AWARE of my demise and cannot stop being aware, cannot simply become the object i wanted to become.

the varg, the susi, the canis lupus, they are closer now. i can see the vapour of their breathing. they are metabolizing, converting matter into heat into motion into patience, and soon they will convert ME into these things, and is this not what i asked for? to be held? the wolf's jaw is a kind of holding, the teeth that close around the throat are intimate in a way that nothing else has been, and i think of everyone who has touched me and how none of them touched me like THIS, with such complete [[thoughts/Attention|attention]], such focus, such unwavering PRESENCE.

_the third stage_. i am so hot, fuck me. this is wrong, i know it is wrong, the air is negative thirty for fuck sake! i'm pulling off my coat, my sweater, and the grey church assembles around me, six of them, eight, i have lost the ability to count, and they are watching me undress like i'm performing a ritual, and perhaps i am, perhaps this is the only sacred thing left, to give yourself to the forest, to stop being a subject and turn into a meal.

the alpha, she does not go for the throat. _this is not how it's supposed to happen_. she goes for the flank, then my genitals. (_makes sense, because efficiency matters, especially for Mother Nature. she is ruthless, because she doesn't care much for romanticism_.) they couldn't care less about the quick death, they care about calories, as i feel her teeth entering my calves, and i think: this is the touch i wanted. this is what being held must've felt like. this is the weight of another creature's attention, in its totality and undivided, AND THE PAIN IS EXTRAORDINARY. the pain is the most real thing that has ever happened to me, and i am finally, FINALLY, not in my head, not watching myself from outside. i am HERE, in this body, in this moment, in this mouth.

the hemoglobin has a viscosity of approximately four centipoise, but it moves faster when the heart is panicking, and my heart is panicking, as a biological mechanism. it is doing its job, pumping blood out of me and onto the snow, where it steams for a moment before freezing, and i watch my own warmth vanish from the physical body and become part of the landscape. this is what you want right? _to stop being contained, and leak into the world, to be held by everything instead of nothing._

the pack feeds, and i'm still fully conscious at this point, _which shouldn't be possible_. but apparently consciousness is the last thing to go, the brain hoards its glucose like a miser, and so i am aware of being devoured by the function of nature, i am aware of becoming less, and there is something almost erotic about it. no, not erotic, that's wrong, something...ECONOMIC, a transaction finally completing, as i'm paying my debt to the biosphere, and i'm repaying the calories i consumed, i.e. the pasta with wine, in addition to those elaborate dinners for people who did not stay.

the fenrir, the old wolf, the myth-wolf, she is eating my liver and i am thinking about Prometheus, who had this done to him daily as punishment. This is not punishment, my dear Prometheus, this is a gift, as in the world accepting to what i offered, and Prometheus, you were wrong to scream. You should have been grateful, to be wanted so completely, to be USEFUL, to have eagles return for you again and again because you are worth returning for.

i am less than i was, perhaps 70 kilograms becoming 60 becoming 50 becoming 40, and the wolves are becoming more, as Lavoisier's principle states. i am becoming six wolves eight wolves, i am becoming the forest, i am becoming the snow that will melt in spring and flow into rivers and eventually into the sea, and is this not what loneliness always wanted? to stop being one thing and become all-of-things? to be held by the entire world bc you are now INSIDE the entire world?

_the last thing i feel is not pain._ less sensation, less thinking, less, _me_. but more the cold ground against my back, pressing up into me as gravity presses down. the last firing neurons from the dying brain towards the remaining muscles, and so what? for what purposes do i feel this way? maybe, this is the embrace i asked for wandering the Finnish forest, the planet itself holding me against its chest, and the wolves are eating and the stars are watching and i am

i am

i

[waking up once more...]

the loneliness will just become a second skeleton residing inside my skeleton, and i will tell no one about it, bc who could i tell, and what would they do, and would their doing-something not simply be another form of the wrong substrate, another computation that fails to compile?

i want to stop moving.

i want to be still and held and i want to stop wanting.

the wolves are patient. the snow is patient. but i'm not. i'm lying here, in this room, in the early morning of January winter, sweaty af after a 30-minute-hypertrophy-workout, in this coffin with wifi, typing this in neovim,

waiting for something to arrive.

waiting for something

wait.

---

- [meta]:
  - date: 2025-12-29 02:48:33 GMT-05:00
  - tags:
    - llm
    - love

I want to build my own [[thoughts/LLMs|ghost]], an autistic, psychopathic qwen3-aarnphm. But I'm afraid I don't know how to encode the mixed feelings of love and relationship towards friends onto the latent space.

---

- [meta]:
  - date: 2025-12-23 03:50:33 GMT-05:00
  - tags:
    - o/eschatology

_tw: death_

If I'm away, I want to be remembered as _a loving friend_. Not as a 1% delta in Blackwell kernels. Not as a footnote in structured outputs. Not as another unit in the scaling war. Just a friend: the kind who noticed the human in front of them.

If I'm away, I'm sorry for the externalities. I'm sorry for the unfinished loops—calls not returned, goodbyes deferred, care I postponed until it expired. I'm sorry for not saying goodbye to my parents. I'm sorry if your last interaction with me is just logistics and entropy. I'm sorry for the rotten food in the fridge that you'll have to clean. I'm sorry that the smell of rotten flesh will be the last thing you've remembered me by.

If I'm away, I'm not sorry in the sense of conceding that the universe gets to assign me a moral score. I did the best I could with the compute I had. If there's a capital-G-agent-or-man-or-being running evaluation, it can inspect the source code; if there isn't, then guilt is just a heuristic firing after the fact. I'm not sorry, because I just need some quiet times, sometimes.

If I'm away, please don't be sad in the way that freezes your future. Grief is a computation; run it to completion, then spend the rest of your life-budget on living. Death doesn't get to claim a second tax in the currency of your attention. Please don't be sad, because life is the existence of suffering, and the call of the eternal life.

Finally, if I'm away, treat my savings like an allocation problem. Put it toward cancer research. Sell the farm so the animals get continuity. Liquidate the S&P 500 if you need liquidity. Pay whatever taxes you legally must (yes, the state will still take its bite). If you want a ritual, put a small fraction on red and salute the god of variance. Then go look after your people. Don't reserve part of your heart for me. I don't deserve any part of it.

---

- [meta]:
  - date: 2025-12-12 18:36:27 GMT-05:00
  - tags:
    - process

![[thoughts/Agency#seven ways to become unstoppably agentic]]

---

## on learning through presence

- [meta]:
  - date: 2025-12-12 12:40:40 GMT-05:00
  - tags:
    - process

from [on learning through presence](https://www.humaninvariant.com/blog/presence):

> When you show up in person, you feel like you don't belong. You quickly learn that others have a deep and rich shared cultural history that has spanned over the better part of a decade, while you are a newcomer. You try your best: you connect with people during [late night conversations](https://www.humaninvariant.com/blog/conversations) you would have never connected with otherwise, and maybe even plan to throw a joint party together in New York.

> You learn more about the shape of the life you want, the types of relationships you want with certain people, and the sacrifices you are willing to make to be ambitious. You learn that ambient ambition is real, and what it really means to love what you do, when you witness someone over twice your age work into the night and enjoy every moment.

> You learn what it feels like to use a typewriter because somebody else cared enough to give you the opportunity to experience that feeling and made it happen. You use that opportunity to ponder [the market structure of writing implements](https://www.humaninvariant.com/blog/lowercase) and to write an endearing note to the people who have given you the opportunity to learn more about yourself. While doing so, you learn what the sound of a bell means when typing on a typewriter.

> You learn that having the courage to care is the scarcest resource in the world. You can [predict the future you want by caring enough to build it](https://www.humaninvariant.com/blog/worldbuilding).

> Most importantly, you learn that the ability and desire to care is built through presence.

there's a striking prior when comparing [[thoughts/writing|writing]] with presence. both are processes where the transformation happens _during_ the act of doing it, rather than afterwards. You're thinking out loud when you write, and oftentimes thoughts are then lossy collections of ideas that you've scribbled onto the pages, and soon thereafter realising half of them are {{sidenotes[wrong.]: I don't mean in the literal sense, but rather an incoherent/illogical collections of word-pile-that-you-vomit-out-onto-the-page.}} presence sort of work the same way here _(or at least from the blog)_ where you discover it by noticing which conversations make you lean in, which type of energy you want to absorb, which futures you find yourself involuntarily imagining.

You can believe in this to your heart's content, but I think they both look like search algorithms running on wetware-presence searches possibility-space for the life-shape that fits with your lived experience. The result of such a product is merely an evidence that such _process_ occurred.

I wonder if this is why reading essays and consuming YouTube videos produce little lasting change if you aren't really putting effort into actually studying the subject at hand. In a sense, you're watching the residue/product of someone else's search process, instead of running your own. The illegible inputs—the drafts, the small talk, the wrong turns—contain the actual epistemic work, which is completely removed from the final products, is the thing that ::matters:: the most.

---

- [meta]:
  - date: 2025-12-03 09:33:42 GMT-05:00
  - tags:
    - love

![[thoughts/love#hw]]

---

- [meta]:
  - date: 2025-11-30 11:01:47 GMT-05:00
  - tags:
    - o/life

![[quotes#^shed]]

## on omnipotence paradox

- [meta]:
  - date: 2025-11-29 13:01:53 GMT-05:00
  - tags:
    - ontology
    - philosophy

can an [omnipotent](https://iep.utm.edu/omnipote/) being create a stone so heavy they cannot lift it? if yes, then they cannot lift it—failure of omnipotence. if no, they cannot create it—failure of omnipotence. either way, {{sidenotes[omnipotence fails.]: atheological arguments use this not primarily as evidence against god's existence but to show "omnipotence" as traditionally conceived may be incoherent.}}

the paradox reveals that "maximal power" might be conceptually malformed, like "set of all sets" or "north of the north pole." not difficult to achieve but impossible to coherently {{sidenotes[specify.]: it is _impossible_ to create an uncreated object—not because of limited power but because the phrase doesn't describe a possible state of affairs.}}

SEP's definition of _omnipotence_ follows:

> Omnipotence is maximal power.

This [comment](https://philosophy.stackexchange.com/a/34397) states that what we are often referring about the paradox is synonymous to a _absolutist proposition_. The three resolutions are as follow:

1. The notion of an absolutely immovable physical object is logically incoherent.
   - To be a physical object means being subject to physical forces, which means having some finite mass or hardness.
   - By definition, anything exceeding that could move or alter it.
   - The concept of an unliftable object is {{sidenotes[self-contradictory.]: [Aquinas](https://en.wikipedia.org/wiki/Thomas_Aquinas) in _Summa Theologica_ I, Q.25, Art.3 argues inability to do contradictory things "does not signify a defect of power" (_non significant defectum potentiae_). Apparent "inabilities" like inability to sin represent perfection of power, not deficiency. Self-contradictory pseudo-tasks aren't genuine objects of power.}}
   - This is Thomistic scholasticism view of the paradox. (or _pseudo-task dissolution_)
2. That any limitation on God is a form of self-restraint rather than fundamental limitation.
   - In other words, God can create an object God says God cannot move, and God won’t move it
   - But not because it is {{sidenotes[immovable]: People that take this view and think there’s a God would be committed to a form of voluntarism.}} per se but instead ==immovable per volens.==
   - This in turn can also be extended by Descartes' argument where _logical necessities themselves are contingent on divine will_
   - Decartes' letters to Mersenne (1630) would emphasize that _god could have made it false that twice four equal eights_ if he wishes so [^letter-to-mersenne]
3. That God can impose self limitations that stand permanently.
   - In other words, God can make a rock God cannot lift.
   - Again, the origin wouldn't be that the rock has infinite mass but that God can manufacture the rock and bind a condition on God's own self to not be able to {{sidenotes[^pick up the rock]}}.
   - (see the footnotes for _postmodern_[^postmodern] interpretation)

{{sidenotes[pick up the rock]}}:
    The closest pre-modern candidate for this can be traced back to Lurianic Kabbalah's _tzimtzum_ (divine contraction, 1570s Safed), where the Infinite (_Ein Sof_) withdraws to create "vacant space" for finite existence. But then, _tikkun_ (repair) implies eventual restoration, and _reshimu_ (residual trace) suggests God never fully withdraws. Which implies there are resistant from making it absolute.

    More importantly, the _potentia absoluta_ vs _potentia ordinata_ distinction (originated by Hugh of St. Cher in 1230s commentary on Lombard's _Sentences_, later refined by [Scotus](https://plato.stanford.edu/entries/duns-scotus/#ProExiGod) and [Ockham](https://plato.stanford.edu/entries/ockham/)) holds that God _reliably would not_ deviate from ordained commitments due to divine faithfulness, not that God _could not_. This implies God _chooses_ not to intervene with such objects, rather than inability to do so.

[^letter-to-mersenne]:
    Don’t hesitate to assert and proclaim everywhere that it’s God who has laid
    down these laws in nature just as a king lays down laws in
    his kingdom. There’s not one of them that we can’t grasp if
    we focus our mind on it. They are all inborn in our minds,
    just as a king would, if he could, imprint his laws on the
    hearts of all his subjects.

    God’s greatness, on the other hand,
    is something that we can’t •grasp even though we •know it.

    But our judging it to be beyond our grasp makes us esteem
    it all the more; just as a king has more majesty when he is
    less familiarly known by his subjects, provided they don’t
    get the idea that they have no king—they must know him
    enough to be in no doubt about that.

    You may say:
    - ‘If God had established these truths he would have been able to change them, as a king changes his laws.’
    To this the answer is:
    - He can change them, if his will can change.
    - ‘But I understand them to be eternal and unchangeable.’
    - And so is God, in my judgment.
    - ‘But his will is free.’
    - Yes, but his power is beyond our grasp. In general we can say that God can do everything that we can grasp, but not that he can’t do what is beyond our grasp. It would be rash to think that our imagination reaches as far as his power.

[^postmodern]:

    The current disagreement surrounds whether the limitation is considered _voluntary_ or _essential_.

    19th century German kenoticism (Thomasius, Gess, Ebrard) developed systematic kenotic Christology from Philippians 2:5-8:
    ```quotes
    5 In your relationships with one another, have the same mindset as Christ Jesus:

    6 Who, being in very nature[a] God,
        did not consider equality with God something to be used to his own advantage;

    7 rather, he made himself nothing
        by taking the very nature[b] of a servant,
        being made in human likeness.

    8 And being found in appearance as a man,
        he humbled himself
        by becoming obedient to death—
            even death on a cross!

    Philippians 2:5-8
    ```

    Here, the logos divested "relative" attributes (omnipresence, omniscience, omnipotence) but retained "immanent" ones (holiness, love, truth) during Incarnation. Critically: _temporary_, ending with Resurrection. The limitation is real but bounded—a divine hiatus, not a permanent restructuring. Orthodox critics (Chalcedonian) argue this creates a "binity problem": if Christ divests divine attributes, the Trinity breaks during Incarnation. The second Person takes leave from Godhead. If Father and Spirit retain omniscience while Son doesn't, they can't share the same substance. Thomasius relocates the two-natures problem rather than solving it.

Can a genuinely free being make an _irrevocable_ choice?
- If revocable, the limitation isn't permanent (Thomasius).
- If irrevocable, has freedom been compromised (Polkinghorne)?
- This paradox of self-binding runs through political philosophy (constitutionalism—can one generation bind the next?), personal ethics (promising—can you obligate your future self?), and now theology—without clean resolution.
- The voluntarist and essentialist positions may be unstable in ways that mirror the original omnipotence paradox they sought to escape.

> If God can limit divine attributes, then why assume any are essential to begin with?

I wonder if we should define omnipotence via **act-theory** (ability to perform any logically consistent action) or **result-theory** (ability to actualize any possible state of affairs)? Result theories handle the paradox where "there being a stone an omnipotent being cannot lift" isn't a possible state of affairs, therefore the _inability_ to actualize it is no limitation. But result theories carry heavy metaphysical commitments that they require omnipotent beings exist **necessarily** and may constrain human freedom.

The paradox reveals more about how we think about power than about divine attributes. We model omnipotence on human power "only without limitations"—but maybe the category doesn't scale. Maybe maximal power is QUALITATIVELY different, not quantitatively maximal. Maybe "maximal power" as conceived in agent-causal term is incoherent at infinite {{sidenotes[extension.]: J.L. Cowan (1965, 1974) argues any attempt to resolve the stone paradox "must fail"—the concept itself is definitively broken. Anthony Kenny's _the god of the philosophers_ (1979) concludes "there can be no such being as the god of traditional natural theology." not "we need better definitions" but "abandon the project."}}

If anyone is familiar with [[thoughts/Wittgenstein#Russell's paradox and the vicious circle principle|Russell's paradox]], then the similarity is uncanny here. "set of all sets that don't contain themselves" generates the paradox via self-reference. Type theory only really replaces the naive concept with a more restricted versions, i.e ==there is no universal set==.
Similarly, "god can do anything" generates the stone paradox through self-references. if the solutions for Russell's paradox is conceptually malformed, then the  same would hold for "maximal power". [^cantor]

[^cantor]:
		Patrick Grim extends this via Cantor, where any set of truths has more subsets than members, each corresponding to a unique truth.

    There's no totality of truths—truth "explodes beyond any attempt to capture it." if omnipotence means power over all possible states of affairs, and there's no ==totality of possible states of affairs== (by analogy to cantor), omnipotence is malformed.

		You can have very extensive power, but "maximal" or "unlimited" may be incoherent the way "set of all sets" is incoherent.

    Cantor distinguished absolute infinity _in deo_ from mathematical transfinite, arguing absolute infinity is "logically inconsistent" and belongs to speculative theology, not mathematics. he avoided treating the mathematical universe as a set, recognizing the paradoxes this generates.

What makes something a "task"?

Tasks presuppose:
- initial conditions
- final conditions
- causal pathway between them
- possibility of failure

"Create a stone you cannot lift" presupposes you have lifting capacity $C$, stones can have weight $W > C$, creating something doesn't change $C$.
But for maximal power, there IS no $C$—no upper bound.
So "stone too heavy to lift" isn't a pseudo-task (aquinas) or ill-formed question (Frankfurt, The Logic of Omnipotence, 1964), but a **CATEGORY ERROR**.

How would one reconceptualising divine power? If "omnipotence" is incoherent, what should replace it?

Tillich placed god as "ground of being" rather than powerful agent. Omnipotence then becomes a symbol expressing "the power of [[thoughts/being|being]] which resists nonbeing"—not ability to perform tasks but condition-of-possibility.

Heidegger's onto-theology critique treated god as _causa sui_ or highest being is bankrupt. The entire framework of "god has power X" is malformed. God isn't a being among beings to which predicates apply.

Caputo's weak theology then explicitly deconstructs divine omnipotence. God understood through Greek metaphysical attributes (immutability, omnipotence, omniscience) should be deconstructed. The name "god" harbors an event rather than naming a powerful being.

We can then shift from "god can do X" to "god grounds the possibility of X." Power-as-condition rather than power-as-capacity. This dissolves the stone paradox bc it's not asking "can the condition-of-possibility create something impossible?"—impossibilities aren't in the domain at all. I understand that the cost of this would make god _metaphysically distant from history/agency_. If you want a god who does things in history (answers prayers, performs miracles), then you're stuck with omni-attributes and their paradoxes (which is fine). But I find these concepts/positions of God makes more sense for the logical brain.

Kenotic theology tries to have both—a god who acts in history but isn't bound by classical omnis—but as we show above, every version either relocates the problem (voluntary kenosis) or diminishes God to where "deity" seems honorific (essential kenosis). There might not BE a stable middle ground.

I then come to the conclusion where the omnipotence paradox is a genuine antinomy (in the Kantian sense)—concepts applied beyond their legitimate domain generate contradictions. Therefore, the right response isn't solving it within the framework but abandoning the framework for once. Standard solutions (pseudo-task dissolution, voluntarism, kenosis, Frankfurt) either deny one horn or simply relocate the problem. (big cope really)

---

- [meta]:
  - date: 2025-11-28 16:11:39 GMT-05:00
  - tags:
    - writing

![[thoughts/writing#as a journey for exploration]]

---

- [meta]:
  - date: 2025-11-27 15:57:58 GMT-05:00
  - tags:
    - alignment

Anthropic found that natural emergent misalignment stems from [_reward hacking_](https://www.anthropic.com/research/emergent-misalignment-reward-hacking). Though, I suspect that ablating these "bad behaviour" wouldn't necessarily make the model more aligned. What if having certain malicious intent is actually helpful?

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
    - o/life
    - o/relationship

I got a cup of hot chocolate today. It reminded me of L, and somehow, we carry fragments of them within us without knowing so.

---

- [meta]:
  - date: 2025-11-21 12:43:25 GMT-05:00
  - tags:
    - o/relationship

I have no desire of making new friends in Toronto anymore. Everything felt so superficial here.

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
    - o/life

Got on a call with middle school friends, seeing my old Vietnamese teacher, B—there's warmth in it. Good memories, honestly. But I'm glad I left when I did.

---

- [meta]:
  - date: 2025-11-15 01:58:17 GMT-05:00
  - tags:
    - love

Why does this hurt so bad? We aren't even in a relationship.

---

- [meta]:
  - date: 2025-11-13 11:33:37 GMT-05:00
  - tags:
    - love

I'm completely broken down, seeing the physical letter L left on the counter. I'm at a loss for words.

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

The [[thoughts/ethics#deontology|deontological]] arguments of life _presupposes_ the feeling of ::interconnectedness{h5}::. I think that if one acknowledges that no one wants one in life, then one might conclude that one's life is not worth living. [[thoughts/Philosophy and Kant|Kant]] would then argue against the notion of "kill oneself" given that it is an act in violation of the [[thoughts/moral|moral]] law, and deemed it as **wrong**.

> I wonder how much of this is due to the lack of imagination. If you're suicidal, it's very difficult to imagine life getting better.

You're depressed, sure, so your cognition is impaired, such that you realised life isn't worth living anymore, which then concludes that suicide is the real solution. So your mind convinces you to just **give in**

```quotes
The impulse to end life and the impulse to further life contradict each other.

Kant, Groundwork _§4:422_
```

But I'm troubled by how close this gets to "just shake it off" or "try something new!" as if suicidal depression were boredom with your routine. The difference, maybe, is that the conventional advice assumes you can white-knuckle through by force of [[thoughts/Will|will]] or positive thinking.

Guzey's argument focuses on a darker path, where you can't think your way out, and you probably can't will your way out either. You need an external disruption large enough to short-circuit the entire system. Which is less hopeful than it sounds—e.g: you're not in control of your own recovery. You're waiting for something to happen that breaks the pattern, or you're throwing yourself at random experiences hoping one of them will.

The wanting to live that exists underneath reasons, that persists even when you've eliminated every reason you can name. Whether that's the body asserting itself, or some pre-rational attachment to [[thoughts/being|being]], or just biochemistry doing its thing.

What brought me back wasn't a good argument for living. It was N's voice, L's conversations, the way light looked one morning, the prospect of missing something I hadn't experienced yet. The stuff that doesn't make sense in the ledger of reasons but somehow weighs more than the whole column of evidence against continuing.

I think, subconsciously, we all have a moral duty to continue to {{sidenotes[live,]: go outside, look at trees, eat an onion sandwich, buy some sourdough}} and only when you have exhausted all possible solutions, then suicide is the last reasonable solution to end the absurd life.

I'm once again, thinking about suicide.


---

- [meta]:
  - date: 2025-11-12 21:33:51 GMT-05:00
  - tags:
    - o/life

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
    - o/life

I find myself using my mechanical keyboards less and less nowadays, using my laptop keyboard instead. This might have to do with the mode of focus the laptop keyboard puts me into—something about focusing on the work itself rather than the tools. Partially because of the wrist pain from long sessions of working at my desk 🐙

---

- [meta]:
  - date: 2025-10-31 18:24:43 GMT-04:00
  - tags:
    - o/life

Writing kernels sounds way more fun than whoring on the streets of Toronto. Happy Halloween 🎃 though.

---

- [meta]:
  - date: 2025-10-30 03:18:38 GMT-04:00
  - tags:
    - writing

```quotes
Writing essays, at its best, is a way of discovering ideas.

Paul Graham, [The Best Essay](https://paulgraham.com/best.html)
```

> An essay should ordinarily start with what I'm going to call a question, though I mean this in a very general sense: it doesn't have to be a question grammatically, just something that acts like one in the sense that it {{sidenotes[spurs some response.]: When you find yourself very curious about an apparently minor question, that's an exciting sign. Evolution has designed you to pay attention to things that matter. So when you're very curious about something random, that could mean you've unconsciously noticed it's less random than it seems.}}

---

- [meta]:
  - date: 2025-10-29 05:18:28 GMT-04:00
  - tags:
    - productivity
    - philosophy

Late night work listening to Dreyfus' lectures hits like smoking a good joint on a Friday night.

![[https://www.youtube-nocookie.com/embed/usxvyf3xqcQ]]

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
    - o/life

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

[[thoughts/Camus]] begins with suicide in [[library/The Myth of Sisyphus]], more or less a demonstration of how absurd life really is. He used suicide as a scapegoat of people who don't have enough courage to deal with the hardship of life being thrown at them, and considered suicide as cowardice.

I have exercised this thought _many times_, in the way one considers a standing appointment. Not with desire of actually doing the deed or the fear of death, but rather a procedure to keep my mind sharp. The world offers no meaning, yet I require meaning, therefore, it doesn't seem reasonable to be alive.

If I could leave, what makes me stay? When I actually sit with this question, as a real possibility, the answers surprise me. I want to see how this conversation with N resolves on Thursday. I'm still waiting for L to send that letter my way. There's a problem on my desk I'm halfway through understanding, and I need to know if my intuition about it is right. I think about my parents getting the phone call, my friends having to sort through my things, and something in me recoils not from death but from inflicting that particular grief.

Turns out the results are mostly attachments. They are somewhat very small, specific to my life, but accumulate gradually.

I watch people in coffee shops, on trains, in office buildings. They seem to continue, mindlessly. Felt like aimless drones just letting time pass through their body. Pour a cup of coffee, send those emails, make plans for next week. Perhaps they have solved something I have not. Perhaps they have simply never filed the paperwork for the question. It is possible everyone considers this and we have agreed, collectively, not to mention it. Like a standing meeting no one enjoys but everyone attends.

My work is not that special, my contributions to the collective projects are temporary. Life won't change a lot if I disappear. Heat death will erase everything eventually. Still, I'm here though. Maybe because of a particular arrangement of attention and time, maybe because I still care enough, maybe the thought of striving another day for N, parents, L, J, C, S are good enough motivators to keep one going.

> Meaning does not require permanence. This seems important but I am not certain why.

[[thoughts/Camus]]'s revolt: Continuing with full knowledge of the absurd. Choosing again each day, not from habit but from decision. I am not certain I achieve this. Most days feel like habit.

The question becomes dangerous when it stops being theoretical. When it moves from mind to body. When the weight becomes physical rather than philosophical. Then one must interrupt the process. Call someone. Leave the room. The distinction is administrative: one is philosophy, the other is emergency. Emergency requires different procedures entirely.

## why superhuman AI won't kill us all.

- [meta]:
  - date: 2025-10-25 23:26:00 GMT-04:00
  - tags:
    - llm

—[Eliezer Yudkowsky](https://www.youtube.com/watch?v=nRvAt4H7d7E)

Yudkowsky's full argument on eschatology isn't productive whatsoever. Feels a lot more science fiction writing, where he claims that these systems will end up "wanting to do their stuff without wanting to take the pills that [we offer to] makes them to do stuff that we _wants_ them to do instead."

Yudkowsky's claim relies on [[thoughts/AGI|superintelligent]] optimizer + misaligned goals = human extinction, a sort of self-recursion improvement towards its' own power-seeking agendas. bc we're made of atoms it can use for something else (i.e a resource allocation problem). Nanotech grey goo. Designer pathogens. Trees that grow computer chips. All physically possible, therefore inevitable once you have sufficient optimization power.

But intelligence in actual systems is jagged, domain-specific, constrained by architecture. AlphaFold is brilliant at protein folding and useless at everything else. [[thoughts/LLMs|GPTs]] can write code but can't reliably count letters (update from 01/09/26: not anymore). The jump from "good at prediction" to "can design novel molecular machinery from first principles" assumes transfer learning we haven't seen. These very much resemble the old GOFAI vs NFAI arguments. Maybe the argument here is to have a _composition of multiple domain-specific superintelligence systems_ that amplify our life.

The "foom" scenario requires explosive recursive self-improvement, which is abstruse. GPT-6 builds GPT-7, capabilities doubling weekly until godlike intelligence. Architecturally speaking, maybe we figure out something that scales with attention, but it has to be beyond just Transformers, maybe in conjunction with something like JEPA. The argument assumes breakthroughs on demand.

He did mention a recursive need for hydrogen, but fwiw physical constraints matter a lot more here. Building nanoassemblers needs: labs, materials, energy, time for experiments. Biology took billions of years of parallel search to reach cells. You can speed that up with intelligence – how much? The argument assumes "enough."

The frame requires that one assume worst case at every branch, assume maximum capability, assume minimal constraints, therefore _doom_. I just don't think that's how you build things. Real systems fail in boring ways. Scaling laws break. Architectures saturate. The chain from "AI breaks up marriages" to "superintelligence converts biosphere to computronium" requires assumptions that would be rejected in any engineering domain.

```quotes
don't build, don't experiment, don't iterate, because any mistake might be the last. That's **not** how we've solved any complex safety problem. Treating everyone who continues working as equivalent to cigarette executives isn't engaging with technical disagreements.

Yudkowsky
```

[[thoughts/Alignment]] is hard. I do think that capabilities scale faster than safety. But the response can't be "stop everything and hope treaties hold." It has to be: build better systems, understand current systems deeply, develop alignment that might work. You need feedback loops. You need to learn from failures at scales where failure isn't extinction.

## how I wrestle with the idea of god

- [meta]:
  - date: 2025-10-24 01:23:00 GMT-04:00
  - tags:
    - theology
    - o/life

"God exists" and "the table exists" share the same grammatical structure, verb, declaration of being. [[thoughts/Wittgenstein|Wittgensteinians]] would tell you these are entirely different language games, different [[thoughts/forms of life]] enacted through identical syntax.

When someone tells me "I feel God's presence," I wonder what they're actually describing. The [[thoughts/Metaphysics|metaphysical]] claim, that an immaterial conscious being is causally interacting with your experience right now, is the biggest cope I've ever seen. The phenomenology itself is trickier to describe: how we say _the feeling of being held_ when no one's holding you, _of mattering_ when the universe gives no indication you do, _of company_ in the dark when empirically you are alone??

I've had these feelings, once. Walking home at 3 AM and suddenly feeling like the street lights are watching over me. Finding a book at exactly the moment I needed it. That uncanny sense of [[thoughts/Alignment|alignment]], of #pattern, of something speaking directly to you through the noise of existence. Theism often offers a ready explanation here, providence and divine orchestration and God moving through your life, and I understand why people find it satisfying.

Strip that away and what type of grammar are we left with to describe this a priori?
- "Coincidence" dismisses the actual experience too quickly, because the word suggests you've explained something when you've only labeled it.
  - The feeling persists: events _mean_ something, your pattern-matching brain encounters random noise and insists there's signal there.
  - This is {{sidenotes[apophenia,]: the tendency to perceive meaningful connections between unrelated things. we see faces in clouds, we find messages in static, we construct narratives from scattered data points.}} the same cognitive architecture that lets you recognize your mother's face in a crowd, now misfiring on the universe itself. And the misfiring feels exactly like insight.
- {{sidenotes[^"Synchronizität"]}} doesn't make a lot of sense here, _meaningful coincidence_. It sounds scientific until you push on it, at which point it dissolves into gesture. Jung wanted to preserve the phenomenology without the theology, to say "this feeling of connection is real" without committing to a god who arranges it. The problem is that "acausal connecting principle" explains nothing, but a label masked as some polysyllabic word destined to be something more


{{sidenotes["Synchronizität"]}}:

    Jung coined the term in 1952, working with pauli (yes, the physicist). the idea was that meaningful coincidences occur through some acausal connecting principle.

    Pauli was drawn to it because quantum mechanics had already broken his intuitions about causality. And Jung liked it because it gave his patients' experiences a respectable-sounding name.


    neither of them could say what the mechanism was.

"God is good" sounds like "water is wet", a property of an existing thing. Maybe it's closer to "justice is sacred", an orientation rather than a description. A way of being in the world, encoded in grammatical structures that trick you into thinking you're making claims about reality when you're actually describing how you move through it.

The analytic philosophers did try to take a stab at this problem:
- A.J Ayer, in {{sidenotes[^that thin slash of a book from 1936]}}, didn't argue that God doesn't exist. He argued that "God exists" doesn't say anything at all. The sentence has the grammar-of-a-claim but the content of a sigh. The claim is *meaningless* rather than false, and if theism is meaningless then atheism is too, because there's nothing to deny.
- W.V. Quine pushed further, though sideways, more in a economic sense: _what beliefs_ {{sidenotes[earn their keep?]: "to be is to be the value of a bound variable" (*From a Logical Point of View*, 1953). your [[thoughts/Ontology|ontology]] is what your best theories quantify over. Quine though never explicitly whether or not this applied this to god, but my inference is that _god doesn't appear in physics or biology or any of our best theories_. The threat is that this criterion may also eliminate numbers, theoretical entities, maybe the self.}} We commit to electrons and genes and spacetime because our best theories quantify over them, they do work, make predictions, connect to other beliefs. God, therefore, a posit, same as Homeric gods, same as physical objects. Physical objects are more useful, because God doesn't pay the rent.
- Antony Flew {{sidenotes[adapted]: the parable originated with John Wisdom in "Gods" (1944). Flew's version added the escalation of immunizing moves: invisible, intangible, insensible, eternal. "death by a thousand qualifications" is flew's phrase.}} the invisible gardener: two explorers find a clearing, one insists a gardener tends it. They set watches and no gardener appears. The believer qualifies: invisible, intangible, insensible, eternal. Flew asks at what point the invisible gardener differs from no gardener at all, because theological claims die by {{sidenotes[death-by-qualification.]: Flew converted to deism in 2004, age 81, after fifty years of arguing against god. "what I think the DNA material has done is show that intelligence must have been involved." the guy who invented the falsification challenge found something he'd count as evidence. the 2007 book *There Is a God* has authorship controversy, largely ghostwritten while Flew had cognitive decline.}} and whatever would count against them gets absorbed into divine mystery.
- Alvin Plantinga's response is that belief in god might be {{sidenotes[properly basic,]: reformed [[thoughts/Epistemology|epistemology]] (*Warranted Christian Belief*, 2000) holds that some beliefs are foundational, requiring no inferential support. you don't prove other people are conscious, instead you just find yourself **believing it**. the sensus divinitatis, calvin's term, works the same way: a faculty that produces belief in god when triggered by beauty, guilt, gratitude. the famous objection is "why can't belief in the great pumpkin also be properly basic?"}} like belief in other minds, something you don't derive from evidence so much as find yourself believing.

{{sidenotes[that thin slash of a book from 1936]}}:

    i.e. [[library/Language, Truth & Logic|_Language, Truth and Logic_]]. he was 24 when he wrote it, around 160 pages.

    the verification principle was self-refuting, which ayer conceded decades later:

    > the verification principle never got itself properly formulated. I tried several times and it always let in either too little or too much."

    logical positivism died by its own sword. and then, in 1988, ayer had a near-death experience, heart stopped for four minutes, wrote about it in "What I Saw When I Was Dead.":

    > i was confronted by a red light, exceedingly bright, responsible for the government of the universe."

    he insisted it didn't change his atheism. his physician would claimed otherwise.

I think about that walk then, the 3 AM street lights, and I'm still uncertain what to call the feeling. "Revelation" carries too much freight, and "delusion" is revelation's shadow, still shaped by what it denies. Ayer would say I wasn't making claims at all, just emoting with declarative syntax. Maybe so. The feeling of meaning-making isn't nothing, though. It's the most robust data point I have. The positivists wanted to clear away metaphysical fog and ended up sweeping aside the only thing worth mapping.

Kierkegaard said the leap of faith requires a certain degree of [[thoughts/Camus|absurdity]], that you leap beyond reason into belief. We're all leaping anyway, every morning, into relationships we know will end, into projects that will be forgotten, into futures that terminate in death. The absurdity is pouring coffee at 7 AM when you know, really know, viscerally rather than just intellectually, that entropy wins and nothing persists and the heat death of the universe is coming for everything you've ever loved. And yet here we are making breakfast, answering emails, planning for Tuesday.

This isn't Kierkegaardian faith in the sense of a transcendent leap into divine arms. There's still a leap, though: acting _as if_ things matter despite having no cosmic guarantee they do. Maybe religious language expresses this fundamental human insistence on continuing, on treating temporary patterns as if they're eternal truths, on finding meaning in noise because meaning-making is what conscious matter does.

The [[/posts/feelings|feeling]] of being held, of mattering, of not being alone, persists even after you've dismantled the theological framework. These feelings are built into consciousness itself, this need for connection and significance and something beyond the merely material. The feelings arise because you're the kind of system that generates them when confronting its own existence.

Strip away providence and you're left with pattern-matching systems encountering randomness and desperately trying to read it as text, as message, as meaning. We can't help it. It's how we're built. The grammar of theism gives us vocabulary for these experiences, but the experiences themselves are what happens when consciousness encounters itself in a universe that doesn't care, when the meaning-making machine meets the meaningless void and keeps making meaning anyway, through the void rather than despite it, because of it.

---

- [meta]:
  - date: 2025-10-25 16:50:00 GMT-04:00
  - tags:
    - fruit

The visceral feeling of returning to a place you once lived. Those familiar streets you'd walk daily without really noticing. The tiny brick that used to catch your shoe—gone now. New shops sprouting up, the decommissioned church turned taco spot. The coffee shop with its familiar faces, where you just walk by and smile at each other, no words needed. That bar you used to haunt.

What gets me is seeing the old faces, the same constellation of people I used to orbit. Five months in my new Toronto apartment (which I love, truly), but it feels like years have passed. Time does this strange thing – warps around you when you leave. For me, whole epochs have unfolded. For them, Tuesday follows Monday follows Sunday. The uncanny valley of temporal perception: you've moved through time at different speeds, yet here you both are, occupying the same present moment.
