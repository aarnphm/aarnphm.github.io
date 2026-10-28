// Halloween grids for the trail, in the holiday palette of holiday-sprites.ts: `h` is mourning
// black, `n` wrought iron, `k` blood, `l` lamp glass that lights after dark. Palette indices only;
// pixels() turns them into 2 px (or 3 px) cells. Victorian penny-dreadful register: gallows, gaslight,
// resurrectionists, a hearse, a crypt.

// Carved faces are `l`: dark by day, lit from inside at night.
export const pumpkin = [
  '....bv....',
  '..iiiiii..',
  '.iosoosoi.',
  'iooloolooi',
  'iolloolloi',
  'iolooooloi',
  'ioolollooi',
  '.iosoosoi.',
  '..iiiiii..',
]

export const pumpkinSmall = [
  '...bv...',
  '.iiiiii.',
  'iolooloi',
  'iooooooi',
  'iolllloi',
  '.iosooi.',
  '..iiii..',
]

export const vine = [
  '......',
  '..v...',
  '.vv.v.',
  'v..vv.',
  'v...v.',
  '.vv.v.',
  '...vv.',
  '....v.',
]

// A dead oak in mourning black: a raven on the left limb, a noose from the right. Drawn at 3 px.
export const gallowsTree = [
  '..............h..............',
  '......h.......h.........h....',
  '......hh.....hh........hh....',
  '.......hh....h........hh.....',
  '........hh..hh.......hh......',
  '.hhg.....hh.h.......hh....h..',
  '.hhhh.....hhh......hh....hh..',
  '...hh......hh.....hh...hhh...',
  '...hh......hh....hh..hhh.....',
  '....hh.....hh...hhhhhh.......',
  '.....hhh...hhhhhhhh..........',
  '.......hhhhhhhhhh..hhhhhhhhhh',
  '.........hhhhhhhhhhh....b....',
  '..........hhhhhhh.......b....',
  '...........hhhhh........b....',
  '...........hhhhh.......bbb...',
  '...........hhhhh.......b.b...',
  '...........hhhhh.......bbb...',
  '...........hhhhh.............',
  '..........hhhhhhh............',
  '..........hhhhhhh............',
  '.........hhhhhhhhh...........',
  '........hhhhhhhhhhh..........',
  '.......hhhh.hhhh.hhh.........',
  '......hhh...hhh...hh.........',
]

// Ravens leave the gallows tree; placed with the bats.
export const raven = [
  '.hh.....',
  'hhhh..hh',
  '.hhhhhh.',
  '...hh...',
]

// A street lamp off the lower trail. The glass is `l`, so it burns after dark.
export const gaslamp = [
  '...n...',
  '..nnn..',
  '.n.l.n.',
  '.nllln.',
  '.nllln.',
  '.nllln.',
  '..nnn..',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '...n...',
  '..nnn..',
  '.nnnnn.',
]

// A family crypt with an iron gate; something has bled on the step.
export const mausoleum = [
  '...........s............',
  '..........sss...........',
  '...........s............',
  '........iiiiiiii........',
  '.......isssssssi........',
  '......isssppsssssi......',
  '.....iiiiiiiiiiiiii.....',
  '.....ippsssssssssppi....',
  '.....ippsiinnnniispi....',
  '.....ippsinhhhhnispi....',
  '.....ippsinhhhhnispi....',
  '.....ippsinhnnhnispi....',
  '.....ippsinhhhhnispi....',
  '.....ippsinhhhhnispi....',
  '.....ippsinhhkhnispi....',
  '.....iiiiiiiikkiiiii....',
  '....isssssssskssssssi...',
  '....iiiiiiiiiiiiiiiii...',
]

export const ironFence = [
  'n...n...n...n...n...n...n...n...n...',
  'nn..nn..nn..nn..nn..nn..nn..nn..nn..',
  'nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn',
  'n...n...n...n...n...n...n...n...n...',
  'nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn',
  'n...n...n...n...n...n...n...n...n...',
]

// Three crows on a pasture rail.
export const crowLine = [
  '...hh.......hh.........hh.......',
  '..hhhs.....hhhs.......hhhs......',
  '..hhhh.....hhhh.......hhhh......',
  '...hh.......hh.........hh.......',
  'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
  'b......b......b......b......b...',
  'b......b......b......b......b...',
]

// A glass-sided hearse with its plumed black horse, waiting beside the road.
export const hearse = [
  '.......h.h.h............................',
  '.......h.h.h..............h.............',
  '......hhhhhhhhhhhhh.......hh............',
  '.....hhhhhhhhhhhhhhh.....hhhh...........',
  '.....hsshsshsshsshh.....hhhhh...........',
  '.....hsshsshsshsshh....hhhhhhh..........',
  '.....hsshsshsshsshh...hhhhhhhhhh........',
  '.....hhhhhhhhhhhhhh..hhhhhhhhhhhh.......',
  '.....hhhhhhhhhhhhhhhhhhhhhhhhhhhhh......',
  '......hhhhhhhhhhhhh..hhhhhhhhhhhhh......',
  '.....nnn.......nnn...hh.hh....hh.hh.....',
  '....n.n.n.....n.n.n..hh.hh....hh.hh.....',
  '....nn.nn.....nn.nn..hh.hh....hh.hh.....',
  '....n.n.n.....n.n.n..hh.hh....hh.hh.....',
  '.....nnn.......nnn...hh.hh....hh.hh.....',
]

// A veiled widow with a wreath; a crimson band at the waist.
export const mourner = [
  '..hhh..',
  '.hhhhh.',
  '.hhhhh.',
  '.hhhhh.',
  '.hhhhh.',
  'hhhhhhh',
  'hhhhhhh',
  'hhhkhhh',
  'hhvvvhh',
  'hvhhhvh',
  'hhvvvhh',
  'hhhhhhh',
  'hhhhhhh',
  'hhhhhhh',
]

// A body snatcher opening a grave by lantern light; the sack is already stained.
export const resurrectionists = [
  '...........i..............',
  '..........ipi.............',
  '..........iii.............',
  '.........ihhhi............',
  '........i.hhh.ii..........',
  '..........hhh...b.........',
  '..........h.h....b....iii.',
  'ss.......hh.hh....ss..ili.',
  'sks......bbbbbbbbbbbbbili.',
  'ssss..bbbhhhhhhhhhhhhbbii.',
  'sss..bbhhhhhhhhhhhhhhhbb..',
  '.....bhhhhhhhhhhhhhhhhhb..',
  '......bbbbbbbbbbbbbbbbb...',
]

// A scarecrow in a dead man's coat; the crows no longer mind it.
export const scarecrow = [
  '.....bbbbb.....',
  '...bbbbbbbbb...',
  '.....ppppp.....',
  '.....pipip.....',
  '.....ppppp.....',
  'hs...piiip...sh',
  'hh.....b.....hh',
  'bbbbbbbhhhhbbbb',
  '...h.hhhhhhh.h.',
  '....hhhhhhhhh..',
  '.....hhhhhhh...',
  '.....hhhhhhh...',
  '.....hyhhhyh...',
  '......yyyyy....',
  '.......bbb.....',
  '.......bbb.....',
  '.......bbb.....',
  '.......bbb.....',
]

// The blade is up; the basket is not empty.
export const guillotine = [
  '..bbbbbbbbbbb..',
  '..bb.......bb..',
  '..bb.sssss.bb..',
  '..bb.ssssk.bb..',
  '..bb.sssk..bb..',
  '..bb.ssk...bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bb.......bb..',
  '..bbbbbbbbbbb..',
  '..bb..khk..bb..',
  '..bbbbbbbbbbb..',
  '..bb.......bb..',
  '.bbbb.....bbbb.',
  'bbbbbb...bbbbbb',
  '..kk.bbbbb..kk.',
  '.kbbbbbkkkbbbk.',
  '..bbbbbkkkbbb..',
]

// Leans on a webbed trunk.
export const coffin = [
  '..iii..',
  '.ibbbi.',
  '.ibbbi.',
  'ibbbbbi',
  'ibbbbbi',
  'ibbbbbi',
  'ibbbbbi',
  'ibbbbbi',
  'ibbbbbi',
  '.ibbbi.',
  '.ibbbi.',
  '..iii..',
]

// Grave candles; `l` flames light after dark.
export const candles = [
  '.l....l...l.',
  '.p....p...p.',
  '.p...pp...p.',
  '.p...pp..pp.',
  'pp...pp..pp.',
  'pp...pp..pp.',
]
