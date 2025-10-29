---
date: "2024-10-11"
id: infinite poem
modified: 2025-10-29 02:14:21 GMT-04:00
tags:
  - seed
title: infinite poem
---

```js
const rules = {
  start: "$line1\n    $line2\n$line3\n  $line4\n$line5",
  line1: "What shall a $dog_breed do?",
  line2: "$verbs through the $nature_place,",
  line3: "Then she $verbs her $dog_feature.",
  line4: "$human_action, I $human_verb",
  line5: "This $adj $noun of $emotion.",
  dog_breed: "labrador (4) | terrier | shepherd | beagle | poodle",
  dog_feature: "floppy ears | wagging tail | wet nose | playful eyes | soft fur",
  verbs: "runs | leaps | bounds | trots | dashes",
  nature_place: "meadow | forest | garden | park | beach",
  human_action: "Watching | Smiling | Laughing | Wondering | Marveling",
  human_verb: "contemplate | ponder | appreciate | cherish | admire",
  adj: "simple | joyful | precious | fleeting | eternal",
  noun: "moment | bond | connection | friendship | companionship",
  emotion: "love | happiness | wonder | gratitude | peace",
}

// Generate and print the poem 5 times
for (let i = 0; i < 10; i++) {
  console.log(`Poem ${i + 1}:`)
  console.log(RiTa.grammar(rules).expand())
  console.log() // Add a blank line between poems
}
```
