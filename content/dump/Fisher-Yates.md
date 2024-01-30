---
id: Fisher-Yates
tags:
  - seed
date: "2024-01-30"
title: Fisher-Yates
---

Pseudocode:

```prolog
Write down the numbers from 1 through N.
Pick a random number k between one and the number of unstruck numbers remaining (inclusive).
Counting from the low end, strike out the kth number not yet struck out, and write it down at the end of a separate list.
Repeat from step 2 until all the numbers have been struck out.
The sequence of numbers written down in step 3 is now a random permutation of the original numbers.
```

Implementation of modern Fisher-Yates algorithm

```js
function sample(obj, n, guard) {
  if (n == null || guard) {
    if (!isArrayLike(obj)) obj = values(obj);
    return obj[random(obj.length - 1)];
  }
  var sample = toArray(obj);
  var length = getLength(sample);
  n = Math.max(Math.min(n, length), 0);
  var last = length - 1;
  for (var index = 0; index < n; index++) {
    var rand = random(index, last);
    var temp = sample[index];
    sample[index] = sample[rand];
    sample[rand] = temp;
  }
  return sample.slice(0, n);
}
  ````
