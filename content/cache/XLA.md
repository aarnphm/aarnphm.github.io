---
title: "XLA"
tags:
  - seed
  - machinelearning
---

- Accelerated Algebra
- Developed from Tensorflow

```python
def calc(x,y,z):
    return tf.reduce_sum(x+y*z)
```

Optimise compute graph via single kernel launch vs. launching three separate kernel

