---
title: "Runner: A universal interface to run distributed ML application"
tags:
  - technical
  - seed
---

# Runner Interface

This design doc tackles the some current drawbacks from our Runner interface

## Consideration?

- Distributed use case:
	- [Triton Inference Server][#triton-inference-server]
- KServe Predict V2: [link][#predict-v2]

## Q?
---

> How one determine payload format?

> How do we schedule runners?


## Requirements
---
- can have multiple interfaces



###### Appendix
[#triton-inference-server]: https://github.com/triton-inference-server/server
[#predict-v2]: https://kserve.github.io/website/modelserving/inference_api/