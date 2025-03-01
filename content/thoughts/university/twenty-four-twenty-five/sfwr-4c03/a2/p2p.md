---
id: p2p
tags:
  - sfwr4c03
  - networking
date: "2025-02-28"
description: and file sharing
modified: 2025-02-28 12:19:55 GMT-05:00
title: p2p
---

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/initial-p2p.webp|Initial P2P]]

By default, we can observe that the ports are being populated correctly.

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/p2p-diff-ports.webp]]

When we changed the ports to `9999`, we observe that it still works as expected

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/ignored-files.webp]]

Note that in this test case, we make sure that the files with ignored extensions won't get transfered.

---

Additional features for `fileSynchronizer.py`:

- graceful termination
- logger

![[thoughts/university/twenty-four-twenty-five/sfwr-4c03/a2/graceful-handling.webp]]
