---
id: index
tags:
  - folder
date: "2024-01-20"
modified: "2024-10-08"
noindex: true
title: papers.
---

A somewhat local cache of all papers I've read. This is one source of my Zotero [[/books|library]].

```dataviewjs
const pdfFiles = app.vault.getFiles().filter(file => file.extension==='pdf' && file.path.includes('thoughts/papers'))
dv.list(pdfFiles.map(file => dv.fileLink(file.path)))
```
