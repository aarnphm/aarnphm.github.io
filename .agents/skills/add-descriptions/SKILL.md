---
name: add-descriptions
description: Add source-grounded description frontmatter to specified Garden notes that lack it.
---

# Description frontmatter

Resolve the target files or directory from the request. Skip existing descriptions unless regeneration was requested. Read each note's title, metadata, and body before writing its description.

Use the note's actual argument or subject. For a book or sparse note, consult an authoritative source when essential facts are missing. Check the work's identity and edition where relevant. Do not add a claim that the available source cannot support.

Write a concise sentence, usually 10 to 20 words and at most 30 when needed. Match neighboring descriptions, preserve proper names, and avoid generic praise or spoilers. See [examples](references/examples.md) only when the content type needs a model.

Insert `description` after `date` when present, preserving all other frontmatter and body formatting. Quote YAML when its punctuation requires it. Report changed files and any meaningful skips or unresolved source gaps.
