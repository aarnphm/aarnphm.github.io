# Library intake

For this chat, a reel, link, or book title requests book identification and addition to `content/library/`. Default status is `to read`. Respect a specified section or reading status. Use this harness to avoid rereading the entire collection. No dependency installation is needed; it uses the repository's js-yaml.

```sh
node harness/library-intake/intake.mjs context
node harness/library-intake/intake.mjs search 'title or author'
node harness/library-intake/intake.mjs render /absolute/path/book.json
```

## Intake procedure

1. Inspect scoped Git changes. Run `context` for current category values and `search` for each title, author, and alternate title. Read matching notes before editing. Preserve reading state, prose, identities, and existing metadata. Check translated titles and series manually as normalized title matching cannot resolve every duplicate.
2. For video, inspect the actual frames, spoken audio, caption, and relevant carousel slides. Record title/author candidates with timestamps or slide numbers and the source URL. Separate visible or spoken evidence from inference. Resolve unclear titles before creating notes; never infer the whole reading list from a thumbnail. If access fails, report the unavailable evidence. Use the Instagram skill's acquisition instructions when needed, with this thread's extraction request taking precedence over its default archive workflow. Archiving the reel is separate from book intake.
3. Verify title, author, publication year, language, description, and edition-specific details against publisher, author, or library catalog sources. Prefer the original publication year unless the note explicitly concerns an edition. Attribute a translator only to a verified edition. Download an actual verified cover into `content/library/posters/` with a descriptive kebab-case filename; inspect the image and use its true file extension. Never substitute generated cover art.
4. Prepare JSON using the fields below. Render validates required metadata, known categories, cover existence, Base exclusion tags, and exact normalized duplicates. It prints Markdown without writing library files. Review successful output, then create `content/library/<id>.md` exclusively (refuse an existing path). For an existing book, edit the existing note directly and preserve its status. No Base edit is needed for ordinary additions.
5. Parse the saved YAML and verify its cover and source links. Confirm the note satisfies the current `content/antilibrary.base` filters: Markdown under library, outside posters, basename other than index, no evergreen/meta tag. The to-read view excludes finished/evergreen status. Bookcase groups by category and reads posters/title/author. If claiming rendered visibility, use the existing watcher and its matching build-ready event. Save repeatable verification evidence under `docs/verification/library-intake/artifacts/`.

## Metadata

Use `id` equal to the filename stem and a faithful `title`. Required populated fields: `author` (list), `year` (integer), `date` (Toronto intake date, YYYY-MM-DD), `description` (brief factual synopsis), `category` (list, established section), `subcategory` (list, specific subjects), `tags` (list), `status`, `posters` (local wikilink), and `external` (verified bibliographic URL). Follow neighboring notes for subject names. All books live directly in library; categories provide sections.

Also include `language`, `translator`, `notebook`, `aliases`, `finished`, and `pdf` when meaningful. Unknown or inapplicable values stay null (aliases may be empty). Never invent a completion year, notebook, translation, PDF, or ISBN to fill a field. These are optional editorial fields in the Base, not requirements for membership. Add verified language when available. Existing status `current` is used in the library and must be preserved.

`body` in input JSON becomes the note body and is removed from frontmatter. Include a short source line with the reel/link and any useful timestamp, plus bibliographic verification links where needed. A title-only request needs the bibliographic source, with no invented recommendation provenance. Keep personal reading commentary intact and avoid writing opinions on Aaron's behalf.

Example input shape (replace placeholders with verified data before rendering):

```json
{
  "id": "Book Title",
  "title": "Book Title",
  "author": ["Author Name"],
  "year": 2000,
  "date": "2026-11-02",
  "description": "Verified short synopsis.",
  "category": ["p/fiction"],
  "subcategory": ["life"],
  "tags": ["literature"],
  "status": "to read",
  "posters": "[[library/posters/book-title.jpg]]",
  "external": "https://publisher.example/book",
  "language": "en",
  "body": "Source: [recommendation](https://example.com/reel), 00:12."
}
```

The harness checks metadata structure and local assets. Source verification and video understanding remain agent tasks. It makes no network requests and never modifies existing notes.
