---
name: garden-add-place
description: Add or update a physical place in Garden's Arena places channel and places-to-go Base from a place name, Google Maps link, or official website, with verified branch identity, address, and coordinates.
---

# Garden add place

Work in `/Users/aarnphm/workspace/garden` unless the current Garden checkout is elsewhere. A request to save a place authorizes both local catalogue edits. This skill works with Luna or any other model; it does not change model settings. Read the checkout's current AGENTS.md and inspect scoped Git state before editing.

## Resolve the location

Accept a place name plus supplied context, a Google Maps link, or a website. Resolve the physical branch before writing. Use city/address details from the request and conversation. If a name or website identifies several plausible branches, ask for the missing city or branch. Do not choose an unrelated business just because its domain matches the name.

For a Maps short link, follow its redirect with the browser or bounded HTTP request. Confirm the place name and address. In a resolved Google Maps place URL, `!3dLAT!4dLON` is the place pin; the `@LAT,LON` pair is the viewport center and can differ. Prefer the verified pin. For website/name input, find the matching branch's Maps listing or another authoritative geocoded source, then cross-check the address. Record source URLs and coordinate provenance in the verification artifact.

Use `[latitude, longitude]` in both catalogues, preserving the verified precision. Check finite values, latitude within -90..90 and longitude within -180..180. Never use a city center, an unrelated branch, or invented coordinates. If the place is identified but coordinates remain unavailable, save the confirmed information without a pin and report it as unpinned. If branch identity remains ambiguous, hold both edits until clarified.

## Dedupe before adding

Search `content/are.na.md` and `content/places/*.md` by name, address, original URL, resolved Maps place ID, and coordinates. Different Maps short links or a Maps link plus website can identify the same branch. Reuse its existing Arena item and note path; a second branch gets a location-qualified title. Preserve IDs, ratings, aliases, categories, metadata, and authored bodies. Patch only confirmed missing or corrected fields. If only one catalogue contains the place, create the missing counterpart.

## Update both catalogues

Inspect the current `## places` section and `content/places to go.base`; use their live schemas. Insert a new place at the top of that section after any channel-level metadata, keeping neighbouring items intact. Store the supplied Maps link when available. For a website or name, choose the verified branch Maps link or official branch website. Use the same primary URL in the Arena item and note; secondary source links may be nested under the Arena item.

Arena uses this grammar (the coordinates are placeholders to replace with verified numbers):

```markdown
- https://example.com/branch -- Place Name Branch
  - [meta]:
    - date: MM/DD/YYYY
    - tags: [coffee shop]
    - coord: [LATITUDE, LONGITUDE]
    - address: Verified street address, city, region, postal code, country
    - later: true
```

Use the current America/Toronto calendar date when no saved date is supplied. New saves default to `later: true`; preserve an existing visit/read state. Choose one or more existing place-type tags based on what the location is, such as `coffee shop`, `restaurant`, `bar`, `theatre`, or `market`. These become the note's `type` list. Avoid adding city tags as place types just for discovery.

Create or update `content/places/<existing path or branch-qualified title>.md` with:

- `title` and `id`: the branch-qualified place name for a new note; preserve an existing identity.
- `category: ['[[places to go]]']`: this exact link satisfies the Base filter; retain additional existing categories.
- `coordinates`: a two-element YAML list of latitude then longitude, using the same values as Arena `coord`.
- `address` and `url`: the verified address and the same primary URL as the Arena entry.
- `type`: a list matching the Arena place-type tags; `tags: [places]` for new notes.
- `date`: ISO YYYY-MM-DD; `modified`: update only when the note changes.
- `description`: one short, source-backed sentence about the place.

Add a personal `rating` only when supplied. A Maps star rating is not Aaron's rating. Leave unknown hours, prices, accessibility, or reviews out. Keep visit notes supplied by Aaron in the note body.

The existing `quartz/scripts/sync-places.ts` is a whole-channel migration and supplies a default rating of 5. Do not run it on the real checkout for a single-place request: it can touch unrelated notes and invent a personal rating. Make the two scoped edits directly. No Base schema change is needed for ordinary additions.

## Verify and report

Run the repeatable production parser/render check from the repository root:

```sh
pnpm exec tsx quartz/scripts/verify-place.ts 'content/places/Place Name Branch.md'
pnpm exec oxfmt --check 'content/places/Place Name Branch.md'
git diff --check
```

The verifier reads the actual Arena section and note, runs the Arena transformer, renders the actual places Base, and records the matched item and map marker under `docs/verification/garden-add-place/artifacts/`. Inspect the scoped diff for duplicate items and unrelated edits. Avoid formatting the entire Arena file.

If browser proof is requested or needed for a rendering issue, use the existing watcher, wait for its matching `build:ready`, confirm HTTP 200, and inspect the Arena item and places map. Preserve existing processes. Production parser/render artifacts do not prove the live page; report a watcher or emission blocker precisely.

Finish with the place name/branch, verified address and coordinates, clickable paths for both catalogue entries, and the checks passed. State whether it was newly added, updated, already present, or saved without a verified pin.
