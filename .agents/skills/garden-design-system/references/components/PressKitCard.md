# PressKitCard

The image card that the press kit draws on its `note` tab. A page and a stream entry use the same frame and the same type, so a shared entry looks like a shared note.

**Source:** `quartz/components/scripts/press-kit-card.ts` (`renderCard`, `chrome`, `drawNote`, `drawEntry`) · the card's data: `quartz/components/scripts/press-kit.inline.ts` (`pageSubject`, `entrySubject`)

## Preview

Still images of the site's canvas output, taken from the dialog on `/stream` (a page) and on the 2026-08-25 stream entry. The left column is the page card and the right column is the entry card, in portrait papier, portrait encre and paysage papier.

## Frame

Every card except the citation draws the same frame (`chrome`):

- **Header:** the breadcrumb on the left, in `meta-mono` and `darkgray`, with `/` between segments. The last segment sits on a `hl` block (`secondary` on papier, rose on encre). When the trail is too long, middle segments become `…`. The date is on the right, in the same mono and colour.
- **Footer:** one row of boxed `#tag` pills (a `darkgray` frame of 1.5px or more, at most 4 tags), with the reading time on the right, aligned to the pill text.
- **Rule:** a 2px line above the footer, in the theme's `rule` colour, spans the text width.
- The `méta` include hides the date, the tags and the reading time. The breadcrumb and the rule stay.

## Text

- **Title:** the header face (`--headerFont`), leading 1.04, sized between the format's title range (40–76px on paysage, 56–116px on portrait).
- **Description:** the body face in italic, leading 1.3, under the title.
- **Body:** the body face, leading 1.42. A note's excerpt is `darkgray` and shows only when `extrait` is on. An entry's body is its post, so it is always shown and stays in ink.
- The text block sits low on the taller formats and centred on paysage.

## Page and entry

| | Page | Stream entry |
| --- | --- | --- |
| Breadcrumb | `site / …slug` | `site / stream / on / yyyy / mm / dd` (the entry's path; the `?entry=` query is not a segment) |
| Date | `formatDate` in the site locale, "24 oct. 2025" | the same format from the entry's ISO date, "25 août 2026", read from the string so the visitor's time zone cannot change the day |
| Title | the page title | the entry heading |
| Description | the page description; `description` toggles it | the entry's `description` meta; always shown, since an entry has no toggle |
| Tags | frontmatter `tags`, first 4 | the entry's tags, first 4 |
| Reading time | the build's reading time | words ÷ 200 (reading-time's rate), rounded up |

## Training and Strava

A training day entry and a Strava activity draw the training card (`press-kit-training-card.ts`) inside the same frame. The Strava overlays (`grille`, `colonne`, `ticket`, `pastilles`, `bord vertical`, `parcours seul`) are transparent stickers without the frame (`press-kit-sticker.ts`).

| | Training day | Strava activity |
| --- | --- | --- |
| Breadcrumb | the entry's path | the activity's path, `site / triathlon / on / yyyy / mm / dd` |
| Date | the day it trained, which can precede the entry's day | the activity's day |
| Tags | the entry's tags | the sport |
| Reading time | none | none |

Every date on these cards, in the header, on block lines and on the stickers, uses the page format ("10 oct. 2026"), through `cardDate` in `press-kit-text.ts`.

## Link card

The `liens` tab previews what a social site unfurls: the target page's own `og:image` and `og:title`. A stream entry's link unfurls its day page (`/stream/on/yyyy/mm/dd`); an activity's link unfurls its triathlon day page. Each stream archive route has its own social image, with that route's date, tags, reading time and the summaries of its newest three public entries.

## Rules

- Draw cards only with `renderCard`. A second renderer would drift from the site.
- Keep crumbs lowercase. Keep tags as written, with `#`.
- An entry card has no avatar or author line. The breadcrumb identifies the source, as on a page card.
- A breadcrumb is an address. Do not invent segments, such as the sport, that the URL does not have.
