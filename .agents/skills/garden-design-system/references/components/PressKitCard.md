# PressKitCard

The image card that the press kit draws on its `note` tab. A page and a stream entry use the same frame and the same type, so a shared entry looks like a shared note.

**Source:** `quartz/components/scripts/press-kit-card/` (`spec.ts` formats, themes and card kinds, `chrome.ts` the frame, `editorial.ts` `drawNote`, `drawQuote` and `drawEntry`, `tractatus.ts` numbered propositions, `typography.ts` fitted text, `math.ts` KaTeX as images, `calendar.ts`, `chart.ts`, `training-card.ts`, `sticker.ts`, `route.ts`, `svg.ts`) · the card's data: `quartz/components/scripts/press-kit/share-subject.ts` (`pageSubject`, `entrySubject`) and `press-kit/tractatus.ts`

## Preview

Still images of the site's canvas output, in a grid of three columns. The first row is the page card on `/stream` in portrait papier, portrait encre and paysage papier. The second row is the stream entry of 2026-08-25 in the same three. The last row is a quote taken by selecting text on `/thoughts/chinese-room`, in portrait papier and portrait encre. The stills come from the dialog in the in-app browser, saved as webp at 1100px on the long side.

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

|              | Page                                            | Stream entry                                                                                                                     |
| ------------ | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Breadcrumb   | `site / …slug`                                  | `site / stream / on / yyyy / mm / dd` (the entry's path; the `?entry=` query is not a segment)                                   |
| Date         | `formatDate` in the site locale, "24 oct. 2025" | the same format from the entry's ISO date, "25 août 2026", read from the string so the visitor's time zone cannot change the day |
| Title        | the page title                                  | the entry heading                                                                                                                |
| Description  | the page description; `description` toggles it  | the entry's `description` meta; always shown, since an entry has no toggle                                                       |
| Tags         | frontmatter `tags`, first 4                     | the entry's tags, first 4                                                                                                        |
| Reading time | the build's reading time                        | words ÷ 200 (reading-time's rate), rounded up                                                                                    |

## Quote and Tractatus

Card kinds are `note`, `quote`, `entry`, `activity`, `day` and `calendar`. A text selection on a page opens the dialog on the `quote` kind. It keeps the frame and replaces the text block:

- **Quote:** a large opening quotation mark in `mark`, the selection in the serif voice at leading 1.28, then the attribution (an em-dash and the page title) in the header voice, and the section in italic when `section` is on. `surligner` paints the selection on a `hl` block.
- **Tractatus:** a selection inside a numbered work draws each proposition on its own row, with its number in the mono face on the left and its text in the body face. Depth indents a row by 0.45 of the text size, at most three steps. The text shrinks to fit. When rows still do not fit, the card shows the first ones and ends with `… +N propositions supplémentaires` (`more propositions` in English).

## Training and Strava

A training day entry and a Strava activity draw the training card (`press-kit-card/training-card.ts`) inside the same frame. The Strava overlays (`grille`, `colonne`, `ticket`, `pastilles`, `bord vertical`, `parcours seul`) are transparent stickers without the frame (`press-kit-card/sticker.ts`).

|              | Training day                                          | Strava activity                                               |
| ------------ | ----------------------------------------------------- | ------------------------------------------------------------- |
| Breadcrumb   | the entry's path                                      | the activity's path, `site / triathlon / on / yyyy / mm / dd` |
| Date         | the day it trained, which can precede the entry's day | the activity's day                                            |
| Tags         | the entry's tags                                      | the sport                                                     |
| Reading time | none                                                  | none                                                          |

Every date on these cards, in the header, on block lines and on the stickers, uses the page format ("10 oct. 2026"), through `cardDate` in `press-kit-text.ts`.

## Link card

The `liens` tab previews what a social site unfurls: the target page's own `og:image` and `og:title`. A stream entry's link unfurls its day page (`/stream/on/yyyy/mm/dd`); an activity's link unfurls its triathlon day page. Each stream archive route has its own social image, with that route's date, tags, reading time and the summaries of its newest three public entries.

## Rules

- Draw cards only with `renderCard`. A second renderer would drift from the site.
- Keep crumbs lowercase. Keep tags as written, with `#`.
- An entry card has no avatar or author line. The breadcrumb identifies the source, as on a page card.
- A breadcrumb is an address. Do not invent segments, such as the sport, that the URL does not have.
