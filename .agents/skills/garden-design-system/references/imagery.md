# Imagery

The site's brand art is one mascot, drawn in code by `quartz/util/mascot.ts` and rasterised by `quartz/scripts/brand-art.ts`.

- `mascot.svg` (1024 × 1024, transparent): a hand-drawn circle with two dash eyes shifted left, as pressure-width pen ribbons in ink #1c1b1a. Use it for any new art.
- `icon.webp` (1024 × 1024): the same drawing on paper. It is the favicon source, the touch icons and the avatar.
- `og-image.webp` (1200 × 630): the default social card. The mascot climbs a peak toward a `salmon` flag, with a `salmon` pack, a paper bedroll and a bucket hat, against a `secondary` range and a `salmon` sun.

The eyes look left, so scenes rise or move to the left. Draw additions with the same pen functions (`stroke`, `pressure`, `hand`), in ink, `salmon`, `secondary` and paper only. Do not set type over it. The site has no logotype; set the name in the `title` family.

## Section icons

Twelve sections dress the mascot in an outfit. The same drawing serves three places: the favicon of every page in the section (`Head`), the header of the section's generated social cards, and the icon after a link to the site that lands in the section. The site pet wears the outfit on those pages (`data-outfit` on `.pet-svg`). `quartz/util/section-icon.ts` maps a page to its section; `quartz/util/mascot/*-outfit(s).ts` draw the parts; `brand-art.ts` writes `quartz/static/icon-<section>.png`.

| section       | pages                               | outfit                                                                                                                             |
| ------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `triathlon`   | `/triathlon` and the triathlon host | a sweatband                                                                                                                        |
| `fr`          | `/fr`                               | a beret, two baguettes (one behind)                                                                                                |
| `posts`       | `/posts`                            | a quill tucked behind the head                                                                                                     |
| `legal`       | terms of service, privacy policy    | a barrister's wig, a tie, a briefcase                                                                                              |
| `art`         | quotes, vibes                       | a palette in front, a pink-tipped brush over the shoulder, a smudge                                                                |
| `library`     | every Base view                     | round glasses, a stack of books                                                                                                    |
| `slides`      | `<note>/slides/<n>`                 | a bow tie, a pointer at a bar chart                                                                                                |
| `stream`      | `/stream`                           | reading glasses, a dog-eared sheet held up with one line marked                                                                    |
| `courses`     | `/courses`                          | a mortarboard, a spiral notepad, a pencil                                                                                          |
| `engineering` | `thoughts/university`               | a hard hat, an Erlenmeyer flask with bubbles                                                                                       |
| `cinema`      | movies, cinematheque                | thick square specs, a striped bucket of popcorn                                                                                    |
| `embeddings`  | `/embeddings`                       | a spyglass up to the near eye, trained on a constellation of five points (one `salmon`), a rolled chart over the shoulder (behind) |

Routing: a slide page wins, then a named page or path (`PAGES`, `PREFIXES`: the cinematheque is a Base but wears the cinema), then a Base (`library`), then the top-level folder (`FOLDERS`). A page with no section keeps the bare icon.

Rules for a new section:

- Draw the outfit with the mascot's pen in `quartz/util/mascot/`: paper, ink, `salmon` (pink) and `secondary` only. A part that sits behind the body is a `-back` part. Hold a prop at the near eye or the front paw; the mascot looks left, so a prop it looks through or at goes to the left.
- Register the name in `SECTIONS` and its routing table, add the parts to `pet-outfits.ts`, the section to the `@each` list in `quartz/components/styles/pet/outfits.scss`, and the icon to `SECTION_ICONS` in `brand-art.ts`. An outfit that keeps the front paw busy also goes on the `lantern-back` list in `pet/weather.scss`, so the night lantern moves to the other paw.
- Motion on an outfit is small and stepped: the embeddings stars twinkle at 2.6s in five steps, staggered by half a second; the courses pencil and the cinema kernel move the same way. `pet/motion.scss` stops every pet animation under reduced motion. Nothing on an icon moves.
