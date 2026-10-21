import { engineer, student } from './academic-outfits'
import { artist, cinephile } from './creative-outfits'
import { lawyer } from './legal-outfit'
import { part, type PetPen } from './pen'
import { librarian, presenter, reader } from './reading-outfits'
import { baguette, beret, bucketHat, quill, sweatband, wildHair } from './small-props'

export function petOutfits(s: PetPen) {
  const { L, w, paper, ink, pink } = s
  // Worn for walks on the home page's trail, as on the brand card; its pen widths match the pet's
  // line, which is 3.2 hat units on the card.
  const hat = part('hat', bucketHat(L / 3.2, paper, ink, pink), 'pet-hat')
  // Section outfits (quartz/util/section-icon.ts), as on the site icons: the sweatband on
  // triathlon pages, the beret and baguettes on French ones, a quill tucked behind the head on
  // posts, the wig, tie and briefcase on the terms and privacy pages, the palette and brush on
  // quotes and images, glasses and books on bases, a bow tie and pointer on slides, a sheet to read
  // on the stream, a mortarboard, notepad and pencil on courses. `-back` parts sit behind the body.
  const outfit = {
    front: [
      part('outfit-triathlon', sweatband(w, ink, pink)),
      part('outfit-fr', `${baguette([-0.62, 1.2], [1.4, 0.3], 227, w, ink)}${beret(w, ink, ink)}`),
      part('outfit-legal', Object.values(lawyer(w, paper, ink, pink)).join('')),
      part(
        'outfit-art',
        (({ palette, smudge }) => `${smudge}${palette}`)(artist(w, paper, ink, pink)),
      ),
      part('outfit-library', Object.values(librarian(w, paper, ink)).join('')),
      part(
        'outfit-slides',
        Object.values(presenter(w, paper, ink, pink))
          .filter((_, i) => i < 2)
          .join(''),
      ),
      // The glasses are the shared `glasses` part below.
      part('outfit-stream', reader(w, paper, ink, pink).sheet),
      part(
        'outfit-courses',
        (({ cap, pad, pencil }) => `${cap}${pad}<g class="pet-pencil">${pencil}</g>`)(
          student(w, paper, ink, pink),
        ),
      ),
      part(
        'outfit-cinema',
        (({ specs, popcorn, kernel }) => `${specs}${popcorn}<g class="pet-kernel">${kernel}</g>`)(
          cinephile(w, paper, ink, pink),
        ),
      ),
      part(
        'outfit-engineering',
        (({ hardhat, flask, bubbles }) => `${hardhat}${flask}${bubbles}`)(
          engineer(w, paper, ink, pink),
        ),
      ),
    ].join(''),
    back: [
      part('outfit-fr-back', baguette([0.3, 1.02], [1.46, -0.2], 223, w, ink)),
      part('outfit-posts-back', quill([0.42, 0.25], [0.95, -0.5], [1.32, -1.6], w, ink, pink).svg),
      part('outfit-art-back', artist(w, paper, ink, pink).brush),
    ].join(''),
  }
  // Reading glasses for reader mode, the stream and technical pages: the librarian's pair.
  const glasses = part('glasses', librarian(w, paper, ink).glasses)
  // Wild hair on philosophy pages, drawn twice to frizz; behind the body, so the roots tuck in.
  const hair = part(
    'hair',
    [0, 1].map(k => `<g class="pet-hair-frame">${wildHair(w * 1.1, ink, k)}</g>`).join(''),
  )
  return { hat, outfit, glasses, hair }
}
