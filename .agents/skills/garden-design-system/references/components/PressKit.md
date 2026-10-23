# PressKit

The share dialog for the current page: an image card in four formats and four themes, a citation and links.

**Source:** `quartz/components/PressKit.tsx` (the dialog shell and tabs), `PressKitEditor.tsx` (the `texte`, `attribution` and `inclure` controls) and `PressKitButtons.tsx` · script: `quartz/components/scripts/press-kit.inline.ts` with `scripts/press-kit/` (`dialog-events.ts` keys and opening, `control-view.ts` control state, `picture-sharing.ts` drawing and download, `link-sharing.ts` the `liens` tab, `share-subject.ts` page and entry data, `tractatus.ts`, `training-controls.ts`, `session-state.ts`) · styles: `quartz/components/styles/pressKit.scss` and `styles/pressKit/` (`controls.scss`, `data-grid.scss`, `stage.scss`)

## Preview

A copy of the server-rendered markup from `/` (selector `dialog.press-kit`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The dialog is shown open without the modal backdrop, and the canvas card is a still image. The preview is static. The tab shown is `note`, with the page card from `/`.

## Rules

- `g` then `s` opens the dialog on any page. It is a native modal `<dialog>`; Escape closes it.
- The dialog is in French on purpose (`lang="fr"`): `kit de presse`, `carré`, `papier`, `télécharger`.
- Tabs: `note` (the image card), `citation` (the same card for a text selection) and `liens`. Arrow keys and Home/End move between tabs; `1` to `3` jump.
- On the note and citation tabs, which both draw an image, `f` and `t` cycle the format (story 9:16, portrait 4:5, carré 1:1, paysage 1.91:1) and the theme (papier, encre, rose, sauge). `d` downloads and `c` copies the image.
- The `inclure` row holds one toggle per part of the card (`description`, `extrait`, `méta`, `illustration`). `réglages avancés` opens a details block with the editable text, the attribution and a language switch.
- The anchor picker (`ancre`: the page or one of its sections) is the shared Select control. The press kit is its first consumer outside triathlon.
- The footer lists the keys in `meta-mono`.
- The image card itself, its frame and its text, is PressKitCard. A stream entry opens the same dialog (`kit de presse · entrée`) and draws the same card.
