// The probe's selection tab: select text in a passage and see it as Gemma 4 E4B reads it, from
// the bake (quartz/scripts/latent/plt_bake.py): per token, the strongest transcoder features at
// each of the 42 layers and the positions it attends to at the global-attention layers. Gemma
// never runs here; localhost reads the bake from plt_serve.py, the site from
// /models/latent-plt/bake/. Highlights use the CSS Custom Highlight API, so the reader's DOM is
// never written (site :has() rules make small DOM writes restyle the whole page).

export { bakeOf, glossOf, labelOf } from './gemma/bake'
export type { Label } from './gemma/bake'
export { clearHighlights, markPassage } from './gemma/highlights'
export { esc, ll } from './gemma/markup'
export { bindGemma, probeTab } from './gemma/selection'
