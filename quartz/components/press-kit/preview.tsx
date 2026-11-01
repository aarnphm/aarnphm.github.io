export const Preview = ({ site }: { site: string }) => (
  <figure class="press-kit-stage">
    <div
      class="press-kit-viewport"
      data-for="image"
      data-site-cursor-native
      tabIndex={0}
      role="region"
      aria-label="aperçu de l’image"
      aria-description="Défiler pour explorer. + et − pour zoomer ; 0 pour ajuster."
    >
      <div class="press-kit-frame">
        <canvas class="press-kit-canvas" role="img" aria-label="aperçu de la carte" />
      </div>
    </div>
    <div class="press-kit-preview-zoom" data-for="image" role="group" aria-label="zoom de l’aperçu">
      <button type="button" data-preview-zoom="out" aria-label="réduire l’aperçu">
        −
      </button>
      <output data-preview-percent aria-label="zoom de l’aperçu">
        —
      </output>
      <button type="button" data-preview-zoom="in" aria-label="agrandir l’aperçu">
        +
      </button>
      <button
        type="button"
        data-preview-zoom="actual"
        aria-label="taille réelle (100 %)"
        aria-pressed="false"
      >
        100%
      </button>
      <button
        type="button"
        data-preview-zoom="fit"
        aria-label="ajuster l’aperçu"
        aria-pressed="true"
      >
        ajuster
      </button>
    </div>
    <div class="press-kit-link-frame" data-for="links">
      <div class="press-kit-unfurl" hidden>
        <img class="press-kit-unfurl-image" alt="" decoding="async" data-ignore-popup />
        <div class="press-kit-unfurl-text">
          <span class="press-kit-unfurl-site">{site}</span>
          <span class="press-kit-unfurl-title" />
        </div>
      </div>
    </div>
    <figcaption class="press-kit-caption" aria-live="polite" />
  </figure>
)
