const ZoomIcon = ({ direction }: { direction: 'in' | 'out' }) => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.8"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <circle cx="10.5" cy="10.5" r="7.5" />
    <path d="m16 16 5 5M7.5 10.5h6" />
    {direction === 'in' && <path d="M10.5 7.5v6" />}
  </svg>
)

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
    <details class="press-kit-preview-tools" open>
      <summary>
        aperçu
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          aria-hidden="true"
        >
          <path d="m6 15 6-6 6 6" />
        </svg>
      </summary>
      <div
        class="press-kit-preview-zoom"
        data-for="image"
        role="group"
        aria-label="zoom de l’aperçu"
      >
        <button
          type="button"
          data-preview-zoom="out"
          aria-label="réduire l’aperçu"
          title="réduire l’aperçu"
        >
          <ZoomIcon direction="out" />
        </button>
        <output data-preview-percent aria-label="zoom de l’aperçu">
          —
        </output>
        <button
          type="button"
          data-preview-zoom="in"
          aria-label="agrandir l’aperçu"
          title="agrandir l’aperçu"
        >
          <ZoomIcon direction="in" />
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
          title="ajuster l’aperçu"
          aria-pressed="true"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" />
          </svg>
        </button>
      </div>
      <div class="press-kit-caption" aria-live="polite" />
    </details>
    <div class="press-kit-link-frame" data-for="links">
      <div class="press-kit-unfurl" hidden>
        <img class="press-kit-unfurl-image" alt="" decoding="async" data-ignore-popup />
        <div class="press-kit-unfurl-text">
          <span class="press-kit-unfurl-site">{site}</span>
          <span class="press-kit-unfurl-title" />
        </div>
      </div>
    </div>
  </figure>
)
