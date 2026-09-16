export function ArenaReaderLink() {
  return (
    <a
      href="/arena/feed"
      class="arena-reader-toggle internal"
      aria-label="Open reader"
      title="Open reader"
      data-no-popover
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        aria-hidden="true"
        focusable="false"
      >
        <rect x="4" y="3" width="16" height="18" />
        <path d="M8 7h8M8 11h8M8 15h8M8 18h5" />
      </svg>
    </a>
  )
}
