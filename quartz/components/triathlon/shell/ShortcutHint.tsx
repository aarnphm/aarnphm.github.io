export const ShortcutHint = ({ children }: { children: string }) => (
  <kbd class="tri-key" aria-hidden="true">
    {children}
  </kbd>
)
