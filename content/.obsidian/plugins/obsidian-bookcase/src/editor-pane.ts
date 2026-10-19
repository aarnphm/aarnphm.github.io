import type { App, WorkspaceLeaf } from 'obsidian'

/**
 * A markdown leaf split off beside the bookcase's own leaf. Every selection reuses it, and the
 * view never closes it: it may hold unsaved edits. Obsidian has no public API to size a split,
 * so it keeps Obsidian's default size.
 */
export class EditorPane {
  private leaf: WorkspaceLeaf | null = null

  constructor(private app: App) {}

  /** True while the leaf is still in the workspace; a closed leaf is forgotten. */
  alive(): boolean {
    const leaf = this.leaf
    if (!leaf) return false
    let found = false
    this.app.workspace.iterateAllLeaves(l => {
      if (l === leaf) found = true
    })
    if (!found) this.leaf = null
    return found
  }

  /**
   * Opens `path` in the leaf, without making it active. The first call splits `host`; with no
   * leaf and no host it does nothing and returns false.
   */
  show(host: WorkspaceLeaf | null, path: string): boolean {
    const file = this.app.vault.getFileByPath(path)
    if (!file) return false
    if (!this.alive()) {
      if (!host) return false
      this.leaf = this.app.workspace.createLeafBySplit(host, 'vertical')
    }
    void this.leaf!.openFile(file, { active: false, state: { mode: 'source' } })
    return true
  }

  /** Moves keyboard focus into the leaf; false when there is none. */
  focus(): boolean {
    if (!this.alive()) return false
    this.app.workspace.setActiveLeaf(this.leaf!, { focus: true })
    return true
  }

  /** Lets go of the leaf without closing it. */
  forget(): void {
    this.leaf = null
  }
}
