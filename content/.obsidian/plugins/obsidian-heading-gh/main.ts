import { App, MarkdownView, Modal, Notice, Plugin, type Modifier } from "obsidian";

type HeadingItem = { line: number; level: number; text: string };

const LETTER_KEYS = "abcdefghijklmnopqrstuvwxyz".split("");
const SECONDARY_KEYS = [
  "a",
  "s",
  "d",
  "f",
  "l",
  "h",
  "g",
  "u",
  "i",
  "o",
  "p",
  "w",
  "e",
  "r",
  "t",
  "y",
  "c",
  "v",
  "b",
  "n",
  "m",
  "x",
  "z",
];

function parseHeadings(text: string): HeadingItem[] {
  const lines = text.split(/\r?\n/);
  const results: HeadingItem[] = [];
  let inFrontmatter = false;
  let checkedFirstContent = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!checkedFirstContent && trimmed !== "") {
      checkedFirstContent = true;
      if (trimmed === "---") {
        inFrontmatter = true;
        continue;
      }
    }

    if (inFrontmatter) {
      if (trimmed === "---" || trimmed === "...") {
        inFrontmatter = false;
      }
      continue;
    }

    const atx = line.match(/^\s{0,3}(#{1,6})\s*(.+?)\s*#*\s*$/);
    if (atx) {
      const [, hashes, title] = atx;
      results.push({ line: i, level: hashes.length, text: title.trim() });
      continue;
    }

    if (i + 1 < lines.length && trimmed !== "") {
      const underline = lines[i + 1];
      if (/^\s{0,3}=+\s*$/.test(underline)) {
        results.push({ line: i, level: 1, text: line.trim() });
      } else if (/^\s{0,3}-+\s*$/.test(underline)) {
        results.push({ line: i, level: 2, text: line.trim() });
      }
    }
  }

  return results;
}

function collectInitials(text: string): string[] {
  const initials: string[] = [];
  const seen = new Set<string>();
  const add = (raw?: string) => {
    if (!raw) return;
    const ch = raw.match(/[A-Za-z]/)?.[0]?.toLowerCase();
    if (!ch || seen.has(ch)) return;
    seen.add(ch);
    initials.push(ch);
  };

  const trimmed = text.trim();
  const link = trimmed.match(/^\[\[([^[\]]+)\]\]/);
  if (link) {
    const inner = link[1];
    const [target, alias] = inner.split("|");
    add(target);
    target?.split("/").forEach(add);
    add(alias);
  } else {
    add(trimmed);
  }

  return initials;
}

class HeadingNavigatorModal extends Modal {
  private headings: HeadingItem[];
  private itemEls: HTMLDivElement[] = [];
  private hintEls: HTMLSpanElement[] = [];
  private groups = new Map<string, number[]>();
  private secondaryMap = new Map<string, number>();
  private cursor = 0;
  private secondaryActive = false;
  private listEl: HTMLDivElement | null = null;
  private visibleIndices: number[] = [];

  constructor(app: App, private view: MarkdownView, headings: HeadingItem[]) {
    super(app);
    this.headings = headings;
    this.visibleIndices = headings.map((_, idx) => idx);
    this.buildGroups();
  }

  onOpen(): void {
    this.modalEl.addClass("heading-gh-modal");
    const container = this.modalEl.closest(".modal-container");
	const bg = container?.querySelector("modal-bg")
    if (bg) bg.addClass("heading-gh-bg");
    this.contentEl.addClass("heading-gh-body");
    this.renderList();
    this.bindKeys();
    this.highlight(this.cursor);
  }

  onClose(): void {
    this.contentEl.empty();
  }

  private buildGroups() {
    this.visibleIndices.forEach((idx) => {
      const h = this.headings[idx];
      for (const init of collectInitials(h.text)) {
        const existing = this.groups.get(init) ?? [];
        existing.push(idx);
        this.groups.set(init, existing);
      }
    });
  }

  private renderList() {
    const list = this.contentEl.createEl("div", { cls: "heading-gh-list" });
    this.listEl = list;
    this.headings.forEach((h, idx) => {
      const item = list.createEl("div", {
        cls: "heading-gh-item",
        attr: { "data-level": `${h.level}` },
      });
      item.style.paddingLeft = `${Math.max(0, h.level - 1) * 14}px`;

      const hint = item.createEl("span", { cls: "heading-gh-hint" });
      const label = item.createEl("span", {
        cls: "heading-gh-text",
        text: h.text,
      });

      this.itemEls[idx] = item;
      this.hintEls[idx] = hint;
      label.title = h.text;
    });
  }

  private bindKeys() {
    const scope = this.scope;

    const register = (mods: Modifier[], key: string, fn: () => void) => {
      scope.register(mods, key, (evt) => {
        evt?.preventDefault();
        evt?.stopPropagation();
        fn();
        return false;
      });
    };

    register([], "j", () => this.move(1));
    register([], "k", () => this.move(-1));
    register(["Ctrl"], "n", () => this.move(1));
    register(["Ctrl"], "p", () => this.move(-1));
    register([], "Enter", () => this.jump(this.cursor));
    register([], "q", () => this.close());
    register([], "Escape", () => {
      if (this.secondaryActive) {
        this.clearSecondary();
      } else {
        this.close();
      }
    });

    LETTER_KEYS.forEach((ch) => {
      register([], ch, () => {
        if (this.secondaryActive) {
          this.useSecondary(ch);
        } else {
          this.handleInitial(ch);
        }
      });
    });
  }

  private move(delta: number) {
    if (this.visibleIndices.length === 0) return;
    const pos = Math.max(0, this.visibleIndices.indexOf(this.cursor));
    const nextPos = Math.max(0, Math.min(this.visibleIndices.length - 1, pos + delta));
    const target = this.visibleIndices[nextPos];
    if (target === undefined) return;
    this.cursor = target;
    this.highlight(target);
  }

  private highlight(idx: number) {
    if (!this.visibleIndices.includes(idx)) return;
    this.itemEls.forEach((el) => el?.removeClass("is-active"));
    this.itemEls[idx]?.addClass("is-active");
    this.itemEls[idx]?.scrollIntoView({ block: "nearest" });
  }

  private jump(idx: number) {
    if (idx < 0) return;
    const target = this.headings[idx];
    if (!target) return;
    this.close();

    const editor = this.view.editor;
    this.view.leaf?.setViewState(this.view.leaf.getViewState(), { focus: true });
    editor.setCursor({ line: target.line, ch: 0 });
    if (typeof editor.scrollIntoView === "function") {
      editor.scrollIntoView(
        { from: { line: target.line, ch: 0 }, to: { line: target.line, ch: 0 } },
        true
      );
    }
    editor.focus();
  }

  private handleInitial(ch: string) {
    const list = this.groups.get(ch);
    if (!list || list.length === 0) return;
    if (list.length === 1) {
      this.jump(list[0]);
    } else {
      this.enterSecondary(list);
    }
  }

  private enterSecondary(indices: number[]) {
    this.secondaryActive = true;
    this.secondaryMap.clear();
    this.clearSecondaryHints();

    const positions = new Map<number, number>();
    this.visibleIndices.forEach((idx, i) => positions.set(idx, i));
    const mid = this.visibleIndices.length / 2;

    indices
      .slice(0, SECONDARY_KEYS.length)
      .sort((a, b) => {
        const pa = positions.get(a) ?? 0;
        const pb = positions.get(b) ?? 0;
        const da = Math.abs(pa - mid);
        const db = Math.abs(pb - mid);
        return da === db ? pa - pb : da - db;
      })
      .forEach((idx, i) => {
        const key = SECONDARY_KEYS[i];
        this.secondaryMap.set(key, idx);
        this.hintEls[idx]?.setText(`${key}`);
      });
  }

  private useSecondary(ch: string) {
    const idx = this.secondaryMap.get(ch);
    if (idx !== undefined) {
      this.jump(idx);
    }
  }

  private clearSecondary() {
    this.secondaryActive = false;
    this.secondaryMap.clear();
    this.clearSecondaryHints();
  }

  private clearSecondaryHints() {
    this.hintEls.forEach((el) => el?.setText(""));
  }
}

export default class HeadingNavigatorPlugin extends Plugin {
  async onload() {
    this.addCommand({
      id: "heading-gh-navigator",
      name: "Jump to heading (gh)",
      callback: () => this.openNavigator(),
    });
  }

  private openNavigator() {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (!view) {
      new Notice("Open a Markdown file to jump to headings.");
      return;
    }

    const headings = parseHeadings(view.editor.getValue());
    if (headings.length === 0) {
      new Notice("No headings found in this note.");
      return;
    }

    new HeadingNavigatorModal(this.app, view, headings).open();
  }
}
