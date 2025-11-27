import { Notice, Plugin, TFile, parseYaml } from "obsidian";

// Minimal grammar dispatcher; future grammars can be added by name.
type GrammarKind = "stream" | "channel";

type Lines = string[];

type Modification = { index: number; payload: string[] };

function extractLiteralKeys(ebnf: string, rule: string): string[] {
  const re = new RegExp(`^\\s*${rule}\\s*=([^;]+);`, "m");
  const m = ebnf.match(re);
  if (!m) return [];
  const rhs = m[1];
  const keys: string[] = [];
  const litRe = /"([^"]+)"/g;
  let lm: RegExpExecArray | null;
  while ((lm = litRe.exec(rhs)) !== null) keys.push(lm[1].trim());
  return keys;
}

function detectGrammar(ebnf: string): GrammarKind | null {
  const firstRule = ebnf
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line.length > 0);
  if (!firstRule) return null;
  const match = firstRule.match(/^([a-zA-Z][\w-]*)\s*=/);
  if (!match) return null;
  const name = match[1].toLowerCase();
  if (name === "stream") return "stream";
  if (name === "channel") return "channel";
  return null;
}

function deriveRequiredKeys(ebnf: string, grammar: GrammarKind): Set<string> {
  const required = new Set<string>(["date", "tags"]);
  if (grammar === "stream") return required;
  for (const key of extractLiteralKeys(ebnf, "key")) {
    if (key === "date" || key === "tags") required.add(key);
  }
  return required;
}

function extractEbnfFromFrontmatter(text: string): string | null {
  const fmMatch = text.match(/^---\s*\n([\s\S]*?)\n(?:---|\.\.\.)\s*\n/);
  if (!fmMatch) return null;
  const raw = fmMatch[1];
  const data = parseYaml(raw) as Record<string, unknown> | null;
  if (!data || typeof data !== "object") return null;
  const metadata = (data as Record<string, unknown>).metadata as
    | Record<string, unknown>
    | undefined;
  const ebnf = metadata?.ebnf;
  return typeof ebnf === "string" ? ebnf : null;
}

function findFrontmatterEnd(lines: Lines): number {
  if (lines.length === 0) return -1;
  if (lines[0].trim() !== "---") return -1;
  for (let i = 1; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === "---" || t === "...") return i;
  }
  return -1;
}

function formatStreamTimestamp(now = new Date()): string {
  const pad = (n: number) => n.toString().padStart(2, "0");
  const offsetMinutes = -now.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const hours = pad(Math.floor(abs / 60));
  const minutes = pad(abs % 60);
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())} GMT${sign}${hours}:${minutes}`;
}

function formatArenaDate(now = new Date()): string {
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(now.getMonth() + 1)}/${pad(now.getDate())}/${now.getFullYear()}`;
}

function indentOf(line: string | undefined): string {
  if (!line) return "";
  const m = line.match(/^\s*/);
  return m ? m[0] : "";
}

function applyModifications(lines: Lines, modifications: Modification[]) {
  modifications
    .sort((a, b) => b.index - a.index)
    .forEach((mod) => lines.splice(mod.index, 0, ...mod.payload));
}

function trimTrailingEmptyLines(lines: Lines) {
  while (lines.length > 0 && lines[lines.length - 1].trim() === "") {
    lines.pop();
  }
}

type StreamTokenKind =
  | "heading"
  | "metaStart"
  | "metaDate"
  | "metaTags"
  | "metaOther"
  | "tagItem"
  | "separator"
  | "blank"
  | "text";

type StreamToken = { kind: StreamTokenKind; line: number };

function lexStream(lines: Lines, start: number, end: number): StreamToken[] {
  const tokens: StreamToken[] = [];
  for (let i = start; i <= end; i++) {
    const raw = lines[i] ?? "";
    const trimmed = raw.trim();
    if (trimmed === "---") tokens.push({ kind: "separator", line: i });
    else if (/^##\s+/.test(trimmed)) tokens.push({ kind: "heading", line: i });
    else if (/^\s*-\s*\[meta\]\s*:/i.test(trimmed)) tokens.push({ kind: "metaStart", line: i });
    else if (/^\s{2,}-\s*date\s*:/i.test(raw)) tokens.push({ kind: "metaDate", line: i });
    else if (/^\s{2,}-\s*tags\s*:/i.test(raw)) tokens.push({ kind: "metaTags", line: i });
    else if (/^\s{2,}-\s*.+?:/.test(raw)) tokens.push({ kind: "metaOther", line: i });
    else if (/^\s{4,}-\s*.+/.test(raw)) tokens.push({ kind: "tagItem", line: i });
    else if (trimmed === "") tokens.push({ kind: "blank", line: i });
    else tokens.push({ kind: "text", line: i });
  }
  return tokens;
}

class StreamParser {
  constructor(
    private lines: Lines,
    private tokens: StreamToken[],
    private requiredKeys: Set<string>
  ) {}

  enforce(): Modification[] {
    const mods: Modification[] = [];
    let idx = 0;
    while (idx < this.tokens.length) {
      const { nextIdx, modifications } = this.parseSection(idx);
      mods.push(...modifications);
      idx = nextIdx;
    }
    return mods;
  }

  private parseSection(start: number): { nextIdx: number; modifications: Modification[] } {
    const modifications: Modification[] = [];
    let idx = start;

    while (this.tokens[idx]?.kind === "blank") idx += 1;

    if (this.tokens[idx]?.kind === "separator") idx += 1; // skip stray separator

    if (this.tokens[idx]?.kind === "heading") idx += 1;

    while (this.tokens[idx]?.kind === "blank") idx += 1;

    // parse meta block
    const metaStartTok = this.tokens[idx];
    if (!metaStartTok || metaStartTok.kind !== "metaStart") {
      const insertAt = this.tokens[idx]?.line ?? this.lines.length;
      modifications.push({
        index: insertAt,
        payload: StreamParser.defaultMetaBlock(this.lines[insertAt], this.requiredKeys),
      });
      // Skip ahead to next separator or end
      while (idx < this.tokens.length && this.tokens[idx].kind !== "separator") idx += 1;
      return { nextIdx: idx, modifications };
    }

    idx += 1;
    const metaBodyStart = metaStartTok.line + 1;
    let hasDate = false;
    let hasTags = false;
    let metaBodyEnd = metaBodyStart;

    while (idx < this.tokens.length) {
      const tok = this.tokens[idx];
      if (tok.kind === "metaDate") hasDate = true;
      else if (tok.kind === "metaTags") hasTags = true;
      else if (tok.kind !== "metaOther" && tok.kind !== "tagItem" && tok.kind !== "blank") break;
      metaBodyEnd = tok.line;
      idx += 1;
    }

    if (this.requiredKeys.has("date") && !hasDate) {
      modifications.push({ index: metaBodyStart, payload: [`  - date: ${formatStreamTimestamp()}`] });
      metaBodyEnd += 1;
    }
    if (this.requiredKeys.has("tags") && !hasTags) {
      const insertionPoint = hasDate ? metaBodyEnd + 1 : metaBodyStart + 1;
      modifications.push({ index: insertionPoint, payload: ["  - tags:", "    - fruit"] });
    } else if (hasTags) {
      // ensure at least one tag item exists after tags:
      const tagLine = this.findLine("metaTags", metaBodyStart, metaBodyEnd);
      if (tagLine !== -1 && !this.hasTagItems(tagLine + 1, metaBodyEnd)) {
        modifications.push({ index: tagLine + 1, payload: ["    - fruit"] });
      }
    }

    while (idx < this.tokens.length && this.tokens[idx].kind !== "separator") idx += 1;
    return { nextIdx: idx, modifications };
  }

  private findLine(kind: StreamTokenKind, startLine: number, endLine: number): number {
    for (const tok of this.tokens) {
      if (tok.line < startLine || tok.line > endLine) continue;
      if (tok.kind === kind) return tok.line;
    }
    return -1;
  }

  private hasTagItems(startLine: number, endLine: number): boolean {
    return this.tokens.some(
      (tok) => tok.kind === "tagItem" && tok.line >= startLine && tok.line <= endLine
    );
  }

  private static defaultMetaBlock(
    nextLine: string | undefined,
    requiredKeys: Set<string>
  ): string[] {
    const block = ["- [meta]:"];
    if (requiredKeys.has("date")) block.push(`  - date: ${formatStreamTimestamp()}`);
    if (requiredKeys.has("tags")) {
      block.push("  - tags:");
      block.push("    - fruit");
    }
    if (nextLine !== undefined && nextLine.trim() !== "") block.push("");
    return block;
  }
}

// Earley parser utilities
type Production = { lhs: string; rhs: string[] };
type CFG = { start: string; productions: Production[] };
type EarleyState = { lhs: string; rhs: string[]; dot: number; start: number };

function arenaCfg(): CFG {
  return {
    start: "Document",
    productions: [
      { lhs: "Document", rhs: ["Channel", "Document"] },
      { lhs: "Document", rhs: ["Channel"] },
      { lhs: "Channel", rhs: ["channel", "ChannelBody"] },
      { lhs: "ChannelBody", rhs: ["ChannelMeta", "Blocks"] },
      { lhs: "ChannelBody", rhs: ["Blocks"] },
      { lhs: "ChannelMeta", rhs: ["channelMeta", "MetaPairs"] },
      { lhs: "Blocks", rhs: ["Block", "Blocks"] },
      { lhs: "Blocks", rhs: ["Block"] },
      { lhs: "Block", rhs: ["item", "MetaSection", "Notes"] },
      { lhs: "MetaSection", rhs: ["metaStart", "MetaPairs"] },
      { lhs: "MetaPairs", rhs: ["metaPair", "MetaPairs"] },
      { lhs: "MetaPairs", rhs: ["metaPair"] },
      { lhs: "Notes", rhs: ["note", "Notes"] },
      { lhs: "Notes", rhs: [] }, // epsilon
    ],
  };
}

function earleyAccept(grammar: CFG, tokens: string[]): boolean {
  const nonterminals = new Set(grammar.productions.map((p) => p.lhs));
  const prodMap = new Map<string, string[][]>();
  for (const prod of grammar.productions) {
    const list = prodMap.get(prod.lhs) ?? [];
    list.push(prod.rhs);
    prodMap.set(prod.lhs, list);
  }

  const chart: EarleyState[][] = Array.from({ length: tokens.length + 1 }, () => []);
  const added: Array<Set<string>> = Array.from({ length: tokens.length + 1 }, () => new Set());
  const startState: EarleyState = { lhs: "$start", rhs: [grammar.start], dot: 0, start: 0 };

  const key = (st: EarleyState) => `${st.lhs}->${st.rhs.join(" ")}·${st.dot}@${st.start}`;
  const pushState = (idx: number, state: EarleyState) => {
    const k = key(state);
    if (added[idx].has(k)) return;
    added[idx].add(k);
    chart[idx].push(state);
  };

  pushState(0, startState);

  for (let i = 0; i <= tokens.length; i++) {
    for (let s = 0; s < chart[i].length; s++) {
      const state = chart[i][s];
      const atEnd = state.dot >= state.rhs.length;
      if (!atEnd) {
        const nextSym = state.rhs[state.dot];
        if (nonterminals.has(nextSym)) {
          for (const rhs of prodMap.get(nextSym) ?? []) {
            pushState(i, { lhs: nextSym, rhs, dot: 0, start: i });
          }
        } else if (tokens[i] === nextSym) {
          pushState(i + 1, { ...state, dot: state.dot + 1 });
        }
      } else {
        for (const st of chart[state.start]) {
          if (st.dot >= st.rhs.length) continue;
          if (st.rhs[st.dot] === state.lhs) pushState(i, { ...st, dot: st.dot + 1 });
        }
      }
    }
  }

  return chart[tokens.length].some(
    (st) => st.lhs === "$start" && st.dot === st.rhs.length && st.start === 0
  );
}

function validateArenaWithEarley(lines: Lines, bodyStart: number): boolean {
  const rawTokens = lexArena(lines, bodyStart).filter((t) => t.kind !== "blank");
  const blockItems = computeBlockItems(rawTokens);
  const tokens: string[] = [];
  let insideChannelMeta = false;
  for (const tok of rawTokens) {
    let kind = tok.kind;
    if (kind === "item" && !blockItems.has(tok.line)) kind = "note";

    if (kind === "channel") {
      insideChannelMeta = false;
      tokens.push(kind);
      continue;
    }
    if (kind === "channelMeta") {
      insideChannelMeta = true;
      tokens.push(kind);
      continue;
    }
    if (insideChannelMeta && kind === "item") insideChannelMeta = false;
    if (insideChannelMeta && kind === "metaStart") continue; // tolerate legacy nested meta
    tokens.push(kind);
  }
  return earleyAccept(arenaCfg(), tokens);
}

function computeBlockItems(tokens: ArenaToken[]): Set<number> {
  const blockLines = new Set<number>();
  let currentChannel: number | null = null;
  let baseline: number | null = null;
  for (const tok of tokens) {
    if (tok.kind === "channel") {
      currentChannel = tok.line;
      baseline = null;
      continue;
    }
    if (tok.kind !== "item" || currentChannel === null) continue;
    if (baseline === null) baseline = tok.indent;
    if (tok.indent === baseline) blockLines.add(tok.line);
  }
  return blockLines;
}

type ArenaTokenKind =
  | "channel"
  | "item"
  | "channelMeta"
  | "metaStart"
  | "metaPair"
  | "note"
  | "blank";

type ArenaToken = { kind: ArenaTokenKind; line: number; indent: number };

function lexArena(lines: Lines, start: number): ArenaToken[] {
  const tokens: ArenaToken[] = [];
  for (let i = start; i < lines.length; i++) {
    const raw = lines[i] ?? "";
    const trimmed = raw.trim();
    const indent = raw.match(/^\s*/)?.[0].length ?? 0;
    if (/^##\s+/.test(trimmed)) tokens.push({ kind: "channel", line: i, indent });
    // channel-level meta: no leading spaces
    else if (/^-\s+\[meta\]\s*:/i.test(raw)) tokens.push({ kind: "channelMeta", line: i, indent });
    else if (/^\s{2}-\s+\[meta\]\s*:/i.test(raw)) tokens.push({ kind: "metaStart", line: i, indent });
    else if (/^\s{2,}-\s+.+?:/.test(raw)) tokens.push({ kind: "metaPair", line: i, indent });
    else if (/^\s{2}-\s+.+/.test(raw)) tokens.push({ kind: "note", line: i, indent });
    // item (one or more leading spaces, but not meta markers)
    else if (/^\s*-\s+/.test(raw)) tokens.push({ kind: "item", line: i, indent });
    else if (trimmed === "") tokens.push({ kind: "blank", line: i, indent });
  }
  return tokens;
}

class ArenaParser {
  private blockItems: Set<number>;

  constructor(
    private lines: Lines,
    private tokens: ArenaToken[],
    private requiredKeys: Set<string>
  ) {
    this.blockItems = computeBlockItems(tokens);
  }

  enforce(): Modification[] {
    const mods: Modification[] = [];
    for (const block of this.extractBlocks()) {
      if (block.metaStarts.length === 0) {
        mods.push({
          index: block.itemLine + 1,
          payload: this.defaultMetaBlock(block.itemLine),
        });
        continue;
      }

      const missing = [...this.requiredKeys].filter((k) => !block.metaPairs.has(k));
      if (missing.length > 0) {
        mods.push({
          index: block.metaStarts[0] + 1,
          payload: this.missingPayload(block.metaStarts[0], missing),
        });
      }
    }
    return mods;
  }

  private extractBlocks(): Array<{
    itemLine: number;
    endLine: number;
    metaStarts: number[];
    metaPairs: Set<string>;
  }> {
    const blocks: Array<{
      itemLine: number;
      endLine: number;
      metaStarts: number[];
      metaPairs: Set<string>;
    }> = [];

    for (let i = 0; i < this.tokens.length; i++) {
      const tok = this.tokens[i];
      if (tok.kind !== "item" || !this.blockItems.has(tok.line)) continue;
      const endLine = this.findBlockEnd(i + 1);
      const metaStarts = this.tokens
        .filter(
          (t) =>
            t.kind === "metaStart" &&
            t.line > tok.line &&
            t.line <= endLine &&
            t.indent <= tok.indent + 2
        )
        .map((t) => t.line);
      const metaPairs = this.collectMetaPairs(metaStarts, endLine);
      blocks.push({ itemLine: tok.line, endLine, metaStarts, metaPairs });
    }

    return blocks;
  }

  private findBlockEnd(nextIdx: number): number {
    for (let j = nextIdx; j < this.tokens.length; j++) {
      const next = this.tokens[j];
      if (
        next.kind === "channel" ||
        next.kind === "channelMeta" ||
        (next.kind === "item" && this.blockItems.has(next.line))
      ) {
        return next.line - 1;
      }
    }
    return this.lines.length - 1;
  }

  private collectMetaPairs(metaStartLines: number[], blockEnd: number): Set<string> {
    const keys = new Set<string>();
    if (metaStartLines.length === 0) return keys;
    const boundaries = [...metaStartLines, blockEnd + 1].sort((a, b) => a - b);
    for (let i = 0; i < metaStartLines.length; i++) {
      const start = metaStartLines[i] + 1;
      const end = boundaries[i + 1] - 1;
      for (const tok of this.tokens) {
        if (tok.kind !== "metaPair") continue;
        if (tok.line < start || tok.line > end) continue;
        const key = this.parseKey(this.lines[tok.line]);
        if (key) keys.add(key);
      }
    }
    return keys;
  }

  private parseKey(raw: string | undefined): string | null {
    if (!raw) return null;
    const match = raw.match(/-\s+([^:]+):/);
    return match ? match[1].trim().toLowerCase() : null;
  }

  private defaultMetaBlock(itemLine: number): string[] {
    const baseIndent = indentOf(this.lines[itemLine]) + "  ";
    const childIndent = baseIndent + "  ";
    const out = [`${baseIndent}- [meta]:`];
    if (this.requiredKeys.has("date")) out.push(`${childIndent}- date: ${formatArenaDate()}`);
    if (this.requiredKeys.has("tags")) out.push(`${childIndent}- tags: []`);
    return out;
  }

  private missingPayload(metaStartLine: number, missing: string[]): string[] {
    const childIndent = indentOf(this.lines[metaStartLine]) + "  ";
    return missing.map((key) => {
      if (key === "date") return `${childIndent}- date: ${formatArenaDate()}`;
      if (key === "tags") return `${childIndent}- tags: []`;
      return `${childIndent}- ${key}:`;
    });
  }
}

export default class MetadataValidatorPlugin extends Plugin {
  private writing = new Set<string>();

  async onload() {
    this.registerEvent(
      this.app.vault.on("modify", async (file) => {
        if (!(file instanceof TFile)) return;
        if (file.extension !== "md") return;
        if (this.writing.has(file.path)) {
          this.writing.delete(file.path);
          return;
        }
        await this.handleFile(file);
      })
    );
  }

  private async handleFile(file: TFile) {
    const content = await this.app.vault.read(file);
    const ebnf = extractEbnfFromFrontmatter(content);
    if (!ebnf) return;
    const grammar = detectGrammar(ebnf);
    if (!grammar) return;

    // Pull required keys from the grammar (derived from literals in key/meta rules)
    const metaKeys = deriveRequiredKeys(ebnf, grammar);

    const updated =
      grammar === "stream"
        ? this.enforceStream(content, metaKeys)
        : this.enforceArena(content, metaKeys);
    if (!updated || updated === content) return;

    this.writing.add(file.path);
    await this.app.vault.modify(file, updated);
    new Notice(`metadata normalized for ${file.name}`);
  }

  private enforceStream(text: string, requiredKeys: Set<string>): string | null {
    const endsWithNewline = text.endsWith("\n");
    const lines = text.split(/\r?\n/);
    const fmEnd = findFrontmatterEnd(lines);
    const bodyStart = fmEnd >= 0 ? fmEnd + 1 : 0;
    if (bodyStart >= lines.length) return null;

    const separators: number[] = [];
    for (let i = bodyStart; i < lines.length; i++) {
      if (lines[i].trim() === "---") separators.push(i);
    }

    const sections: Array<{ start: number; end: number }> = [];
    let currentStart = bodyStart;
    for (const sep of [...separators, lines.length]) {
      if (sep < currentStart) continue;
      const end = sep - 1;
      if (currentStart <= end) sections.push({ start: currentStart, end });
      currentStart = sep + 1;
    }

    const modifications: Modification[] = [];
    for (const section of sections) {
      const tokens = lexStream(lines, section.start, section.end);
      const parser = new StreamParser(lines, tokens, requiredKeys);
      modifications.push(...parser.enforce());
    }

    if (modifications.length === 0) return null;

    applyModifications(lines, modifications);

    trimTrailingEmptyLines(lines);
    const updated = lines.join("\n") + (endsWithNewline ? "\n" : "");
    return updated === text ? null : updated;
  }

  private enforceArena(text: string, requiredKeys: Set<string>): string | null {
    const endsWithNewline = text.endsWith("\n");
    const lines = text.split(/\r?\n/);
    const fmEnd = findFrontmatterEnd(lines);
    const bodyStart = fmEnd >= 0 ? fmEnd + 1 : 0;
    if (bodyStart >= lines.length) return null;

    const tokens = lexArena(lines, bodyStart);
    const parser = new ArenaParser(lines, tokens, requiredKeys);
    const modifications = parser.enforce();
    if (modifications.length === 0) return null;

    const preview = [...lines];
    applyModifications(preview, [...modifications]);
    const valid = validateArenaWithEarley(preview, bodyStart);
    if (!valid) console.warn("arena metadata normalization left grammar invalid");

    applyModifications(lines, modifications);

    trimTrailingEmptyLines(lines);
    const updated = lines.join("\n") + (endsWithNewline ? "\n" : "");
    return updated === text ? null : updated;
  }
}
