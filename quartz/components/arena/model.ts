import type { Cmd } from "../../functional"
import { none, of } from "../../functional"

export type SearchScope = "channel" | "index"
export type SearchResultOptions = { focus?: boolean; scroll?: boolean }

export type ArenaModel = {
  ready: boolean
}

export type ArenaEvent =
  | { type: "nav.ready" }
  | { type: "ui.modal.open"; blockId: string }
  | { type: "ui.modal.close" }
  | { type: "ui.block.navigate"; direction: number }
  | { type: "ui.copy"; button: HTMLElement }
  | { type: "ui.search.focus"; prefill?: string }
  | { type: "ui.search.clear"; blur?: boolean }
  | { type: "ui.search.close" }
  | { type: "ui.search.result.activate"; index: number | null; options?: SearchResultOptions }
  | { type: "ui.search.query"; query: string; scope: SearchScope }
  | { type: "search.index.request"; scope: SearchScope }

export type ArenaEffect =
  | { type: "modal.open"; blockId: string }
  | { type: "modal.close" }
  | { type: "block.navigate"; direction: number }
  | { type: "copy"; button: HTMLElement }
  | { type: "search.focus"; prefill?: string }
  | { type: "search.clear"; blur?: boolean }
  | { type: "search.close" }
  | { type: "search.result.activate"; index: number | null; options?: SearchResultOptions }
  | { type: "search.query"; query: string; scope: SearchScope }
  | { type: "search.index.build"; scope: SearchScope }

export const reduce = (
  model: ArenaModel,
  event: ArenaEvent,
): { model: ArenaModel; effects: Cmd<ArenaEffect> } => {
  switch (event.type) {
    case "nav.ready":
      return { model: { ...model, ready: true }, effects: none() }
    case "ui.modal.open":
      return { model, effects: of({ type: "modal.open", blockId: event.blockId }) }
    case "ui.modal.close":
      return { model, effects: of({ type: "modal.close" }) }
    case "ui.block.navigate":
      return { model, effects: of({ type: "block.navigate", direction: event.direction }) }
    case "ui.copy":
      return { model, effects: of({ type: "copy", button: event.button }) }
    case "ui.search.focus":
      return { model, effects: of({ type: "search.focus", prefill: event.prefill }) }
    case "ui.search.clear":
      return { model, effects: of({ type: "search.clear", blur: event.blur }) }
    case "ui.search.close":
      return { model, effects: of({ type: "search.close" }) }
    case "ui.search.result.activate":
      return {
        model,
        effects: of({ type: "search.result.activate", index: event.index, options: event.options }),
      }
    case "ui.search.query":
      return {
        model,
        effects: of({ type: "search.query", query: event.query, scope: event.scope }),
      }
    case "search.index.request":
      return { model, effects: of({ type: "search.index.build", scope: event.scope }) }
  }
}
