import { none, type Cmd } from '../../../functional'

export interface DatePickerModel {
  open: boolean
  selected: string | undefined
  viewMonth: string
}

export type DatePickerMessage =
  | { type: 'open'; selected: string | undefined; viewMonth: string }
  | { type: 'sync'; selected: string | undefined }
  | { type: 'close'; restoreFocus?: boolean }
  | { type: 'move-month'; viewMonth: string }
  | { type: 'focus-date'; date: string; viewMonth: string }
  | { type: 'select'; date: string }
  | { type: 'clear' }

export type DatePickerEffect =
  | { type: 'render'; focusDate?: string }
  | { type: 'close-panel'; restoreFocus: boolean }
  | { type: 'notify-select'; date: string }
  | { type: 'notify-clear' }

export const initialDatePickerModel = (): DatePickerModel => ({
  open: false,
  selected: undefined,
  viewMonth: '',
})

export const updateDatePicker = (
  model: DatePickerModel,
  message: DatePickerMessage,
): { model: DatePickerModel; effects: Cmd<DatePickerEffect> } => {
  switch (message.type) {
    case 'open':
      return {
        model: { open: true, selected: message.selected, viewMonth: message.viewMonth },
        effects: [{ type: 'render' }],
      }
    case 'close':
      return model.open
        ? {
            model: { ...model, open: false },
            effects: [{ type: 'close-panel', restoreFocus: message.restoreFocus ?? false }],
          }
        : { model, effects: none() }
    case 'sync':
      return { model: { ...model, selected: message.selected }, effects: [{ type: 'render' }] }
    case 'move-month':
      return { model: { ...model, viewMonth: message.viewMonth }, effects: [{ type: 'render' }] }
    case 'focus-date':
      return {
        model: { ...model, viewMonth: message.viewMonth },
        effects: [{ type: 'render', focusDate: message.date }],
      }
    case 'select':
      return {
        model: { ...model, open: false, selected: message.date },
        effects: [
          { type: 'notify-select', date: message.date },
          { type: 'close-panel', restoreFocus: false },
        ],
      }
    case 'clear':
      return {
        model: { ...model, open: false, selected: undefined },
        effects: [{ type: 'notify-clear' }, { type: 'close-panel', restoreFocus: false }],
      }
  }
}
