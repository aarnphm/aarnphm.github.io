import { none, type Cmd } from '../../../functional'

export interface GearRatioModel {
  layout: 1 | 2
  cassetteId: string
  chainringPresetId: string | null
  chainrings: readonly [number, number]
}

export type GearRatioMessage =
  | { type: 'set-layout'; layout: 1 | 2; maximumChainrings: 1 | 2 }
  | { type: 'set-cassette'; cassetteId: string; maximumChainrings: 1 | 2 }
  | { type: 'set-chainring-preset'; presetId: string; chainrings: readonly [number, number] }
  | { type: 'set-chainring'; index: 0 | 1; value: number }

export type GearRatioEffect = { type: 'render' }

export const initialGearRatioModel = (
  cassetteId: string,
  chainringPresetId: string | null,
  firstChainring: number,
  secondChainring: number,
): GearRatioModel => ({
  layout: 2,
  cassetteId,
  chainringPresetId,
  chainrings: [firstChainring, secondChainring],
})

export const updateGearRatio = (
  model: GearRatioModel,
  message: GearRatioMessage,
): { model: GearRatioModel; effects: Cmd<GearRatioEffect> } => {
  switch (message.type) {
    case 'set-layout': {
      const layout = message.layout === 1 || message.maximumChainrings === 1 ? 1 : 2
      return layout === model.layout
        ? { model, effects: none() }
        : { model: { ...model, layout }, effects: [{ type: 'render' }] }
    }
    case 'set-cassette': {
      const layout = message.maximumChainrings === 1 ? 1 : model.layout
      return {
        model: { ...model, cassetteId: message.cassetteId, layout },
        effects: [{ type: 'render' }],
      }
    }
    case 'set-chainring-preset':
      return {
        model: {
          ...model,
          layout: 2,
          chainringPresetId: message.presetId,
          chainrings: message.chainrings,
        },
        effects: [{ type: 'render' }],
      }
    case 'set-chainring': {
      const chainrings: [number, number] = [...model.chainrings]
      chainrings[message.index] = message.value
      return {
        model: { ...model, chainringPresetId: null, chainrings },
        effects: [{ type: 'render' }],
      }
    }
  }
}
