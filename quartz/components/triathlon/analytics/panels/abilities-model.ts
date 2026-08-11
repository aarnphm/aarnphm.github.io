import type { Sport } from '../../../../plugins/stores/strava'

export interface AbilitiesModel {
  average: boolean
  sports: readonly Sport[]
}

export type AbilitiesMessage =
  | { type: 'restore'; model: AbilitiesModel }
  | { type: 'toggle-average' }
  | { type: 'toggle-sport'; sport: Sport }

export const initialAbilitiesModel = (available: readonly Sport[]): AbilitiesModel => ({
  average: false,
  sports: available.length === 0 ? [] : [available.includes('bike') ? 'bike' : available[0]],
})

export const updateAbilities = (
  model: AbilitiesModel,
  message: AbilitiesMessage,
  available: readonly Sport[],
): AbilitiesModel => {
  if (message.type === 'restore') {
    const sports = message.model.sports.filter(
      (sport, index, values) => available.includes(sport) && values.indexOf(sport) === index,
    )
    return {
      average: message.model.average,
      sports: message.model.sports.length === 0 || sports.length > 0 ? sports : model.sports,
    }
  }
  if (message.type === 'toggle-average') return { ...model, average: !model.average }
  if (!available.includes(message.sport)) return model
  if (model.average) return { average: false, sports: [message.sport] }
  return model.sports.includes(message.sport)
    ? { ...model, sports: model.sports.filter(sport => sport !== message.sport) }
    : { ...model, sports: [...model.sports, message.sport] }
}
