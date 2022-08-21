import { Translation, CalloutTranslation } from "./locales/definition"
import enUs from "./locales/en-US"
import enGb from "./locales/en-GB"
import fr from "./locales/fr-FR"
import it from "./locales/it-IT"
import vi from "./locales/vi-VN"

export const TRANSLATIONS = {
  "en-US": enUs,
  "en-GB": enGb,
  "fr-FR": fr,
  "it-IT": it,
  "vi-VN": vi,
} as const

export const defaultTranslation = "en-US"
export const i18n = (locale: ValidLocale): Translation => TRANSLATIONS[locale ?? defaultTranslation]
export type ValidLocale = keyof typeof TRANSLATIONS
export type ValidCallout = keyof CalloutTranslation
