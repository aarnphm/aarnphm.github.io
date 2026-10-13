import type { ValidLocale } from '../i18n'
import { i18n } from '../i18n'
import style from './styles/disabled-plugin-notice.scss'

export const disabledPluginNoticeStyle = style

type Props = {
  label: string
  locale: ValidLocale
  setting?: string
  source?: string
  homeHref?: string
}

export function DisabledPluginNotice({ label, locale, setting, source, homeHref }: Props) {
  const text = i18n(locale).components.disabledPlugin
  const inline = source !== undefined
  return (
    <section
      class={`disabled-plugin-notice${inline ? ' disabled-plugin-notice-inline' : ''}`}
      data-disabled-plugin={label}
      aria-label={label}
    >
      {inline ? (
        <p class="disabled-plugin-title">
          <strong>{label}</strong>
        </p>
      ) : (
        <h1 class="disabled-plugin-title">{label}</h1>
      )}
      <p>{text.message}</p>
      <p class="disabled-plugin-help">
        {setting ? text.enableSetting({ setting }) : text.enablePlugin}
      </p>
      {inline && (
        <pre class="disabled-plugin-source">
          <code>{source}</code>
        </pre>
      )}
      {homeHref && (
        <a class="disabled-plugin-home internal" href={homeHref}>
          {text.returnHome}
        </a>
      )}
    </section>
  )
}
