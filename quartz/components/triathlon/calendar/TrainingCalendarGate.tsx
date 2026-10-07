import type { TriathlonFormatter } from '../runtime/formatter'
import { TRAINING_CALENDAR_SESSION_API } from '../../../util/training-calendar-access'
import { DEFAULT_TRIATHLON_PRESENTATION } from '../../../util/triathlon-presentation'
import { createTriathlonFormatter } from '../runtime/formatter'

export const TrainingCalendarGate = ({
  id,
  busy = false,
  message = '',
  formatter = createTriathlonFormatter(DEFAULT_TRIATHLON_PRESENTATION),
}: {
  id: string
  busy?: boolean
  message?: string
  formatter?: TriathlonFormatter
}) => {
  const { text } = formatter
  return (
    <section class="tri-training-gate" aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`}>{text('training calendar')}</h2>
      <p>{text('Enter the password to view the TrainingPeaks calendar.')}</p>
      <form
        data-training-login
        method="post"
        action={TRAINING_CALENDAR_SESSION_API}
        aria-busy={busy}
      >
        <label for={`${id}-password`}>{text('calendar password')}</label>
        <div class="tri-training-gate-controls">
          <input
            id={`${id}-password`}
            name="password"
            type="password"
            autoComplete="current-password"
            required
            maxLength={256}
            aria-describedby={`${id}-status`}
            disabled={busy}
          />
          <button type="submit" disabled={busy}>
            {text(busy ? 'unlocking…' : 'unlock calendar')}
          </button>
        </div>
        <p id={`${id}-status`} class="tri-training-gate-status" role="status" aria-live="polite">
          {text(message)}
        </p>
        {message === 'Calendar hidden. Could not end the session; retry locking.' && (
          <button type="button" data-training-lock disabled={busy}>
            {text('lock calendar')}
          </button>
        )}
      </form>
    </section>
  )
}
