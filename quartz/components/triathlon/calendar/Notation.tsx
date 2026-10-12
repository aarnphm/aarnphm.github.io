import type { Locale } from '../../../util/triathlon-presentation'
import { triText } from '../../../util/triathlon-i18n'

type NotationCard = { term: string; meaning: string; example: string }
type NotationSport = 'swim' | 'bike' | 'run'

const CARDS: Record<NotationSport, NotationCard[]> = {
  swim: [
    {
      term: '4 × 100 m @ 1:45',
      meaning:
        'Four 100 m repeats, starting every 1 min 45 s. Swimming and rest share that interval.',
      example: 'Finish in 1:30 → rest 15 s → start the next 100.',
    },
    {
      term: 'RI :20 · r = 20 s',
      meaning: 'Rest interval: take 20 seconds after each repeat, whatever your swim time.',
      example: '4 × 100 m, RI :20 → 400 m of swimming, with 20 s rests.',
    },
    {
      term: 'FR / f/s · BK · BR · FL',
      meaning: 'Freestyle (usually front crawl), backstroke, breaststroke, butterfly.',
      example: '4 × 50 m FR → four 50 m freestyle repeats.',
    },
    {
      term: 'IM · IMO',
      meaning:
        'Individual medley: butterfly, backstroke, breaststroke, freestyle. IMO follows that order across repeats.',
      example: '100 m IM → 25 m of each stroke. 4 × 100 m IMO → one stroke per 100.',
    },
    {
      term: 'desc · build',
      meaning: 'Descend: each repeat gets faster. Build: increase speed within one repeat.',
      example: '4 × 100 desc 1–4 → four successively faster 100s.',
    },
    {
      term: 'pull · kick · drill',
      meaning:
        'Pull focuses on the arms; kick focuses on the legs; a drill isolates a skill. Use the equipment named by the coach.',
      example: '4 × 50 m (25 drill / 25 swim) → each 50 combines both.',
    },
    {
      term: '2 × (4 × 50 m)',
      meaning:
        'Repeat the whole bracket twice. Keep the specified rest between repeats and between rounds.',
      example: '2 × 4 × 50 m = 400 m. The brackets alone do not specify rest.',
    },
    {
      term: 'm · yd · /100 m',
      meaning:
        'Read the distance unit and pool length. Swim pace is often time per 100 metres or 100 yards.',
      example: '1:45/100 m → 105 seconds per 100 metres.',
    },
  ],
  bike: [
    {
      term: '3 × (8 min @ 90% FTP / 4 min EZ)',
      meaning: 'Repeat eight minutes at the stated power, then four minutes easy, three times.',
      example: '3 × (8 + 4) = 36 min, including the final easy block.',
    },
    {
      term: 'W · FTP · %FTP',
      meaning:
        'Watts measure power. Functional threshold power is the reference for percentage-based targets.',
      example: '90% of a 250 W FTP = 225 W. Use your current cycling FTP.',
    },
    {
      term: 'rpm',
      meaning: 'Cadence in pedal revolutions per minute, separate from the power target.',
      example: '90 rpm → 90 complete pedal revolutions in one minute.',
    },
    {
      term: 'Z1 · Z2 · Z3…',
      meaning: 'Training zones depend on the sport, metric, and zone system in the plan.',
      example: 'Z2 power and Z2 heart rate have different units and boundaries.',
    },
    {
      term: 'WU · MS · CD · EZ',
      meaning:
        'Warm-up, main set, cool-down, easy. Recovery between efforts may be easy pedalling.',
      example: '10 min WU + 30 min MS + 5 min CD = 45 min.',
    },
    {
      term: 'HR · bpm · RPE',
      meaning:
        'Heart rate in beats per minute; rating of perceived exertion describes effort on the scale named by the plan.',
      example: 'RPE 6/10 uses a different scale from RPE 6–20.',
    },
  ],
  run: [
    {
      term: '6 × (400 m / 200 m jog)',
      meaning: 'Run 400 metres at the prescribed effort, then jog 200 metres; repeat six times.',
      example: '2.4 km of work + 1.2 km of recovery = 3.6 km.',
    },
    {
      term: 'min/km · min/mi',
      meaning:
        'Pace is the time to cover one kilometre or mile. Keep the written unit when reading the target.',
      example: '5:00/km → five minutes per kilometre.',
    },
    {
      term: '4 × (3′ / 90″ jog)',
      meaning:
        'A single prime denotes minutes; a double prime denotes seconds. Here each effort has a timed jogging recovery.',
      example: '4 × (3 min + 90 s) = 18 min.',
    },
    {
      term: 'strides',
      meaning:
        'Short, controlled accelerations with recovery between efforts. Follow the duration and recovery in the plan.',
      example: '6 × (20 s strides / 60 s easy) → six accelerations.',
    },
    {
      term: 'tempo · threshold · Z2',
      meaning:
        'These labels depend on the coach’s system. Use the prescribed pace, heart rate, or effort for this run.',
      example: '20 min tempo → check the target in the workout details.',
    },
    {
      term: 'WU · CD · rec · EZ',
      meaning:
        'Warm-up, cool-down, recovery, easy. Recovery can be a jog, walk, or rest as specified.',
      example: '5 × 1 km, 2 min jog rec → jog for two minutes between efforts.',
    },
  ],
}

const SOURCES: Record<NotationSport, { label: string; href: string }[]> = {
  swim: [
    {
      label: 'r/Swimming · notation',
      href: 'https://www.reddit.com/r/Swimming/comments/elk0n/swimming_notation_understanding_a_swim_training/',
    },
    {
      label: 'U.S. Masters Swimming · training',
      href: 'https://www.usms.org/fitness-and-training/guides/swimming-101/adult-swim-training',
    },
  ],
  bike: [
    {
      label: 'TrainingPeaks · glossary',
      href: 'https://help.trainingpeaks.com/hc/en-us/articles/115001271712-Glossary',
    },
    {
      label: 'TrainingPeaks · power',
      href: 'https://www.trainingpeaks.com/blog/how-to-get-started-training-with-power/',
    },
  ],
  run: [
    {
      label: 'TrainingPeaks · speedwork',
      href: 'https://www.trainingpeaks.com/blog/getting-up-to-speedwork/',
    },
    {
      label: 'TrainingPeaks · zones',
      href: 'https://help.trainingpeaks.com/hc/en-us/articles/360017420092-Zones-Calculator-Overview',
    },
  ],
}

export const NotationIcon = () => (
  <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
    <path d="M8 3.5C6.5 2.5 4 2 1.5 2.5v10C4 12 6.5 12.5 8 13.5m0-10c1.5-1 4-1.5 6.5-1v10C12 12 9.5 12.5 8 13.5m0-10v10" />
  </svg>
)

export const Notation = ({ id, locale = 'en' }: { id: string; locale?: Locale }) => {
  const copy = (key: string) => <span data-i18n={key}>{triText(locale, key)}</span>
  const sports: NotationSport[] = ['swim', 'bike', 'run']
  return (
    <section class="tri-notation" aria-labelledby={`${id}-title`}>
      <header class="tri-notation-header">
        <h2 id={`${id}-title`}>{copy('workout notation')}</h2>
        <p>
          {copy(
            'Read the set, then the effort and recovery. Examples explain the notation; follow the targets and units in your plan.',
          )}
        </p>
      </header>
      <fieldset class="tri-notation-categories">
        <legend class="sr-only">{copy('sport')}</legend>
        {sports.map(sport => (
          <label data-sport={sport} key={sport}>
            <input
              type="radio"
              name={`${id}-sport`}
              value={sport}
              defaultChecked={sport === 'swim'}
              aria-controls={`${id}-${sport}`}
            />
            {copy(sport)}
          </label>
        ))}
      </fieldset>
      {sports.map(sport => (
        <section
          key={sport}
          id={`${id}-${sport}`}
          data-notation-sport={sport}
          aria-label={triText(locale, sport)}
          data-i18n-aria-label={sport}
        >
          <dl class="tri-notation-grid">
            {CARDS[sport].map(card => (
              <div class="tri-notation-card" key={card.term}>
                <dt>
                  <code>{card.term}</code>
                </dt>
                <dd>{copy(card.meaning)}</dd>
                <dd class="tri-notation-example">{copy(card.example)}</dd>
              </div>
            ))}
          </dl>
          <footer class="tri-notation-sources">
            {SOURCES[sport].map(source => (
              <a
                key={source.href}
                href={source.href}
                target="_blank"
                rel="noopener noreferrer"
                data-no-popover
                data-router-ignore
              >
                {source.label} ↗
              </a>
            ))}
          </footer>
        </section>
      ))}
    </section>
  )
}
