interface PressKitButtonsProps {
  className?: string
  entryId?: string
  entryUrl?: string
  citation?: boolean
}

export default function PressKitButtons({
  className,
  entryId,
  entryUrl,
  citation = true,
}: PressKitButtonsProps) {
  return (
    <>
      {[
        { tab: 'note', label: 'carte' },
        { tab: 'quote', label: 'citation' },
        { tab: 'links', label: 'liens' },
      ]
        .filter(({ tab }) => citation || tab !== 'quote')
        .map(({ tab, label }) => (
          <button
            key={tab}
            type="button"
            class={className}
            data-press-kit={tab}
            data-press-kit-entry={entryId}
            data-press-kit-url={entryUrl}
            aria-label={entryId ? `${label} pour cette entrée` : undefined}
            aria-haspopup="dialog"
          >
            {label}
          </button>
        ))}
    </>
  )
}
