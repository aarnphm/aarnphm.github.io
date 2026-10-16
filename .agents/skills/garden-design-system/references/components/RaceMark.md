# RaceMark

A race-series mark for the race calendar: the series fill beside an event name in the series ink.

**Source:** `triathlon-calendar.scss` (`[data-series]` with `--calendar-event`, `--calendar-event-ink`, `--calendar-event-on`). The event names in the preview are samples.

## Inputs

- The series key and the event name.

## Rules

- Each series has three colours: the fill (`race-*`) for marks, a label ink (`race-*-ink`) that reads on paper in each theme, and a fixed colour for text on the fill (white for IRONMAN and TCS; the series navy or black for T100, Supertri and HYROX).
- These are other organisations' brand colours. Use them only to identify those series.
