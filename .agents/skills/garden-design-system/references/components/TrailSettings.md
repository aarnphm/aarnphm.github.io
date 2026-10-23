# TrailSettings

# TrailSettings

The settings panel of the home page trail. It opens from the boulder sign in the trail scene. It holds the season and holiday buttons, then five sections: animation, forest, animals, campsites and people.

**Source:** `quartz/components/visitor-trail/scene-controls.ts` (`mountSceneControls`: sections, sliders, choices, reset) and `season-toggle.ts` (the panel shell, the season and scene buttons, the popover) · defaults and limits: `visitor-trail/scene-settings.ts` · styles: `quartz/components/styles/trail-scene-controls.scss`

## Preview

A copy of the live panel from `/`, taken from the trail's shadow root with the trail open (the panel is `.trail-scene-controls`, a `popover="auto"` in the top layer). The preview keeps the real height (560px) and scrolls like the site. It drops the fixed position and the popover attribute, and carries its own rules because the site draws the panel inside the trail's shadow root.

## Rows

Every control is a row of one list. Rows are separated by a 1px `lightgray` rule. A section is a `section` whose `h3` is a full-width band.

| Row    | Left                                               | Value cell                                              | Below                                                                  |
| ------ | -------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------- |
| Button | label (`season`, `scene`)                          | current value, plain text                               | nothing                                                                |
| Slider | label                                              | `output`, 1px `darkgray` box, right-aligned, 3.5em wide | range input, 4px track filled to the value                             |
| Choice | `legend`                                           | nothing                                                 | one radio per option in a row of equal boxes; the chosen box is tinted |
| Action | label (`reset tuning`, `pack up the visited camp`) | nothing                                                 | nothing                                                                |

## Rules

- The panel is 244px wide and at most 560px high. Its font is the page code font at `.66rem`. Corners are square and there is no shadow.
- A slider has no description line. The unit sits in the value cell (`%`, a degree sign) and in `aria-valuetext`.
- The forest sliders change the tree shape, so they apply on release. The panel says so in one note line, and the other sliders apply after a 180ms pause.
- A choice is a native `fieldset` of radios. The radio is visually hidden and the label box shows the state, so keyboard focus is a 2px `dark` ring inside the box.
- `reset tuning` stays disabled and loses its underline until a value differs from the default.
- An action that cannot be undone from the panel (`pack up the visited camp`) is a button.
- Errors and status go in one visually hidden `role="status"` region. Escape closes the panel and returns focus to the sign. Arrow, Home, End and Page keys stay inside the controls.
