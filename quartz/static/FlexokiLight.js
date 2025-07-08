// Flexoki theme for Blink Terminal
// Based on Flexoki color scheme by @kepano (https://github.com/kepano/flexoki)
// Using 400-level accent colors as requested

// Set the 16 color palette for ANSI colors
// Order: [black, red, green, yellow, blue, magenta, cyan, white,
//         lightBlack, lightRed, lightGreen, lightYellow, lightBlue, lightMagenta, lightCyan, lightWhite]
t.prefs_.set("color-palette-overrides", [
  "#100F0F",
  "#D14D41", "#879A39", "#D0A215", "#4385BE", "#CE5D97", "#3AA99F",
  "#9F9D96", "#6F6E69",
  "#D14D41", "#879A39", "#D0A215", "#4385BE", "#CE5D97", "#3AA99F",
  "#FFFCF0",
])
t.prefs_.set("background-color", "#FFFCF0")
t.prefs_.set("foreground-color", "#100F0F")
t.prefs_.set("cursor-color", "#403E3C")
