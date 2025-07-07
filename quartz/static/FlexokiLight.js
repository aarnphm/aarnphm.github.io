// Flexoki theme for Blink Terminal
// Based on Flexoki color scheme by @kepano (https://github.com/kepano/flexoki)
// Using 400-level accent colors as requested

// Flexoki color definitions
const flexoki = {
// Base colors
black: ‘#100F0F’,
paper: ‘#FFFCF0’,
base400: ‘#9F9D96’,
base600: ‘#6F6E69’,
base800: ‘#403E3C’,

// Accent colors (400-level)
red400: ‘#D14D41’,
orange400: ‘#DA702C’,
yellow400: ‘#D0A215’,
green400: ‘#879A39’,
cyan400: ‘#3AA99F’,
blue400: ‘#4385BE’,
purple400: ‘#8B7EC8’,
magenta400: ‘#CE5D97’
};

// Set the 16 color palette for ANSI colors
// Order: [black, red, green, yellow, blue, magenta, cyan, white,
//         lightBlack, lightRed, lightGreen, lightYellow, lightBlue, lightMagenta, lightCyan, lightWhite]
t.prefs_.set(‘color-palette-overrides’, [
flexoki.black,        // 0: black
flexoki.red400,       // 1: red
flexoki.green400,     // 2: green
flexoki.yellow400,    // 3: yellow
flexoki.blue400,      // 4: blue
flexoki.magenta400,   // 5: magenta
flexoki.cyan400,      // 6: cyan
flexoki.base400,      // 7: white (light gray)
flexoki.base600,      // 8: light black (dark gray)
flexoki.red400,       // 9: light red
flexoki.green400,     // 10: light green
flexoki.yellow400,    // 11: light yellow
flexoki.blue400,      // 12: light blue
flexoki.magenta400,   // 13: light magenta
flexoki.cyan400,      // 14: light cyan
flexoki.paper         // 15: light white (paper)
]);

// Light theme variant (default)
t.prefs_.set(‘background-color’, flexoki.paper);
t.prefs_.set(‘foreground-color’, flexoki.black);
t.prefs_.set(‘cursor-color’, flexoki.base800);

// Optional: Uncomment the following lines for dark theme variant
// t.prefs_.set(‘background-color’, flexoki.black);
// t.prefs_.set(‘foreground-color’, flexoki.paper);
// t.prefs_.set(‘cursor-color’, flexoki.base400);
