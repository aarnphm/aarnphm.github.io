"""Paths, kept features, OFL renames and the FACES table."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]  # harness/fonts/subset/faces.py
STATIC = ROOT / 'quartz' / 'static'
CACHE = ROOT / 'quartz' / '.quartz-cache' / 'fonts'
SRC_CACHE = CACHE / 'src'
OUT_KV = CACHE / 'out'
STYLES = ROOT / 'quartz' / 'styles'
KV_NAMESPACE = 'aec7fd95ca404e8c9941993b0e65821d'
# The worker 403s font requests from other referers, and the Cloudflare edge 403s the default
# python-urllib user agent, so every fetch of the live site sends both.
FETCH_HEADERS = {
  'Referer': 'https://aarnphm.xyz/',
  'User-Agent': 'garden-fonts-subset (+https://aarnphm.xyz)',
}

# Features browsers apply by default plus the ones the stylesheets request: tnum/lnum/onum via
# font-variant-numeric and font-feature-settings, zero via slashed-zero (content/park.html).
KEEP_FEATURES = [
  'ccmp',
  'locl',
  'mark',
  'mkmk',
  'calt',
  'clig',
  'liga',
  'rlig',
  'rclt',
  'curs',
  'dist',
  'kern',
  'tnum',
  'lnum',
  'onum',
  'zero',
]

# Share of pages a character must appear on to sit in the preloaded core file. An ext file
# costs about 20 KB whenever one of its characters shows up, a core glyph about 60 B on every
# page, so the break-even is roughly 0.3% of pages (10 of 3370); below that the ext fetch is
# cheaper on average.
CORE_PAGE_SHARE = 0.003

# OFL fonts with a Reserved Font Name must not carry it once modified. CSS keeps the old
# font-family strings, since those name the @font-face rule, not the file's internal name.
RENAMES = {
  'Space Grotesk': 'Garden Grotesk',
  'Playfair Display': 'Garden Display Serif',
  'Source Serif 4': 'Garden Text Serif',
  'Departure Mono': 'Garden Pixel Mono',
}


@dataclass
class Face:
  family: str  # CSS font-family (unchanged)
  source: str  # original file name in quartz/static or the FONTS KV
  origin: str  # where the original comes from: 'static' (quartz/static) or 'kv'; outputs always go to KV
  weight: str  # CSS font-weight descriptor exactly as fonts.scss declared it
  style: str = 'normal'
  pin: dict[str, float] = field(
    default_factory=dict
  )  # variable axes to instance
  rename: str | None = None
  enabled: bool = True
  note: str = ''

  @property
  def stem(self) -> str:
    return re.sub(r'\.(woff2?|ttf|otf)$', '', self.source)


# One row per @font-face the site declared before subsetting. `enabled=False` rows are faces no
# rule or markup can request (see docs/verification/font-subset/README.md); they are kept here so
# re-enabling one is a flag flip.
FACES: list[Face] = [
  Face(
    'PP Neue Montreal', 'PPNeueMontreal-Thin.woff2', 'kv', '100', enabled=False
  ),
  Face(
    'PP Neue Montreal',
    'PPNeueMontreal-ThinItalic.woff2',
    'kv',
    '100',
    'italic',
    enabled=False,
  ),
  Face('PP Neue Montreal', 'PPNeueMontreal-Light.woff2', 'kv', '300'),
  Face(
    'PP Neue Montreal',
    'PPNeueMontreal-LightItalic.woff2',
    'kv',
    '300',
    'italic',
  ),
  Face(
    'PP Neue Montreal', 'PPNeueMontreal-Book.woff2', 'kv', '350', enabled=False
  ),
  Face(
    'PP Neue Montreal',
    'PPNeueMontreal-BookItalic.woff2',
    'kv',
    '350',
    'italic',
    enabled=False,
  ),
  Face('PP Neue Montreal', 'PPNeueMontreal-Regular.woff2', 'kv', '400'),
  Face(
    'PP Neue Montreal', 'PPNeueMontreal-Italic.woff2', 'kv', '400', 'italic'
  ),
  Face('PP Neue Montreal', 'PPNeueMontreal-Medium.woff2', 'kv', '500'),
  Face(
    'PP Neue Montreal',
    'PPNeueMontreal-MediumItalic.woff2',
    'kv',
    '500',
    'italic',
  ),
  Face('PP Neue Montreal', 'PPNeueMontreal-SemiBold.woff2', 'kv', '600'),
  Face(
    'PP Neue Montreal',
    'PPNeueMontreal-SemiBolditalic.woff2',
    'kv',
    '600',
    'italic',
  ),
  Face('PP Neue Montreal', 'PPNeueMontreal-Bold.woff2', 'kv', '700'),
  Face(
    'PP Neue Montreal',
    'PPNeueMontreal-BoldItalic.woff2',
    'kv',
    '700',
    'italic',
  ),
  Face(
    'Berkeley Mono', 'TX-02-N14XWV3W-Thin.woff2', 'kv', '100', enabled=False
  ),
  Face(
    'Berkeley Mono',
    'TX-02-Thin-Oblique.woff2',
    'kv',
    '100',
    'italic',
    enabled=False,
  ),
  Face('Berkeley Mono', 'TX-02-ExtraLight.woff2', 'kv', '200', enabled=False),
  Face(
    'Berkeley Mono',
    'TX-02-ExtraLight-Oblique.woff2',
    'kv',
    '200',
    'italic',
    enabled=False,
  ),
  Face('Berkeley Mono', 'TX-02-Light.woff2', 'kv', '300', enabled=False),
  Face(
    'Berkeley Mono',
    'TX-02-Light-Oblique.woff2',
    'kv',
    '300',
    'italic',
    enabled=False,
  ),
  Face('Berkeley Mono', 'TX-02-SemiLight.woff2', 'kv', '350', enabled=False),
  Face(
    'Berkeley Mono',
    'TX-02-SemiLight-Oblique.woff2',
    'kv',
    '350',
    'italic',
    enabled=False,
  ),
  Face('Berkeley Mono', 'TX-02-Book.woff2', 'kv', '375', enabled=False),
  Face(
    'Berkeley Mono',
    'TX-02-Book-Oblique.woff2',
    'kv',
    '375',
    'italic',
    enabled=False,
  ),
  Face('Berkeley Mono', 'TX-02-Regular.woff2', 'kv', '400'),
  Face('Berkeley Mono', 'TX-02-Oblique.woff2', 'kv', '400', 'italic'),
  # 450 is only hit mid-transition (font-weight 400→600 tweens in kvHeadGrouping/razorCompression).
  Face('Berkeley Mono', 'TX-02-Retina.woff2', 'kv', '450'),
  Face(
    'Berkeley Mono',
    'TX-02-Retina-Oblique.woff2',
    'kv',
    '450',
    'italic',
    enabled=False,
  ),
  Face('Berkeley Mono', 'TX-02-Medium.woff2', 'kv', '500'),
  Face('Berkeley Mono', 'TX-02-Medium-Oblique.woff2', 'kv', '500', 'italic'),
  Face('Berkeley Mono', 'TX-02-SemiBold.woff2', 'kv', '600'),
  Face('Berkeley Mono', 'TX-02-SemiBold-Oblique.woff2', 'kv', '600', 'italic'),
  Face('Berkeley Mono', 'TX-02-Bold.woff2', 'kv', '700'),
  Face('Berkeley Mono', 'TX-02-Bold-Oblique.woff2', 'kv', '700', 'italic'),
  Face('Berkeley Mono', 'TX-02-ExtraBold.woff2', 'kv', '800', enabled=False),
  Face(
    'Berkeley Mono',
    'TX-02-ExtraBold-Oblique.woff2',
    'kv',
    '800',
    'italic',
    enabled=False,
  ),
  Face('Berkeley Mono', 'TX-02-Black.woff2', 'kv', '900', enabled=False),
  Face(
    'Berkeley Mono',
    'TX-02-Black-Oblique.woff2',
    'kv',
    '900',
    'italic',
    enabled=False,
  ),
  Face(
    'Parclo Serif', 'ParcloSerifStudent-Thin.woff2', 'kv', '100', enabled=False
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-ThinItalic.woff2',
    'kv',
    '100',
    'italic',
    enabled=False,
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-Light.woff2',
    'kv',
    '300',
    enabled=False,
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-LightItalic.woff2',
    'kv',
    '300',
    'italic',
    enabled=False,
  ),
  Face('Parclo Serif', 'ParcloSerifStudent-Regular.woff2', 'kv', '400'),
  Face(
    'Parclo Serif', 'ParcloSerifStudent-Italic.woff2', 'kv', '400', 'italic'
  ),
  Face('Parclo Serif', 'ParcloSerifStudent-Medium.woff2', 'kv', '500'),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-MediumItalic.woff2',
    'kv',
    '500',
    'italic',
  ),
  Face('Parclo Serif', 'ParcloSerifStudent-SemiBold.woff2', 'kv', '600'),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-SemiBoldItalic.woff2',
    'kv',
    '600',
    'italic',
  ),
  Face('Parclo Serif', 'ParcloSerifStudent-Bold.woff2', 'kv', '700'),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-BoldItalic.woff2',
    'kv',
    '700',
    'italic',
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-Black.woff2',
    'kv',
    '800',
    enabled=False,
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-BlackItalic.woff2',
    'kv',
    '800',
    'italic',
    enabled=False,
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-ExtraBlack.woff2',
    'kv',
    '900',
    enabled=False,
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-ExtraBlackItalic.woff2',
    'kv',
    '900',
    'italic',
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-Ultra.woff2',
    'kv',
    '950',
    enabled=False,
  ),
  Face(
    'Parclo Serif',
    'ParcloSerifStudent-UltraItalic.woff2',
    'kv',
    '950',
    'italic',
    enabled=False,
  ),
  Face(
    'Source Serif 4',
    'source-serif-4-Regular.woff2',
    'kv',
    '400',
    rename=RENAMES['Source Serif 4'],
  ),
  Face(
    'Source Serif 4',
    'source-serif-4-It.woff2',
    'kv',
    '400',
    'italic',
    rename=RENAMES['Source Serif 4'],
  ),
  Face(
    'Source Serif 4',
    'source-serif-4-Semibold.woff2',
    'kv',
    '600',
    rename=RENAMES['Source Serif 4'],
  ),
  Face(
    'Source Serif 4',
    'source-serif-4-SemiboldIt.woff2',
    'kv',
    '600',
    'italic',
    rename=RENAMES['Source Serif 4'],
  ),
  # Declared at a single weight, so the browser clamps every heading to wght 400. The default
  # master is 300, so a static 400 instance rounds every interpolated coordinate and moves
  # heading pixels; the file stays variable (subset of glyphs only).
  Face(
    'Space Groteskque',
    'SpaceGrotesk-VariableFont_wght.ttf',
    'static',
    'normal',
    rename=RENAMES['Space Grotesk'],
  ),
  Face(
    'Playfair Display',
    'PlayfairDisplay.woff2',
    'static',
    'normal',
    pin={'wght': 400},
    rename=RENAMES['Playfair Display'],
  ),
  Face(
    'ITCGaramondStdLtCond', 'ITCGaramondStdLtCond.woff2', 'static', 'normal'
  ),
  Face('GT Sectra Book', 'GT-Sectra-Book.woff', 'static', 'normal'),
  Face(
    'GT Sectra Display',
    'GT-Sectra-Display-Regular.woff',
    'static',
    'normal',
    enabled=False,
    note='no stylesheet rule; og.tsx reads the original file at build time',
  ),
  Face(
    'EditorialNew Ultralight',
    'EditorialNew-Ultralight.woff2',
    'static',
    'normal',
    enabled=False,
    note='no reference outside fonts.scss',
  ),
  Face(
    'ABC Diatype Light',
    'ABCDiatype-Light.woff2',
    'static',
    'normal',
    enabled=False,
    note='no reference outside fonts.scss',
  ),
  Face(
    'Departure Mono',
    'DepartureMono-Regular.woff2',
    'static',
    'normal',
    rename=RENAMES['Departure Mono'],
  ),
]

PRELOAD = [
  'PPNeueMontreal-Regular',
  'SpaceGrotesk-VariableFont_wght',
  'TX-02-Regular',
]


# ---------------------------------------------------------------------------------------------
# Character census
