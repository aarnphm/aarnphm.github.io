"""The fixed workloads. Each scenario is a page path, a ready predicate (a JS expression), the ms to
settle after it, and a script body that runs inside the page with the PRELUDE helpers in scope. A
hill climb may add a scenario; it must not change an existing one, or its baseline stops meaning
anything. `node`-free: these strings go to the page through the CUA broker (see bench.py)."""

from workloads.bases import cards, kanban, places_map, table, vault
from workloads.bookcase import bookcase, bookcase_preview, bookcase_search
from workloads.prelude import HELPERS, PRELUDE
from workloads.triathlon import (
  tri_analytics,
  tri_calendar,
  tri_day,
  tri_hub,
  tri_maps,
  tri_training,
)

__all__ = ['GROUPS', 'HELPERS', 'PRELUDE', 'SCENARIOS']

SCENARIOS = {
  'vault': vault,
  'vault-directors': {**vault, 'path': '/cinematheque/directors'},
  'cards-films': cards('/cinematheque/gallery'),
  'kanban': kanban,
  'bookcase': bookcase,
  'bookcase-preview': bookcase_preview,
  'bookcase-search': bookcase_search,
  'cards-books': cards('/antilibrary/gallery'),
  'table-to-read': table('/antilibrary/to-read'),
  'table-ideas': table('/ideas'),
  'places-map': places_map,
  'tri-hub': tri_hub,
  'tri-day': tri_day,
  'tri-analytics': tri_analytics,
  'tri-calendar': tri_calendar,
  'tri-training': tri_training,
  'tri-maps': tri_maps,
}

# Worker groups: each climb owns a set of scenarios and the files behind them.
GROUPS = {
  'vault': ['vault', 'vault-directors', 'cards-films', 'kanban'],
  'views': [
    'bookcase',
    'bookcase-preview',
    'bookcase-search',
    'cards-books',
    'table-to-read',
    'table-ideas',
  ],
  'triathlon': [
    'tri-hub',
    'tri-day',
    'tri-analytics',
    'tri-calendar',
    'tri-training',
  ],
  'maps': ['tri-maps', 'places-map'],
}
