from pathlib import Path

import nltk
import svgling
from nltk import CFG

G0_PRIME = CFG.fromstring("""
    S -> NP VP
    NP -> PN | D N
    VP -> V | V NP | VP PP
    PP -> P NP
    PN -> 'Kevin' | 'Dave'
    D -> 'a' | 'the'
    N -> 'banana' | 'apple' | 'park' | 'child'
    V -> 'eats' | 'runs'
    P -> 'in' | 'on'
""")


def parse_and_draw(grammar: CFG, sentence: str, output_path: Path | None = None) -> str:
  tokens = sentence.split()
  parser = nltk.ChartParser(grammar)

  for tree in parser.parse(tokens):
    svg = svgling.draw_tree(tree)
    svg_content = svg._repr_svg_()

    if output_path:
      output_path.parent.mkdir(parents=True, exist_ok=True)
      output_path.write_text(svg_content)

    return svg_content

  raise ValueError(f'no parse found for: {sentence}')


if __name__ == '__main__':
  here = Path(__file__).parent
  svg = parse_and_draw(G0_PRIME, 'the child eats a banana in the park', here / 'img' / 'parse_tree_a1.svg')
